"""
Clinical Trials Hub Evaluation System

Evaluates model predictions against ground truth clinical trial data.
Supports semantic similarity scoring via API judge model.

Key Features:
- Hungarian algorithm for optimal list item alignment
- Hierarchical field flattening and comparison
- TP/FP/FN/EV categorization for key and value level
- Similarity matrix visualization for debugging

Usage:
    python eval.py --config /path/to/config.yaml [--override-case-id-start N] [--override-case-id-end M]
"""

from __future__ import annotations
import argparse, json, os, re, logging
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional, Set

import pandas as pd
import wandb
import torch
from huggingface_hub import login

try:
    import openai
except Exception:
    openai = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

try:
    import yaml
except Exception:
    yaml = None


if torch.cuda.is_available():
    torch.cuda.empty_cache()

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR

def setup_logging_eval(logs_dir: Path, judge_model: str, config_name: str) -> tuple[logging.Logger, Path]:
    """Configure a per-run logger for the evaluator.
    Creates logs/eval_<config_name>_<JUDGE>_<YYYYMMDD_HHMMSS>.log and logs to console too.
    """
    logs_dir = logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"eval_{config_name}_{judge_model}_{ts}.log"

    # Try to reuse format from config.yaml if available
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.INFO
    # Try to reuse format from YAML if available (best-effort; not fatal on failure)
    cfg_path = logs_dir.parent / "config_snapshot.yaml"
    if cfg_path.exists() and yaml is not None:
        try:
            _cfg = yaml.safe_load(cfg_path.read_text()) or {}
            _log = _cfg.get("logging", {})
            if isinstance(_log, dict):
                log_fmt = _log.get("format", log_fmt)
                lvl = str(_log.get("level", "INFO")).upper()
                log_level = getattr(logging, lvl, logging.INFO)
        except Exception:
            pass

    logger = logging.getLogger("evaluator")
    logger.setLevel(log_level)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler()
    formatter = logging.Formatter(log_fmt)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    return logger, log_file

def setup_wandb_eval(full_cfg: Dict[str, Any], judge_model: str) -> Optional[wandb.sdk.wandb_run.Run]:
    if yaml is None:
        return None
    try:
        wb = (full_cfg.get("wandb") or {}) if isinstance(full_cfg, dict) else {}
        if not wb or not wb.get("enabled", False):
            return None
        # Login and init
        api_key = wb.get("api_key")
        if api_key:
            try:
                wandb.login(key=api_key)
            except Exception:
                pass
        # Build a compact config to log
        wandb_config = {
            "processing": full_cfg.get("processing"),
            "models": full_cfg.get("models"),
            "evaluation": full_cfg.get("evaluation"),
            "performance": full_cfg.get("performance"),
            "logging": full_cfg.get("logging"),
            "environment": full_cfg.get("environment"),
            "judge": full_cfg.get("judge"),
        }
        run = wandb.init(
            project=wb.get("project"),
            entity=wb.get("entity"),
            tags=wb.get("tags"),
            group=wb.get("group"),
            config=wandb_config,
            name=f"evaluator-{judge_model}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        return run
    except Exception:
        return None

EVAL_TARGET_ONLY = True
LITELLM_API_KEY = None
LITELLM_BASE_URL = None
JUDGE_MODEL = "GPT-5.1"
TOKEN_USAGE = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

@dataclass
class EvalPaths:
    base: Path
    doc: Path = field(init=False)

    def __post_init__(self):
        repo_root = self.base.parent
        self.doc  = repo_root / "data-collection" / "CTG_DOCUMENT"

@dataclass
class Thresholds:
    list: float = 0.70
    struct: float = 0.70
    leaf: float = 0.70

@dataclass(frozen=True)
class FieldMeta:
    idx: str
    type: str
    is_list: bool
    is_leaf: bool
    target_leaf: bool
    piece: str
    description: str
    target_yn: str
    is_enum: bool = False  # New: whether this field is an enum

class FieldCatalog:
    def __init__(self, csv_path: Path, enum_path: Path):
        df = pd.read_csv(csv_path)
        df.sort_values("No", inplace=True)
        clean_indices = [idx.split('[')[0] for idx in df["FieldIndex"]]

        self.field_indexes = clean_indices
        self.types        = dict(zip(clean_indices, df["SourceType"]))
        self.list_flags   = dict(zip(clean_indices, df["ListYN"].eq("Y")))
        self.leaf_flags   = dict(zip(clean_indices, df["IsLeaf"].eq("Y")))
        self.target_leaf  = set(df[(df["TargetYN"]=="Y")&(df["IsLeaf"]=="Y")]["FieldIndex"])
        self.leaf_only    = set(df[df["IsLeaf"]=="Y"]["FieldIndex"])  # All leaf fields (no TargetYN filter)
        self.pieces       = dict(zip(clean_indices, df["Piece"]))
        self.order        = dict(zip(clean_indices, df["No"]))
        self.descriptions = dict(zip(clean_indices, df["Description"].fillna("").astype(str)))
        self.rules        = dict(zip(clean_indices, df["Rules"].fillna("").astype(str)))
        self.titles       = dict(zip(clean_indices, df["Title"].fillna("").astype(str)))
        self.target_flags = dict(zip(clean_indices, df["TargetYN"]))
        # Piece to keys mapping
        self.piece_to_keys = defaultdict(list)
        for idx, piece in self.pieces.items():
            self.piece_to_keys[piece].append(idx)
        self.piece_to_keys = dict(self.piece_to_keys)
        # Optional columns
        if 'Explanation' in df.columns:
            self.explanations = dict(zip(clean_indices, df['Explanation'].fillna("").astype(str)))
        else:
            self.explanations = {k: '' for k in clean_indices}
        # Enum detection (support multiple possible column names)
        enum_col = None
        for cand in ['IsEnum','EnumYN','Enum','EnumFlag']:
            if cand in df.columns:
                enum_col = cand
                break
        if enum_col:
            col_series = df[enum_col]
            # Normalize values to Y/N boolean
            def _is_enum_val(v):
                if pd.isna(v):
                    return False
                sval = str(v).strip().lower()
                return sval in ('y','yes','1','true')
            self.enum_flags = dict(zip(clean_indices, col_series.map(_is_enum_val)))
        else:
            self.enum_flags = {k: False for k in clean_indices}

    def meta(self, field_key: str) -> FieldMeta:
        clean_key = re.sub(r"\[\d+\]", "", field_key)
        # pick first non-empty: explanation → description → rules → title → piece
        desc = (self.explanations.get(clean_key)
                or self.descriptions.get(clean_key)
                or self.titles.get(clean_key)
                or self.pieces.get(clean_key, ""))
        last_part = clean_key.split('.')[-1]
        return FieldMeta(
            idx=clean_key,
            type=self.types.get(clean_key, "STRING"),
            is_list=self.list_flags.get(clean_key, False),
            is_leaf=self.leaf_flags.get(clean_key, False),
            target_leaf=clean_key in self.target_leaf,
            piece=self.pieces.get(clean_key, last_part),
            description=desc,
            target_yn=self.target_flags.get(clean_key, "Y"),
            is_enum=self.enum_flags.get(clean_key, False)
        )

def clean_path(s: str) -> str:
    return re.sub(r'\[\d+\]', '', s)

def _semantic_sim(a: str, b: str, context: str = "", retry_count: int = 0, field_key: str = "", call_purpose: str = "comparison") -> Tuple[float,str]:
    """
    Compute semantic similarity between two texts using judge model via LiteLLM.
    
    Args:
        a, b: Texts to compare
        context: Field description for context
        retry_count: Current retry attempt
        field_key: Field identifier for logging
        call_purpose: Purpose of this call - "comparison" (final scoring) or "alignment" (list pairing)
    
    Returns:
        (similarity_score, status)
    """
    logger = logging.getLogger("evaluator")
    
    a, b = str(a).strip().lower(), str(b).strip().lower()
    
    # Quick exitsDEMO2_EVAL/IE/data/ctg/resultsSection
    if a == b:
        logger.debug(f"[{call_purpose.upper()}] ✓ Exact match: '{field_key}' = 1.0")
        return 1.0, "exact"
    if not a or not b:
        logger.debug(f"[{call_purpose.upper()}] ✗ Empty value: '{field_key}' = 0.0")
        return 0.0, "empty"
    
    log_prefix = "🔗 ALIGN" if call_purpose == "alignment" else "📊 SCORE"
    logger.debug(f"{log_prefix} | Field: {field_key} | Len: ({len(a)}, {len(b)}) | Attempt: {retry_count + 1}")
    
    return _call_judge_model(a, b, context, field_key, retry_count, call_purpose, logger)


def _call_judge_model(a: str, b: str, context: str, field_key: str, retry_count: int, call_purpose: str, logger) -> Tuple[float, str]:
    """Call judge model via LiteLLM."""
    global TOKEN_USAGE, JUDGE_MODEL
    if openai is None:
        raise ImportError("openai package not installed")
    if not (LITELLM_API_KEY and LITELLM_BASE_URL):
        raise ValueError("LiteLLM credentials required for judge model")
    
    # Use OpenAI client with timeout (similar to inference.py)
    client = openai.OpenAI(
        api_key=LITELLM_API_KEY, 
        base_url=LITELLM_BASE_URL,
        timeout=60.0  # 60 second timeout for judge calls
    )
    
    prefix = f"Compare these clinical trial field values for '{context}'" if context else "Compare these clinical trial field values"
    prompt = f"{prefix}. Return only a number 0-1 for semantic similarity.\nText1: {a}\nText2: {b}"
    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        if response.usage:
            TOKEN_USAGE['prompt_tokens'] += response.usage.prompt_tokens
            TOKEN_USAGE['completion_tokens'] += response.usage.completion_tokens
            TOKEN_USAGE['total_tokens'] += response.usage.total_tokens
        
        raw_text = response.choices[0].message.content.strip()
        score_match = re.search(r'(\d*\.?\d+)', raw_text)
        
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
            logger.debug(f"  → Judge Score: {score:.3f}")
            return score, "success"
        
        return 0.0, f"error:parse_failed:{raw_text}"
    
    except openai.AuthenticationError as e:
        return 0.0, f"error:auth:{str(e)}"
    except openai.NotFoundError as e:
        return 0.0, f"error:not_found:{str(e)}"
    except Exception as e:
        return 0.0, f"error:api_call:{str(e)}"


# Legacy local judge model code removed - now using API only (via LiteLLM)

_norm = lambda v: " ".join(str(v).lower().split()) if v is not None else ""

def compare_scalar(p: Any, r: Any, meta: FieldMeta, th: Thresholds) -> Tuple[bool,float,bool,str]:
    """Compare scalar field values with semantic similarity."""
    sp, sr = _norm(p), _norm(r)
    logger = logging.getLogger("evaluator")
    
    if sp == sr:
        return True, 1.0, False, "exact"
    if not sp or not sr:
        return False, 0.0, False, "empty"
    
    try:
        s, status = _semantic_sim(sp, sr, context=meta.description, field_key=meta.idx, call_purpose="comparison")
        result = s >= th.leaf
        logger.debug(f"  → Result: {'✓ MATCH' if result else '✗ DIFF'} ({s:.3f} {'≥' if result else '<'} {th.leaf:.3f})")
        return result, s, False, status
    except Exception as e:
        logger.error(f"  → ✗ FAILED: {e}")
        return False, 0.0, True, "exception"

def compare_struct(p: Any, r: Any, parent_meta: FieldMeta, th: Thresholds) -> Tuple[bool,float,bool,str]:
    """Compare structured (dict) field values."""
    p_str = json.dumps(p, ensure_ascii=False, indent=2)
    r_str = json.dumps(r, ensure_ascii=False, indent=2)
    logger = logging.getLogger("evaluator")
    
    try:
        s, status = _semantic_sim(p_str, r_str, context=parent_meta.description, field_key=parent_meta.idx, call_purpose="comparison")
        result = s >= th.struct
        logger.debug(f"  → Result: {'✓ MATCH' if result else '✗ DIFF'} ({s:.3f} {'≥' if result else '<'} {th.struct:.3f})")
        return result, s, False, status
    except Exception as e:
        logger.error(f"  → ✗ FAILED: {e}")
        return False, 0.0, True, "exception"

def compare_any(p: Any, r: Any, meta: FieldMeta, cat: FieldCatalog, th: Thresholds) -> Tuple[bool,float,bool,str]:
    if isinstance(p, dict) or isinstance(r, dict):
        return compare_struct(p or {}, r or {}, meta, th)
    return compare_scalar(p, r, meta, th)

def compare_list(p_list: Sequence[Any], r_list: Sequence[Any], meta: FieldMeta, cat: FieldCatalog, th: Thresholds) -> Tuple[bool, float, bool, str]:
    """
    Compare list fields using set-based or structured comparison.
    For scalar lists: use recall metric (all ref items present in pred).
    For structured lists: serialize and compare as JSON.
    """
    if p_list is None or r_list is None:
        raise ValueError(f"compare_list called with None: p={p_list}, r={r_list}")
    
    logger = logging.getLogger("evaluator")
    
    # Both empty = perfect match
    if not r_list and not p_list:
        return True, 1.0, False, "exact"

    # Detect scalar vs structured lists
    scalar_like = not any(isinstance(x, (dict, list)) for x in (r_list + p_list))
    
    try:
        if scalar_like:
            # Scalar list: set-based recall comparison
            norm = lambda v: _norm(v)
            ref_norm = [norm(x) for x in r_list if norm(x)]
            pred_norm = [norm(x) for x in p_list if norm(x)]
            ref_set, pred_set = set(ref_norm), set(pred_norm)
            inter = ref_set & pred_set
            
            recall = len(inter) / len(ref_set) if ref_set else 1.0
            match = recall >= th.list
            
            if recall == 1.0 and len(pred_set) == len(ref_set):
                status = "exact"
            elif recall == 1.0:
                status = "superset"
            elif recall == 0.0:
                status = "disjoint"
            else:
                status = "partial"
            
            logger.debug(f"  → Scalar list: recall={recall:.3f}, {'✓ MATCH' if match else '✗ DIFF'}, status={status}")
            return match, recall, False, status
        else:
            # Structured list: JSON comparison fallback
            ref_json = json.dumps(r_list, ensure_ascii=False, sort_keys=True)
            pred_json = json.dumps(p_list, ensure_ascii=False, sort_keys=True)
            
            score, status = _semantic_sim(ref_json, pred_json, context=meta.description, field_key=meta.idx, call_purpose="comparison")
            match = score >= th.struct
            logger.debug(f"  → Structured list: score={score:.3f}, {'✓ MATCH' if match else '✗ DIFF'}")
            return match, score, False, status
    except Exception as e:
        logger.error(f"  → ✗ List comparison failed: {e}")
        return False, 0.0, True, "exception"

@dataclass
class EvalRec:
    model: str|None
    flat_path: str|None
    key_cat: str|None
    val_cat: str|None
    ref_val: Any|None
    pred_val: Any|None
    val_match: bool|None
    sim: float|None
    score_status: str|None = None  # explicit / heuristic / failed / exception / partial / superset etc.
    piece: str|None = None
    norm_path: str|None = None
    from_pred: bool = False
    in_ref: bool = False
    in_pred: bool = False
    type_mismatch: bool = False
    is_enum: bool = False
    sim_failed: bool = False

class Evaluator:
    def __init__(self, th: Thresholds, cat: FieldCatalog):
        self.th, self.cat = th, cat
        self.hungarian_results: List[Dict[str, Any]] = []  # Store Hungarian algorithm results

    # Parallel flatten with list alignment -------------
    def _flatten_parallel(self, ref: Any, pred: Any, base: str="", parent_from_pred: bool = False, type_mismatch: bool = False) -> Tuple[Dict[str,Any],Dict[str,Any],Dict[str,bool],Dict[str,bool]]:
        ref_flat, pred_flat, pred_flag, type_mismatch_flag = {}, {}, {}, {}
        
        def emit(path,rv,pv,from_pred=False,is_type_mismatch=False): 
            # Only emit if the value actually exists (not just the key)
            # For lists, check if they're not empty
            has_ref_value = rv is not None and (not isinstance(rv, list) or len(rv) > 0)
            has_pred_value = pv is not None and (not isinstance(pv, list) or len(pv) > 0)
                
            # Only add to dictionaries if there's actual data
            if has_ref_value:
                ref_flat[path] = rv
            if has_pred_value:
                pred_flat[path] = pv
                
            # Only set flags if there's a key to track
            if has_ref_value or has_pred_value:
                pred_flag[path] = from_pred
                type_mismatch_flag[path] = is_type_mismatch

        if isinstance(ref,dict) or isinstance(pred,dict):
            # to maintain order, we collect keys from both ref and pred
            keys=[]
            if isinstance(ref,dict): 
                for k in ref.keys():
                    if k not in keys:
                        keys.append(k)
            if isinstance(pred,dict): 
                for k in pred.keys():
                    if k not in keys:
                        keys.append(k)
                        
            for k in keys:
                child=f"{base}.{k}" if base else k
                r_has_key = isinstance(ref, dict) and (k in ref)
                r_child = ref.get(k) if r_has_key else None
                pred_has_key = isinstance(pred, dict) and (k in pred)
                p_child = pred.get(k) if pred_has_key else None
                
                # Skip processing if both are empty lists
                if isinstance(r_child, list) and len(r_child) == 0 and isinstance(p_child, list) and len(p_child) == 0:
                    continue
                    
                # Track if this key actually exists in pred (even if value is None)
                child_from_pred = parent_from_pred or (not r_has_key and pred_has_key)
                
                # Handle type mismatch: if one is list and other is not
                if isinstance(r_child, list) != isinstance(p_child, list):
                    if r_child is not None and p_child is not None:
                        # Type mismatch - treat as different fields
                        if isinstance(r_child, list):
                            # ref is list, pred is not
                            r_map,_,_,_ = self._flatten_parallel(r_child, None, child, parent_from_pred, type_mismatch)
                            ref_flat.update(r_map)
                            # If pred is a dict, flatten it to record all wrong fields
                            if isinstance(p_child, dict):
                                _,p_map,flag_map,tm_map = self._flatten_parallel(None, p_child, child, True, True)
                                pred_flat.update(p_map)
                                pred_flag.update(flag_map)
                                type_mismatch_flag.update(tm_map)
                            else:
                                # Emit pred as single value with different path
                                single_path = f"{child}"
                                pred_flat[single_path] = p_child
                                pred_flag[single_path] = True
                                type_mismatch_flag[single_path] = True
                        else:
                            # pred is list, ref is not
                            _,p_map,flag_map,tm_map = self._flatten_parallel(None, p_child, child, True, True)
                            pred_flat.update(p_map)
                            pred_flag.update(flag_map)
                            type_mismatch_flag.update(tm_map)
                            # Emit ref as single value
                            ref_flat[child] = r_child
                            pred_flag[child] = False
                            type_mismatch_flag[child] = False
                        continue
                
                # Process normally - pass whether key exists in pred
                # Only process if at least one side has the key with a meaningful value
                if r_has_key or pred_has_key:
                    r_map,p_map,flag_map,tm_map=self._flatten_parallel(r_child,p_child,child,child_from_pred,type_mismatch)
                    ref_flat.update(r_map); pred_flat.update(p_map); pred_flag.update(flag_map); type_mismatch_flag.update(tm_map)
        
        elif isinstance(ref, list) or isinstance(pred, list):
            ref_l, pred_l = ref if ref is not None else [], pred if pred is not None else []
            clean_key = re.sub(r"\[\d+\]", "", base)
            
            # Modified empty list handling: process only if field actually exists
            if ref is None and pred is None:
                return ref_flat, pred_flat, pred_flag, type_mismatch_flag
                
            meta = self.cat.meta(clean_key) if clean_key in self.cat.field_indexes else None
            if not meta:
                # Unknown field - emit as is
                if ref is not None or pred is not None:
                    emit(base, ref, pred, parent_from_pred or (ref is None and pred is not None), type_mismatch)
                return ref_flat, pred_flat, pred_flag, type_mismatch_flag
            
            # Check if both lists have the same structure type
            ref_has_complex = any(isinstance(x, (dict, list)) for x in ref_l)
            pred_has_complex = any(isinstance(x, (dict, list)) for x in pred_l)
            structure_mismatch = ref_has_complex != pred_has_complex
            
            is_scalar_list = meta.is_list and meta.type != "STRUCT" and not ref_has_complex and not pred_has_complex
            
            if structure_mismatch:
                # Type mismatch: one is scalar array, other is struct array
                # Emit both as type_mismatch and process items separately
                if ref is not None or pred is not None:
                    emit(base, ref, pred, 
                         from_pred=parent_from_pred or (ref is None and pred is not None),
                         is_type_mismatch=True)
                
                # Also process individual items to capture the structural difference
                max_len = max(len(ref_l), len(pred_l))
                for i in range(max_len):
                    child = f"{base}[{i}]"
                    r_child = ref_l[i] if i < len(ref_l) else None
                    p_child = pred_l[i] if i < len(pred_l) else None
                    child_from_pred = parent_from_pred or (r_child is None and p_child is not None)
                    r_map, p_map, flag_map, tm_map = self._flatten_parallel(r_child, p_child, child, child_from_pred, True)
                    ref_flat.update(r_map)
                    pred_flat.update(p_map)
                    pred_flag.update(flag_map)
                    type_mismatch_flag.update(tm_map)
            elif is_scalar_list:
                # Scalar list processing - distinguish between None and empty list
                # If ref is None, key is missing in ref
                # If pred is None, key is missing in pred
                if ref is not None or pred is not None:
                    # Do not emit only if both are empty lists (no actual difference)
                    # If only one is empty list, it's a significant difference, so emit
                    if not (isinstance(ref, list) and len(ref) == 0 and isinstance(pred, list) and len(pred) == 0):
                        emit(base, ref, pred, 
                             from_pred=parent_from_pred or (ref is None and pred is not None),
                             is_type_mismatch=type_mismatch)
            else:
                # Both are structured lists - use Hungarian alignment
                mapping = self._align_struct_list(ref_l, pred_l, base)
                for i, j in mapping:
                    child = f"{base}[{i if i is not None else j}]"
                    r_child = ref_l[i] if i is not None else None
                    p_child = pred_l[j] if j is not None else None
                    child_from_pred = parent_from_pred or (i is None and j is not None)
                    r_map, p_map, flag_map, tm_map = self._flatten_parallel(r_child, p_child, child, child_from_pred, type_mismatch)
                    ref_flat.update(r_map)
                    pred_flat.update(p_map)
                    pred_flag.update(flag_map)
                    type_mismatch_flag.update(tm_map)
                    
                # Add unmatched pred items
                matched_j = {j for i, j in mapping if j is not None}
                for j in range(len(pred_l)):
                    if j not in matched_j:
                        child = f"{base}[{j}]"
                        r_map, p_map, flag_map, tm_map = self._flatten_parallel(None, pred_l[j], child, True, type_mismatch)
                        ref_flat.update(r_map)
                        pred_flat.update(p_map)
                        pred_flag.update(flag_map)
                        type_mismatch_flag.update(tm_map)
            

        else:
            # Scalar values - emit if key exists in either ref or pred
            # If parent_from_pred is True, key comes from pred, so emit unconditionally
            if ref is not None or pred is not None or parent_from_pred:
                emit(base, ref, pred, from_pred=parent_from_pred or (ref is None and pred is not None), is_type_mismatch=type_mismatch)
            
        return ref_flat, pred_flat, pred_flag, type_mismatch_flag

    def _align_struct_list(self, ref_l: List[Any], pred_l: List[Any], base: str) -> List[Tuple[Optional[int],Optional[int]]]:
        """
        Align list of structured items using Hungarian algorithm.
        Returns list of (ref_index, pred_index|None) for each reference item.
        """
        logger = logging.getLogger("evaluator")
        clean_key = re.sub(r"\[\d+\]", "", base)
        list_meta = self.cat.meta(clean_key)
        m, n = len(ref_l), len(pred_l)
        
        logger.info(f"{'='*80}")
        logger.info(f"🔗 HUNGARIAN ALIGNMENT: {base}")
        logger.info(f"   Reference items: {m} | Prediction items: {n}")
        logger.info(f"{'='*80}")
        
        # Build similarity matrix
        sim_matrix: List[List[float]] = []
        for rv in ref_l:
            row: List[float] = []
            for pv in pred_l:
                try:
                    _, sim, failed, _status = compare_any(pv, rv, list_meta, self.cat, self.th)
                except Exception:
                    sim, failed = 0.0, True
                row.append(0.0 if failed or sim is None else float(sim))
            sim_matrix.append(row)

        if m == 0:
            logger.info("  → No reference items")
            return []
        if n == 0:
            logger.info("  → No prediction items, all unmatched")
            return [(i, None) for i in range(m)]

        # Visualize similarity matrix
        self._log_similarity_matrix(sim_matrix, m, n, base)

        if linear_sum_assignment is not None and np is not None:
            K = max(m, n)
            pad_cost = 0.0
            cost = np.full((K, K), pad_cost, dtype=float)
            for i in range(m):
                for j in range(n):
                    cost[i, j] = 1.0 - sim_matrix[i][j]
            row_ind, col_ind = linear_sum_assignment(cost)
            pairs: List[Tuple[Optional[int], Optional[int]]] = []
            
            logger.info(f"\n📋 ALIGNMENT RESULTS:")
            matched_pairs = []
            for i in range(m):
                ref_item = ref_l[i]
                try:
                    j = int(col_ind[list(row_ind).index(i)])
                except ValueError:
                    pairs.append((i, None))
                    logger.info(f"  Ref[{i}] → (unmatched)")
                    matched_pairs.append({
                        "ref_idx": i, 
                        "pred_idx": None, 
                        "score": None, 
                        "status": "unmatched",
                        "ref_item": ref_item,
                        "pred_item": None
                    })
                    continue
                if j < n:
                    pred_item = pred_l[j]
                    score = sim_matrix[i][j]
                    pairs.append((i, j))
                    logger.info(f"  Ref[{i}] → Pred[{j}] | Score: {score:.3f}")
                    matched_pairs.append({
                        "ref_idx": i, 
                        "pred_idx": j, 
                        "score": score, 
                        "status": "matched",
                        "ref_item": ref_item,
                        "pred_item": pred_item
                    })
                else:
                    pairs.append((i, None))
                    logger.info(f"  Ref[{i}] → (padded)")
                    matched_pairs.append({
                        "ref_idx": i, 
                        "pred_idx": None, 
                        "score": None, 
                        "status": "padded",
                        "ref_item": ref_item,
                        "pred_item": None
                    })
            
            # Store Hungarian result
            self.hungarian_results.append({
                "field_path": base,
                "algorithm": "hungarian",
                "ref_count": m,
                "pred_count": n,
                "similarity_matrix": sim_matrix,
                "matched_pairs": matched_pairs
            })
            
            return pairs
        
        if not hasattr(self, "_hungarian_warned"):
            logger.warning("⚠️ SciPy unavailable, greedy fallback")
            self._hungarian_warned = True  # type: ignore[attr-defined]
        
        used = set()
        pairs: List[Tuple[Optional[int], Optional[int]]] = []
        logger.info(f"\n📋 GREEDY ALIGNMENT RESULTS:")
        matched_pairs = []
        for i, row in enumerate(sim_matrix):
            ref_item = ref_l[i]
            best_j = None
            best_sim = -1.0
            for j, sim in enumerate(row):
                if j in used:
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j is not None:
                pred_item = pred_l[best_j]
                used.add(best_j)
                pairs.append((i, best_j))
                logger.info(f"  Ref[{i}] → Pred[{best_j}] | Score: {best_sim:.3f}")
                matched_pairs.append({
                    "ref_idx": i, 
                    "pred_idx": best_j, 
                    "score": best_sim, 
                    "status": "matched",
                    "ref_item": ref_item,
                    "pred_item": pred_item
                })
            else:
                pairs.append((i, None))
                logger.info(f"  Ref[{i}] → (unmatched)")
                matched_pairs.append({
                    "ref_idx": i, 
                    "pred_idx": None, 
                    "score": None, 
                    "status": "unmatched",
                    "ref_item": ref_item,
                    "pred_item": None
                })
        
        # Store greedy alignment result
        self.hungarian_results.append({
            "field_path": base,
            "algorithm": "greedy",
            "ref_count": m,
            "pred_count": n,
            "similarity_matrix": sim_matrix,
            "matched_pairs": matched_pairs
        })
        
        return pairs
    
    def _log_similarity_matrix(self, matrix: List[List[float]], m: int, n: int, field_path: str):
        """Log similarity matrix as formatted table."""
        logger = logging.getLogger("evaluator")
        
        logger.info(f"\n📊 SIMILARITY MATRIX: {field_path}")
        
        # Header
        header = "      " + "".join(f"  P{j:2d}" for j in range(n))
        logger.info(header)
        logger.info("    " + "─" * (6 * n + 2))
        
        # Rows
        for i in range(m):
            row_str = f" R{i:2d} │"
            for j in range(n):
                score = matrix[i][j]
                if score >= 0.9:
                    row_str += f" {score:.2f}✓"
                elif score >= 0.7:
                    row_str += f" {score:.2f}~"
                elif score >= 0.5:
                    row_str += f" {score:.2f}·"
                else:
                    row_str += f" {score:.2f}✗"
            logger.info(row_str)
        logger.info("")
    
    def save_hungarian_results(self, output_path: Path, case_id: int, model: str, metadata: Dict[str, Any] = None) -> None:
        """Save Hungarian algorithm results to JSON file."""
        logger = logging.getLogger("evaluator")
        
        if not self.hungarian_results:
            logger.debug(f"No Hungarian results to save for case {case_id}")
            return
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                "metadata": {
                    "case_id": case_id,
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "total_alignments": len(self.hungarian_results),
                    **(metadata or {})
                },
                "alignments": self.hungarian_results
            }
            
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Hungarian results saved → {output_path} ({len(self.hungarian_results)} alignments)")
        except Exception as e:
            logger.error(f"✗ Failed to save Hungarian results: {e}")
    
    def clear_hungarian_results(self) -> None:
        """Clear stored Hungarian results for next case."""
        self.hungarian_results.clear()
    

    
    def evaluate_objects(self, ref_data: Dict[str, Any], pred_data: Dict[str, Any], model: str, target_fields: Optional[Set[str]] = None) -> List[EvalRec]:
        """
        Evaluate prediction against reference for all fields.
        Returns list of EvalRec with categorization (TP/FP/FN/EV/NA).
        """
        logger = logging.getLogger("evaluator")
        logger.info(f"\n{'='*80}")
        logger.info(f"📝 EVALUATING MODEL: {model}")
        logger.info(f"{'='*80}\n")
        
        ref_flat, pred_flat, pred_flag, type_mismatch_flag = self._flatten_parallel(ref_data, pred_data, "", False, False)
        all_keys = set(ref_flat.keys()) | set(pred_flat.keys())
        recs: List[EvalRec] = []
        
        for flat_path in all_keys:
            flag_pred = pred_flag.get(flat_path, False)
            is_type_mismatch = type_mismatch_flag.get(flat_path, False)
            norm_key = re.sub(r"\[\d+\]", "", flat_path)
            pv, rv = pred_flat.get(flat_path), ref_flat.get(flat_path)
            in_ref, in_pred = flat_path in ref_flat, flat_path in pred_flat
            
            meta = self.cat.meta(norm_key) if norm_key in self.cat.field_indexes else None
            piece = meta.piece if meta else None
            
            # Skip non-leaf fields (IsLeaf=N)
            if meta is not None and not self.cat.leaf_flags.get(norm_key, False):
                continue
            
            # Create evaluation record based on field presence and values
            rec = self._categorize_field(flat_path, norm_key, rv, pv, in_ref, in_pred, 
                                        meta, piece, flag_pred, is_type_mismatch, model)
            if rec:
                recs.append(rec)
        
        logger.info(f"✅ Evaluation complete: {len(recs)} records generated\n")
        return recs
    
    def _categorize_field(self, flat_path: str, norm_key: str, rv: Any, pv: Any, 
                         in_ref: bool, in_pred: bool, meta: Optional[FieldMeta], 
                         piece: Optional[str], flag_pred: bool, is_type_mismatch: bool, 
                         model: str) -> Optional[EvalRec]:
        """Categorize a single field into TP/FP/FN/EV/NA."""
        
        # Case 1: GT absent, Pred present
        if rv is None and pv is not None:
            if norm_key not in self.cat.field_indexes:
                # Unknown field -> FP
                return EvalRec(model, flat_path, "FP", "FP", None, pv, False, 0.0, None, 
                             None, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
            else:
                # Known leaf field -> EV (expected value)
                return EvalRec(model, flat_path, "EV", "EV", None, pv, None, None, None, 
                             piece, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
        
        # Case 2: GT present, Pred absent
        elif rv is not None and pv is None:
            key_cat = "FN" if not in_pred else "TP"
            val_cat = "FN" if not in_pred else "FP"
            return EvalRec(model, flat_path, key_cat, val_cat, rv, None, False, 0.0, None, 
                         piece, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
        
        # Case 3: Both absent
        elif rv is None and pv is None:
            if is_type_mismatch and in_pred:
                return EvalRec(model, flat_path, "FP", "TN", None, None, True, 1.0, "exact", 
                             None, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
            elif in_pred and meta:
                return EvalRec(model, flat_path, "TP", "TN", None, None, True, 1.0, "exact", 
                             piece, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
            elif in_pred and not meta:
                return EvalRec(model, flat_path, "FP", "TN", None, None, True, 1.0, "exact", 
                             None, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
            else:
                return EvalRec(model, flat_path, "NA", "NA", None, None, None, None, None, 
                             piece, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
        
        # Case 4: Both present
        else:
            if is_type_mismatch:
                return EvalRec(model, flat_path, "FP", "FP", rv, pv, False, None, None, 
                             None, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
            elif not meta:
                return EvalRec(model, flat_path, "FP", "FP", rv, pv, False, None, None, 
                             None, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch)
            else:
                # Compare values
                if isinstance(pv, list) and isinstance(rv, list):
                    value_match, sim, failed, status = compare_list(pv, rv, meta, self.cat, self.th)
                else:
                    value_match, sim, failed, status = compare_any(pv, rv, meta, self.cat, self.th)
                
                key_cat = "TP"
                val_cat = "TP" if value_match else "FP"
                return EvalRec(model, flat_path, key_cat, val_cat, rv, pv, value_match, sim, status, 
                             piece, norm_key, flag_pred, in_ref, in_pred, is_type_mismatch, 
                             meta.is_enum, failed)

def calc_metrics(records: List[EvalRec]) -> Dict[str, float]:
    """Calculate precision, recall, F1 for key and value level predictions."""
    logger = logging.getLogger("evaluator")
    
    key_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "EV": 0, "NA": 0}
    val_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "EV": 0, "NA": 0}

    for r in records:
        key_counts[r.key_cat] = key_counts.get(r.key_cat, 0) + 1
        val_counts[r.val_cat] = val_counts.get(r.val_cat, 0) + 1

    def calc_prf(counts):
        tp, fp, fn = counts.get("TP", 0), counts.get("FP", 0), counts.get("FN", 0)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    key_p, key_r, key_f1 = calc_prf(key_counts)
    val_p, val_r, val_f1 = calc_prf(val_counts)

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 METRICS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total fields: {len(records)}")
    logger.info(f"\nKey-level:   P={key_p:.3f}  R={key_r:.3f}  F1={key_f1:.3f}")
    logger.info(f"  Counts: {key_counts}")
    logger.info(f"\nValue-level: P={val_p:.3f}  R={val_r:.3f}  F1={val_f1:.3f}")
    logger.info(f"  Counts: {val_counts}")
    logger.info(f"{'='*80}\n")

    # Collect field lists by category
    categorize = lambda cat_attr, cat_val: [r.norm_path for r in records if getattr(r, cat_attr) == cat_val]
    
    return {
        "total": len(records),
        "key_precision": key_p,
        "key_recall": key_r,
        "key_f1": key_f1,
        "value_precision": val_p,
        "value_recall": val_r,
        "value_f1": val_f1,
        "key_counts": key_counts,
        "value_counts": val_counts,
        "key_EV_fields": categorize("key_cat", "EV"),
        "key_FN_fields": categorize("key_cat", "FN"),
        "key_FP_fields": categorize("key_cat", "FP"),
        "key_TP_fields": categorize("key_cat", "TP"),
        "key_TN_fields": categorize("key_cat", "TN"),
        "key_NA_fields": categorize("key_cat", "NA"),
        "val_EV_fields": categorize("val_cat", "EV"),
        "val_FN_fields": categorize("val_cat", "FN"),
        "val_FP_fields": categorize("val_cat", "FP"),
        "val_TP_fields": categorize("val_cat", "TP"),
        "val_TN_fields": categorize("val_cat", "TN"),
        "val_NA_fields": categorize("val_cat", "NA"),
    }

def calculate_range(case_id: int, range_size: int = 100) -> str:
    """Calculate range string for case_id (e.g., 1→'1_100', 150→'101_200')."""
    start = ((case_id - 1) // range_size) * range_size + 1
    end = start + range_size - 1
    return f"{start}_{end}"

def main():
    global BASE_DIR, EVAL_TARGET_ONLY, LITELLM_API_KEY, LITELLM_BASE_URL, JUDGE_MODEL

    parser = argparse.ArgumentParser(description="Clinical Trials Hub Evaluation")
    parser.add_argument("--config", required=True, help="Absolute path to evaluation YAML config (e.g., config_index_lines.yaml)")
    parser.add_argument("--override-case-id-start", type=int, help="Override processing.case_id_start")
    parser.add_argument("--override-case-id-end", type=int, help="Override processing.case_id_end")
    parser.add_argument("--judge-model", help="Override judge.model")
    parser.add_argument("--shard-id", type=int, help="Shard id (0-indexed) for parallel range splitting")
    parser.add_argument("--num-shards", type=int, help="Total number of shards for parallel range splitting")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse the config file")
    full_cfg = yaml.safe_load(config_path.read_text()) or {}

    # Persist snapshot of config near outputs (path decided below) later

    # Populate globals from YAML
    EVAL_TARGET_ONLY = bool(full_cfg.get("evaluation", {}).get("target_only", True))
    LITELLM_API_KEY = full_cfg.get("judge", {}).get("litellm_api_key") or os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = full_cfg.get("judge", {}).get("litellm_base_url") or os.getenv("LITELLM_BASE_URL")
    JUDGE_MODEL = args.judge_model or full_cfg.get("judge", {}).get("model", "GPT-5.1")

    # HF token from YAML models section
    hf_token = full_cfg.get("models", {}).get("hf_token")
    if hf_token:
        try:
            login(token=hf_token)
            print("✅ HF login via YAML token succeeded")
        except Exception as e:
            print(f"⚠️ HF login failed: {e}")

    # Determine BASE_DIR context (root for relative references). Use config file parent by default.
    BASE_DIR = config_path.parent

    # Build output root path from config or default
    config_name = config_path.stem
    # YOURPATH = {YOURPATH}
    default_output_root = f"/YOURPATH/eval_output/{config_name}"
    output_root_str = full_cfg.get("output", {}).get("root", default_output_root)
    output_root_str = output_root_str.format(config_name=config_name)
    output_root = Path(output_root_str).expanduser().resolve()
    logs_dir = output_root / "logs"
    metrics_dir = output_root / "metrics"
    metadata_dir = output_root / "metadata"
    for d in (logs_dir, metrics_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save timestamped config snapshot (avoid overwrite across parallel runs)
    snapshot_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = metadata_dir / f"config_snapshot_{snapshot_ts}.yaml"
    snapshot_path.write_text(yaml.safe_dump(full_cfg, sort_keys=False))

    # Prepare logger
    logger, log_path = setup_logging_eval(logs_dir, JUDGE_MODEL, config_name)
    wb_run = setup_wandb_eval(full_cfg, JUDGE_MODEL)
    logger.info("=" * 80)
    logger.info("Evaluator Startup Configuration")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Judge Model: {JUDGE_MODEL} (via LiteLLM)")
    logger.info(f"Target-only (TargetYN=Y, leaf): {EVAL_TARGET_ONLY}")
    logger.info(f"LiteLLM BASE_URL set: {bool(LITELLM_BASE_URL)} | API KEY set: {bool(LITELLM_API_KEY)}")
    if wb_run:
        logger.info(f"WandB URL: {wandb.run.url}")
    
    # Test Judge API connection
    logger.info(f"Testing {JUDGE_MODEL} API connection...")
    if openai is None:
        raise ImportError("openai package not installed. Install with: pip install openai")
    if not LITELLM_API_KEY or not LITELLM_BASE_URL:
        raise ValueError("LITELLM_API_KEY and LITELLM_BASE_URL must be set (check config or env vars)")
    
    client = openai.OpenAI(
        api_key=LITELLM_API_KEY, 
        base_url=LITELLM_BASE_URL,
        timeout=60.0
    )
    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": "Test connection. Reply with OK."}],
            max_tokens=10,
            # temperature=0
        )
        content = response.choices[0].message.content
        logger.info(f"✅ {JUDGE_MODEL} API connected successfully! Response: {content.strip()}")
    except Exception as e:
        logger.error(f"❌ {JUDGE_MODEL} API connection test failed: {e}")
        raise RuntimeError(f"Failed to connect to {JUDGE_MODEL} API: {e}")
    
    logger.info("Starting evaluation...")

    # Determine processing mode directly from provided config
    processing_mode = 'index_lines'  # Force single supported mode
    index_cfg = full_cfg

    # Helper: remove ignored keys recursively
    def _remove_ignored(obj: Any, ignore: List[str]):
        if isinstance(obj, dict):
            return {k: _remove_ignored(v, ignore) for k, v in obj.items() if k not in ignore}
        if isinstance(obj, list):
            return [_remove_ignored(x, ignore) for x in obj]
        return obj

    # Helper: trim to target sections
    def _select_sections(data: Dict[str, Any], sections: List[str]):
        if not isinstance(data, dict):
            return {}
        return {k: data.get(k) for k in sections if k in data}

    # Helper: fallback – hoist nested target sections if wrapped (e.g. predictions.protocolSection)
    def _fallback_hoist(data: Dict[str, Any], sections: List[str]) -> Dict[str, Any]:
        """If none of the target sections are present at root, search one level deep
        and hoist them up (common pattern: {"predictions": {"protocolSection": {...}}}).
        Returns new dict with any recovered sections or empty dict if not found.
        """
        if not isinstance(data, dict):
            return {}
        # Fast path: if at least one section already present, do nothing
        if any(sec in data for sec in sections):
            return {k: data.get(k) for k in sections if k in data}
        recovered = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for sec in sections:
                    if sec in v and sec not in recovered:
                        recovered[sec] = v[sec]
        return recovered

    paths = EvalPaths(Path(BASE_DIR))
    global catalog
    # Prefer absolute doc paths from config (doc_paths section). Fallback to legacy relative.
    doc_cfg = (index_cfg.get('doc_paths') if isinstance(index_cfg, dict) else None) or {}
    data_def_path = doc_cfg.get('data_definition_csv')
    enums_path    = doc_cfg.get('enums_json') or str(paths.doc / 'enums.json')
    if not Path(data_def_path).exists():
        raise FileNotFoundError(f"DataDefinition CSV not found: {data_def_path}")
    if not Path(enums_path).exists():
        raise FileNotFoundError(f"Enums JSON not found: {enums_path}")
    catalog = FieldCatalog(Path(data_def_path), Path(enums_path))
    logger.info(f"Loaded catalog from:\n  DataDefinition: {data_def_path}\n  Enums: {enums_path}")
    # Instantiate evaluator (paths no longer required after legacy removal)
    evalr = Evaluator(Thresholds(), catalog)

    summary: Dict[str, Dict[str, float]] = {}

    if processing_mode == 'index_lines':
        logger.info("Running in index_lines evaluation mode (index-based model outputs)")
        proc_cfg = index_cfg.get('processing', {})
        eval_cfg = index_cfg.get('evaluation', {}) or {}
        index_file = proc_cfg.get('index_file')
        # Case-id based range (inclusive)
        base_case_start = int(proc_cfg.get('case_id_start', 1))
        base_case_end = int(proc_cfg.get('case_id_end', base_case_start))
        original_case_start, original_case_end = base_case_start, base_case_end
        # CLI overrides
        if args.override_case_id_start is not None:
            base_case_start = int(args.override_case_id_start)
        if args.override_case_id_end is not None:
            base_case_end = int(args.override_case_id_end)
        if base_case_end < base_case_start:
            raise ValueError("case_id_end cannot be less than case_id_start")
        # Shard splitting on case-id range
        if args.shard_id is not None and args.num_shards is not None:
            if args.shard_id < 0 or args.num_shards <= 0 or args.shard_id >= args.num_shards:
                raise ValueError("Invalid shard specification")
            total = base_case_end - base_case_start + 1
            per = total // args.num_shards
            rem = total % args.num_shards
            shard_sizes = [per + (1 if i < rem else 0) for i in range(args.num_shards)]
            shard_starts = []
            acc = base_case_start
            for sz in shard_sizes:
                shard_starts.append(acc)
                acc += sz
            shard_start = shard_starts[args.shard_id]
            shard_end = shard_start + shard_sizes[args.shard_id] - 1
            case_id_start, case_id_end = shard_start, shard_end
            logger.info(f"Shard {args.shard_id}/{args.num_shards} -> case_ids {case_id_start}-{case_id_end} (total {total})")
        else:
            case_id_start, case_id_end = base_case_start, base_case_end
        logger.info(f"Effective evaluation case_id range: {case_id_start}..{case_id_end}")
        models: List[Dict[str, str]] = eval_cfg.get('models', [])
        target_fields_cfg: Dict[str, List[str]] = eval_cfg.get('target_fields', {})
        ignore_keys: List[str] = eval_cfg.get('ignore_field_keys') or ['error']
        
        # Get target section from config (fallback to auto-detection from path)
        config_target_section = eval_cfg.get('target_section', None)
        
        eval_output_dir = metrics_dir
        logger.info(f"Index file: {index_file}")
        logger.info(f"Case ID range {case_id_start}..{case_id_end}")
        logger.info(f"Models: {[m['name'] for m in models]}")
        if config_target_section:
            logger.info(f"Target section (from config): {config_target_section}")
        logger.info(f"Target fields config: protocolSection={len(target_fields_cfg.get('protocolSection', []))} pieces, resultsSection={len(target_fields_cfg.get('resultsSection', []))} pieces")
        logger.info(f"Ignore field keys: {ignore_keys}")

        if not index_file or not Path(index_file).exists():
            raise FileNotFoundError(f"Index CSV not found: {index_file}")
        df_index = pd.read_csv(index_file)
        if 'case_id' not in df_index.columns:
            raise ValueError("Index CSV must contain 'case_id' column for case-id based filtering")
        # Filter by inclusive case_id range
        df_slice = df_index[(df_index['case_id'] >= case_id_start) & (df_index['case_id'] <= case_id_end)]
        logger.info(f"Loaded {len(df_slice)} rows (case_id filtered)")
        
        for _, row in df_slice.iterrows():
            try:
                case_id = int(row.get('case_id')) if not pd.isna(row.get('case_id')) else None
            except Exception:
                case_id = None
            if case_id is None:
                logger.warning("Skipping row without valid case_id")
                continue
            
            ctg_path = row.get('ctg_path', '')
            if not ctg_path or not Path(ctg_path).exists():
                logger.warning(f"Case {case_id}: invalid or missing ctg_path: {ctg_path}")
                continue
            
            # Determine target section: use config value if specified, otherwise auto-detect from path
            if config_target_section:
                target_section = config_target_section
            elif '/protocolSection/' in ctg_path:
                target_section = 'protocolSection'
            elif '/resultsSection/' in ctg_path:
                target_section = 'resultsSection'
            else:
                logger.warning(f"Case {case_id}: cannot determine section from ctg_path: {ctg_path}")
                logger.warning(f"  → Please add 'target_section: protocolSection' or 'target_section: resultsSection' to evaluation config")
                continue
            
            target_pieces = target_fields_cfg.get(target_section, [])
            if target_section == 'resultsSection' and 'NCTId' in target_pieces:
                target_pieces = [p for p in target_pieces if p != 'NCTId']
            
            # Only include IsLeaf=Y fields (leaf nodes) for evaluation
            target_fields = set()
            for piece in target_pieces:
                if piece in catalog.piece_to_keys:
                    # Filter only fields where IsLeaf == 'Y'
                    leaf_keys = [k for k in catalog.piece_to_keys[piece] if catalog.leaf_flags.get(k, False)]
                    target_fields.update(leaf_keys)
            
            logger.info(f"Case {case_id}: section={target_section}, target pieces={len(target_pieces)}, target fields={len(target_fields)}")
            
            try:
                ref_full = json.loads(Path(ctg_path).read_text())
            except Exception as e:
                logger.warning(f"Case {case_id}: failed to read ground truth JSON: {e}")
                continue
            
            ref_trim = _select_sections(ref_full, [target_section])
            ref_trim = _remove_ignored(ref_trim, ignore_keys)
            
            for model in models:
                model_name = model['name']
                path_pattern = model['path_pattern']
                
                # Reset token usage for this case/model evaluation
                global TOKEN_USAGE
                TOKEN_USAGE = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                
                # Determine section suffix early for skip check
                section_suffix = "_protocol" if target_section == "protocolSection" else "_results"
                
                # Check if output already exists - skip evaluation if so
                shard_range = calculate_range(case_id)
                model_root_dir = eval_output_dir / model_name / shard_range
                metrics_sub_dir = model_root_dir / 'metrics'
                met_out = metrics_sub_dir / f"case{case_id}{section_suffix}_metrics.json"
                
                if met_out.exists():
                    logger.info(f"⏭️  SKIP Case {case_id} Model {model_name} ({target_section}): output already exists → {met_out}")
                    continue
                
                # Support both {case_id_range} pattern and direct path
                if "{case_id_range}" in path_pattern:
                    range_str = calculate_range(case_id)
                    dir_path = Path(path_pattern.replace("{case_id_range}", range_str))
                else:
                    # Direct path - no subdirectory by range
                    dir_path = Path(path_pattern)
                
                if not dir_path.exists():
                    logger.warning(f"Case {case_id}: prediction directory not found for {model_name}: {dir_path}")
                    continue                
                
                pred_files = list(dir_path.glob(f"{case_id}_*.json"))
                if not pred_files:
                    logger.warning(f"Case {case_id}: no prediction file found for {model_name} in {dir_path}")
                    continue
                if len(pred_files) > 1:
                    logger.warning(f"Case {case_id}: multiple prediction files found for {model_name} in {dir_path}, using first: {pred_files[0]}")
                pred_path = str(pred_files[0])
                try:
                    pred_full = json.loads(Path(pred_path).read_text())
                except Exception as e:
                    logger.warning(f"Case {case_id}: failed to read prediction JSON for {model_name}: {e}")
                    continue
                
                if isinstance(pred_full, dict) and set(pred_full.keys()) == {'error'}:
                    logger.warning(f"Case {case_id}: only error field present for {model_name} -> skip")
                    continue
                
                pred_trim = _select_sections(pred_full, [target_section])
                if not pred_trim:
                    hoisted = _fallback_hoist(pred_full, [target_section])
                    if hoisted:
                        logger.info(f"Case {case_id}: hoisted nested target sections for {model_name} -> {list(hoisted.keys())}")
                        pred_trim = hoisted
                pred_trim = _remove_ignored(pred_trim, ignore_keys)
                
                if not any(isinstance(pred_trim.get(sec), dict) and pred_trim.get(sec) for sec in pred_trim):
                    logger.warning(f"Case {case_id}: no target sections present for {model_name} -> skip")
                    continue
                logger.info(f"Evaluating Case {case_id}, Model {model_name} (index_lines mode)...")
                recs = evalr.evaluate_objects(ref_trim, pred_trim, model_name, target_fields=target_fields)
                
                # Apply filtering based on target_only setting
                target_only = eval_cfg.get('target_only', True)
                before_cnt = len(recs)
                
                if target_only:
                    # Original: filter to TargetYN=Y AND IsLeaf=Y fields
                    recs = [r for r in recs if (r.norm_path in target_fields) or (r.in_pred and r.key_cat in ("FP","EV","FN"))]
                    logger.info(f"Target-only filter applied (TargetYN=Y AND IsLeaf=Y, preserve FP/EV/FN): kept {len(recs)}/{before_cnt} records")
                else:
                    # New: filter to only IsLeaf=Y fields (no TargetYN requirement)
                    leaf_only_fields = catalog.leaf_only
                    recs = [r for r in recs if (r.norm_path in leaf_only_fields) or (r.in_pred and r.key_cat in ("FP","EV","FN"))]
                    logger.info(f"Leaf-only filter applied (IsLeaf=Y, preserve FP/EV/FN): kept {len(recs)}/{before_cnt} records")
                
                recs = [r for r in recs if r.norm_path is None or r.norm_path.split('.')[-1] not in ignore_keys]
                recs.sort(key=lambda rec: (evalr.cat.order.get(rec.norm_path, 999999), rec.flat_path or ""))
                
                df_recs = pd.DataFrame([r.__dict__ for r in recs])
                
                # Add section suffix to distinguish protocol vs results
                section_suffix = "_protocol" if target_section == "protocolSection" else "_results"
                
                shard_range = calculate_range(case_id)
                model_root_dir = eval_output_dir / model_name / shard_range
                records_dir = model_root_dir / 'records'
                metrics_sub_dir = model_root_dir / 'metrics'
                hungarian_dir = model_root_dir / 'hungarian_alignments'
                for d in (records_dir, metrics_sub_dir, hungarian_dir):
                    d.mkdir(parents=True, exist_ok=True)
                
                csv_out = records_dir / f"case{case_id}{section_suffix}_records.csv"
                df_recs.to_csv(csv_out, index=False)
                
                # Save Hungarian alignment results
                hungarian_out = hungarian_dir / f"case{case_id}{section_suffix}_hungarian.json"
                evalr.save_hungarian_results(
                    hungarian_out, 
                    case_id, 
                    model_name,
                    metadata={
                        "target_section": target_section,
                        "ground_truth_path": ctg_path,
                        "prediction_path": pred_path,
                        "shard_range": shard_range,
                    }
                )
                

                
                met = calc_metrics(recs)
                met_meta = {
                    "case_id": case_id,
                    "shard_range": shard_range,
                    "ground_truth_path": ctg_path,
                    "prediction_path": pred_path,
                    "model": model_name,
                    "target_section": target_section,
                    "target_pieces_count": len(target_pieces),
                    "target_fields_count": len(target_fields),
                    "response_metadata": {
                        "prompt_tokens": TOKEN_USAGE['prompt_tokens'],
                        "completion_tokens": TOKEN_USAGE['completion_tokens'],
                        "total_tokens": TOKEN_USAGE['total_tokens'],
                        "model": JUDGE_MODEL
                    }
                }
                met = {**met_meta, **met}
                met_out = metrics_sub_dir / f"case{case_id}{section_suffix}_metrics.json"
                met_out.write_text(json.dumps(met, indent=2))
                summary[f"case{case_id}_{model_name}{section_suffix}"] = met
                
                # Legacy judge interaction logging removed (API-only evaluation)
                
                # Clear Hungarian results for next case
                evalr.clear_hungarian_results()
                
                logger.info(f"Case {case_id} Model {model_name}: {len(recs)} records -> metrics saved")
                if wb_run:
                    try:
                        wandb.log({
                            "case": case_id,
                            "model": model_name,
                            "key_precision": met.get("key_precision"),
                            "key_recall": met.get("key_recall"),
                            "key_f1": met.get("key_f1"),
                            "value_precision": met.get("value_precision"),
                            "value_recall": met.get("value_recall"),
                            "value_f1": met.get("value_f1"),
                            "records_count": len(recs),
                            "mode": "index_lines"
                        })
                    except Exception:
                        pass
        # Write summary at root metrics dir
        # Judge stats removed (API-only evaluation)
        meta_summary = {"cases": summary}
        # Determine shard-aware naming if shard args used
        shard_id = getattr(args, 'shard_id', None)
        num_shards = getattr(args, 'num_shards', None)
        # Use the effective line range we logged earlier (start_line/end_line still in scope here)
        range_suffix = f"case_ids_{case_id_start}_{case_id_end}" if 'case_id_start' in locals() and 'case_id_end' in locals() else None
        if shard_id is not None and num_shards is not None and range_suffix and 'original_case_start' in locals():
            summary_name = f"summary_metrics_case_ids_{original_case_start}_{original_case_end}_shard{shard_id}.json"
        elif range_suffix:
            summary_name = f"summary_metrics_{range_suffix}.json"
        else:
            summary_name = "summary_metrics.json"
        (eval_output_dir / summary_name).write_text(json.dumps(meta_summary, indent=2))
        logger.info(f"Summary metrics saved -> {(eval_output_dir / summary_name)}")
    # Common cleanup for index_lines mode (if that branch was used)
    if processing_mode == 'index_lines':
        logger.info("Final cleanup (index_lines mode)...")
        logger.info("Cleanup completed.")
        if wb_run:
            try:
                wandb.finish()
            except Exception:
                pass

if __name__ == "__main__":
    main()
