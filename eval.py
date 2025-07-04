from __future__ import annotations

from curses import meta
import argparse, json, os, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import pandas as pd
from dotenv import load_dotenv

###############################################################################
# CONFIGURATION                                                               #
###############################################################################
load_dotenv()

BASE_DIR = Path(os.getenv('BASE_DIR', '.')).resolve()

CASES = [1, 2]

MODELS = ["claude-4-sonnet", "gpt-4o", "llama-70b"]

AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_KEY=os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_API_VERSION=os.getenv('AZURE_OPENAI_API_VERSION')

@dataclass
class EvalPaths:
    base: Path
    data: Path = field(init=False)
    doc: Path = field(init=False)
    ctg: Path = field(init=False)
    model_out: Path = field(init=False)
    output: Path = field(init=False)

    def __post_init__(self):
        self.doc       = self.base / "CTG_DOCUMENT"
        self.data      = self.base / "DATA"
        self.ctg       = self.base / "DATA" / "CTG"
        self.model_out = self.base / "DATA" / "MODEL_OUTPUT"
        self.output    = self.base / "DATA" / "EVAL_METRICS_OUTPUT"
        self.output.mkdir(parents=True, exist_ok=True)

@dataclass
class Thresholds:
    list: float = 0.70
    struct: float = 0.70
    leaf: float = 0.70

###############################################################################
# FIELD METADATA                                                              #
###############################################################################
@dataclass(frozen=True)
class FieldMeta:
    idx: str
    type: str
    is_list: bool
    is_leaf: bool
    target_leaf: bool
    piece: str
    description: str

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
        self.pieces       = dict(zip(clean_indices, df["Piece"]))
        self.order        = dict(zip(clean_indices, df["No"]))
        self.descriptions = dict(zip(clean_indices, df["Description"].fillna("").astype(str)))
        self.rules        = dict(zip(clean_indices, df["Rules"].fillna("").astype(str)))
        self.titles       = dict(zip(clean_indices, df["Title"].fillna("").astype(str)))

    def meta(self, field_key: str) -> FieldMeta:
        clean_key = re.sub(r"\[\d+\]", "", field_key)
        # pick first non-empty: description → rules → title → piece
        desc = (self.descriptions.get(clean_key)
                or self.rules.get(clean_key)
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
            description=desc
        )

###############################################################################
# SEMANTIC SIMILARITY                                                         #
###############################################################################
try:
    import openai
    _AZ_KEY, _AZ_ENDPOINT = AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT
except ImportError:
    openai, _AZ_KEY = None, None

def clean_path(s: str) -> str:
    return re.sub(r'\[\d+\]', '', s)

def _semantic_sim(a: str, b: str, context: str = "") -> float:
    """Compute semantic similarity for clinical trial field comparison."""
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 1.0
    if not (_AZ_KEY and openai):
        raise ValueError("OpenAI API key and endpoint must be set.")
    client = openai.AzureOpenAI(
        api_key=_AZ_KEY,
        azure_endpoint=_AZ_ENDPOINT,
        api_version="2024-02-01"
    )
    # prepend a tiny clinical-trial context + field description
    prefix = "Compare these clinical trial field values"
    if context:
        prefix += f" for '{context}'"
    prompt = (
        f"{prefix}. Return only a number 0-1 for semantic similarity.\n"
        f"Text1: {a}\nText2: {b}"
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            max_tokens=4,
            temperature=0
        )
        return float(r.choices[0].message.content)
    except Exception:
        return 0.0

_norm = lambda v: " ".join(str(v).lower().split()) if v is not None else ""

def compare_scalar(p: Any, r: Any, meta: FieldMeta, th: Thresholds) -> Tuple[bool,float]:
    sp, sr = _norm(p), _norm(r)
    if sp == sr:
        return True, 1.0
    if not sp or not sr:
        return False, 0.0
    # pass the field description as context
    s = _semantic_sim(sp, sr, context=meta.description)
    return (s >= th.leaf), s

def compare_struct(p: Any, r: Any, parent_meta: FieldMeta, th: Thresholds) -> Tuple[bool,float]:
    p_str = json.dumps(p, ensure_ascii=False, indent=2)
    r_str = json.dumps(r, ensure_ascii=False, indent=2)
    s = _semantic_sim(p_str, r_str, context=parent_meta.description)
    return (s >= th.struct), s

def compare_any(p: Any, r: Any, meta: FieldMeta, cat: FieldCatalog, th: Thresholds) -> Tuple[bool,float]:
    if isinstance(p, dict) or isinstance(r, dict):
        return compare_struct(p or {}, r or {}, meta, th)
    return compare_scalar(p, r, meta, th)

def compare_list(p_list: Sequence[Any], r_list: Sequence[Any], meta: FieldMeta, cat: FieldCatalog, th: Thresholds) -> Tuple[bool, float]:
    if not r_list:
        print("\n *** [def compare_list - if not r_list:] This should not happen *** \n")
        return True, 1.0
    if not p_list:
        print("\n *** [def compare_list - if not p_list:] This should not happen *** \n")
        return False, 0.0
    if meta.type != "STRUCT":
        max_len = max(len(p_list), len(r_list))
        sims = [compare_scalar(p_list[i] if i < len(p_list) else None, r_list[i] if i < len(r_list) else None, meta, th)[1] for i in range(max_len)]
        sim = sum(sims) / len(sims)
        return sim >= th.list, sim
    # struct list handled by Evaluator's alignment – return placeholder
    return False, 0.0

###############################################################################
# RECORD & METRICS                                                            #
###############################################################################
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
    piece: str|None
    norm_path: str|None
    from_pred: bool = False

###############################################################################
# EVALUATOR                                                                   #
###############################################################################
class Evaluator:
    def __init__(self, paths: EvalPaths, th: Thresholds, cat: FieldCatalog):
        self.paths, self.th, self.cat = paths, th, cat
        self._tok = re.compile(r"([^\.\[\]]+)(?:\[(\d+)\])?")

    # Parallel flatten with list alignment (unchanged from v3) -------------
    def _flatten_parallel(self, ref: Any, pred: Any, base: str="") -> Tuple[Dict[str,Any],Dict[str,Any]]:
        ref_flat, pred_flat, pred_flag = {}, {}, {}
        
        def emit(path,rv,pv,from_pred=False): 
            ref_flat[path]=rv
            pred_flat[path]=pv
            pred_flag[path] = pred_flag.get(path, False) or from_pred

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
                r_child=ref.get(k) if isinstance(ref,dict) else None
                pred_has_key = isinstance(pred, dict) and (k in pred)
                p_child = pred.get(k) if pred_has_key else None
                r_map,p_map,flag_map=self._flatten_parallel(r_child,p_child,child)
                ref_flat.update(r_map); pred_flat.update(p_map); pred_flag.update(flag_map);
        
        elif isinstance(ref, list) or isinstance(pred, list):
            ref_l, pred_l = ref or [], pred or []
            clean_key = re.sub(r"\[\d+\]", "", base)
            meta = self.cat.meta(clean_key)

            is_scalar_list = meta.is_list and meta.type != "STRUCT" and not any(isinstance(x, (dict, list)) for x in ref_l + pred_l)
            if is_scalar_list:
                emit(base, ref_l, pred_l, from_pred=bool(pred_l))

            else:
                mapping = self._align_struct_list(ref_l, pred_l, base)
                for i, j in mapping:
                    child = f"{base}[{i if i is not None else j}]"
                    r_child = ref_l[i] if i is not None else None
                    p_child = pred_l[j] if j is not None else None
                    child_from_pred = j is not None
                    r_map, p_map, flag_map = self._flatten_parallel(r_child, p_child, child)
                    ref_flat.update(r_map)
                    pred_flat.update(p_map)
                    pred_flag.update(flag_map)

        else:
            emit(base, ref, pred, from_pred=(pred is not None))
            
        return ref_flat, pred_flat, pred_flag

    def _align_struct_list(self, ref_l: List[Any], pred_l: List[Any], base: str) -> List[Tuple[int|None,int|None]]:
        clean_key = re.sub(r"\[\d+\]", "", base)
        list_meta = self.cat.meta(clean_key)
        sim_matrix: List[List[float]] = []
        for rv in ref_l:
            row = []
            for pv in pred_l:
                _, sim = compare_any(pv, rv, list_meta, self.cat, self.th)
                row.append(sim)
            sim_matrix.append(row)
        used = set()
        pairs: List[Tuple[int|None,int|None]] = []
        for i, row in enumerate(sim_matrix):
            best_j = None
            best_sim = 0.0
            for j, sim in enumerate(row):
                if j in used: continue
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            # require minimum similarity to align
            if best_j is not None and best_sim >= self.th.struct:
                used.add(best_j)
                pairs.append((i, best_j))
            else:
                pairs.append((i, None))
        return pairs

    # ---------------------- CASE EVAL ------------------------------------
    def evaluate_case(self, cid: int, model: str) -> Tuple[List[EvalRec], Dict[str, float]]:
        case_dir = f"case_{cid}"

        ref_files = list(self.paths.ctg.glob(f"{case_dir}_*.json"))
        if not ref_files:
            raise FileNotFoundError(f"No CTG file for case {cid} in {self.paths.ctg}")
        ref_path = ref_files[0]

        pred_dir = self.paths.model_out / case_dir
        pred_files = list(pred_dir.glob(f"*{model}*.json"))
        if not pred_files:
            raise FileNotFoundError(f"No prediction file for case {cid}, model {model} in {pred_dir}")
        pred_path = pred_files[0]

        ref_data  = json.loads(ref_path.read_text())
        pred_data = json.loads(pred_path.read_text())

        ref_flat, pred_flat, pred_flag = self._flatten_parallel(ref_data, pred_data)
        
        all_keys: List[str] = list(ref_flat.keys())
        for k in pred_flat.keys():
            if k not in all_keys:
                all_keys.append(k)

        recs: List[EvalRec] = []
        
        ref_norms = {re.sub(r"\[\d+\]", "", k): v for k, v in ref_flat.items()}
        pred_norms = {re.sub(r"\[\d+\]", "", k): v for k, v in pred_flat.items()}
        

        for flat_path in all_keys:
            flag_pred = pred_flag.get(flat_path, False)
            norm_key = re.sub(r"\[\d+\]", "", flat_path)
            pv = pred_flat.get(flat_path)
            rv = ref_flat.get(flat_path)

            meta  = self.cat.meta(norm_key) if norm_key in self.cat.field_indexes else None
            piece = meta.piece if meta else None
        

            if rv is None and pv is not None:
                if norm_key not in self.cat.field_indexes:
                    recs.append(EvalRec(model, flat_path, "FP", "FP", None, pv, None, None, None, norm_key, flag_pred))
                    continue
                else:
                    recs.append(EvalRec(model, flat_path, "EV", "EV", None, pv, None, None, piece, norm_key, flag_pred))
                    continue
            elif rv is not None and pv is None:
                if flag_pred and meta:
                    recs.append(EvalRec(model, flat_path, "TP", "FP", rv, None, None, None, piece, norm_key, flag_pred))
                    continue
                else:
                    recs.append(EvalRec(model, flat_path, "FN", "FN", rv, None, None, None, piece, norm_key, flag_pred))
                    continue
            elif rv is None and pv is None:
                if flag_pred and meta:
                    recs.append(EvalRec(model, flat_path, "EV", "TN", None, None, None, None, None, norm_key, flag_pred))
                    continue
                if flag_pred and meta is None:
                    recs.append(EvalRec(model, flat_path, "FP", "TN", None, None, None, None, piece, norm_key, flag_pred))
                    continue
                if not flag_pred and meta:
                    recs.append(EvalRec(model, flat_path, "NA", "NA", None, None, None, None, None, norm_key, flag_pred))
                    continue
            elif rv is not None and pv is not None:
                if isinstance(pv, dict) or isinstance(rv, dict):
                    raise ValueError("Should not happen: struct should not be in here")
                value_match, sim = compare_any(pv, rv, meta, self.cat, self.th)
                if value_match:
                    recs.append(EvalRec(model, flat_path, "TP", "TP", rv, pv, value_match, sim, piece, norm_key, flag_pred))
                else:
                    recs.append(EvalRec(model, flat_path, "TP", "FP", rv, pv, value_match, sim, piece, norm_key, flag_pred))
            else:
                print("\n *** [def evaluate_case - final Else] Unexpected Case. {} \n".format(flat_path))
                print("rv: {}, pv: {}".format(rv, pv))
                print("meta: {}, norm_key: {}".format(meta, norm_key))
                
        return recs

# ---------------------------------------------------------------------------
# METRICS (external helper to avoid forward ref) -----------------------------
# ---------------------------------------------------------------------------
def calc_metrics(records: List[EvalRec]) -> Dict[str, float]:
    # Count key‐level categories per flat field
    key_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    # Count value‐level categories per flat field
    val_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for r in records:
        # each r.field is the flat key
        key_counts[r.key_cat] = key_counts.get(r.key_cat, 0) + 1
        val_counts[r.val_cat] = val_counts.get(r.val_cat, 0) + 1

    def prf(counts):
        tp = counts.get("TP", 0)
        fp = counts.get("FP", 0)
        fn = counts.get("FN", 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall    = tp / (tp + fn) if tp + fn else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    key_p, key_r, key_f1 = prf(key_counts)
    val_p, val_r, val_f1 = prf(val_counts)

    total = sum(key_counts.values())

    print(f"Key counts: {key_counts}")
    print(f"Value counts: {val_counts}")
    print(f"Key P/R/F1: {key_p:.3f}/{key_r:.3f}/{key_f1:.3f}")
    print(f"Value P/R/F1: {val_p:.3f}/{val_r:.3f}/{val_f1:.3f}")
    print(f"Total fields evaluated: {len(records)}")
    key_EV_fields = [r.flat_path for r in records if r.key_cat == "EV"]
    key_FN_fields = [r.flat_path for r in records if r.key_cat == "FN"]
    key_FP_fields = [r.flat_path for r in records if r.key_cat == "FP"]
    key_TP_fields = [r.flat_path for r in records if r.key_cat == "TP"]
    key_TN_fields = [r.flat_path for r in records if r.key_cat == "TN"]
    val_EV_fields = [r.flat_path for r in records if r.val_cat == "EV"]
    val_FN_fields = [r.flat_path for r in records if r.val_cat == "FN"]
    val_FP_fields = [r.flat_path for r in records if r.val_cat == "FP"]
    val_TP_fields = [r.flat_path for r in records if r.val_cat == "TP"]
    val_TN_fields = [r.flat_path for r in records if r.val_cat == "TN"]

    return {
        "total":           total,
        "key_precision":   key_p,
        "key_recall":      key_r,
        "key_f1":          key_f1,
        "value_precision": val_p,
        "value_recall":    val_r,
        "value_f1":        val_f1,
        "key_counts":      key_counts,
        "value_counts":    val_counts,
        "key_EV_fields":   key_EV_fields,
        "key_FN_fields":   key_FN_fields,
        "key_FP_fields":   key_FP_fields,
        "key_TP_fields":   key_TP_fields,
        "key_TN_fields":   key_TN_fields,
        "val_EV_fields":   val_EV_fields,
        "val_FN_fields":   val_FN_fields,
        "val_FP_fields":   val_FP_fields,
        "val_TP_fields":   val_TP_fields,
        "val_TN_fields":   val_TN_fields,
    }

# ---------------------------------------------------------------------------
# -----------------------------  CLI / MAIN  ---------------------------------
# ---------------------------------------------------------------------------


def main():
    paths = EvalPaths(Path(BASE_DIR))
    global catalog
    catalog = FieldCatalog(paths.doc / "DataDefinition.csv", paths.doc / "Enums.json")
    evalr = Evaluator(paths, Thresholds(), catalog)

    summary: Dict[str, Dict[str, float]] = {}

    for cid in CASES:
        for model in MODELS:
            try:
                recs = evalr.evaluate_case(cid, model)
                # write recs to CSV
                df = pd.DataFrame([r.__dict__ for r in recs])
                csv_out = paths.output / f"case{cid}_{model}_records.csv"
                df.to_csv(csv_out, index=False)
                print(f"Total records evaluated: {len(recs)}. Saving to {csv_out}")

                met = calc_metrics(recs)
                out = paths.output / f"case{cid}_{model}_metrics.json"
                out.write_text(json.dumps(met, indent=2))
                print(f"Case {cid}, Model {model} | Saving metrics to {out}")
                
            except FileNotFoundError as e:
                print(e)
                continue
            summary[f"case{cid}_{model}"] = met
        (paths.output / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
        print("path to summary:", paths.output / "summary_metrics.json")
    print("Evaluation finished.")

if __name__ == "__main__":
    main()
