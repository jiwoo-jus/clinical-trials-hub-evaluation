"""Refactored Evaluation Statistics Generator

Usage:
python generate_eval_stats.py --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import yaml
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import pandas as pd
import numpy as np
from typing import Set

# Custom color palette for plots
CUSTOM_PALETTE = ['#de324c', '#f4895f', '#f8e16f', '#95cf92', '#369acc', '#9656a2']

# Custom heatmap colormap
CUSTOM_HEATMAP_CMAP = LinearSegmentedColormap.from_list("custom_heatmap", ["#fc0217", "#f7fc62", "#90f47a"])

# Model colors (pastel tones) - use internal model names as keys
MODEL_COLORS = {
    'gemini-3-pro-preview': '#B4E7D8',      # Mint green
    'gpt-5.1-2025-11-13': '#F9A8A8',        # Coral
    'Anthropic Claude 4.5 Sonnet': '#BAE0ED',  # Soft blue
    # Fallbacks or short names if they appear differently
    'Gemini-3-Pro': '#95E1D3',
    'GPT-5.1': '#F38181',
    'Claude-4.5-Sonnet': '#A8D8EA'
}
# Fallback list if model name not found
FALLBACK_PALETTE = ['#B4E7D8', '#F9A8A8', '#BAE0ED', '#f8e16f', '#95cf92', '#369acc']

PLOT_STYLE = {
    'alpha': 0.85,
    'edgecolor': '#2C3E50',
    'linewidth': 1.5,
    'title_fontsize': 14,
    'label_fontsize': 12,
    'tick_fontsize': 11,
    'value_fontsize': 10,
    'grid_alpha': 0.3,
    'dpi': 300
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Config:
    """Configuration container for all settings"""
    # Paths
    index_csv: Path
    output_base: Path
    field_catalog_path: Path
    
    # Per-model paths
    models: List[str]
    model_metrics_paths: Dict[str, Path] = field(default_factory=dict)
    # Optional per-section metrics roots
    model_metrics_paths_by_section: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    
    # Processing options
    limit: Optional[int] = None
    extended: bool = False
    metrics_subdir: str = 'metrics'
    variants: List[str] = field(default_factory=lambda: ['metrics'])
    
    # Plotting
    plots: bool = False
    plot_output_dir: Optional[Path] = None
    plot_top_fields: int = 40
    title_model_bar: str = 'Model Value F1 (All Cases)'
    title_field_heatmap: str = 'Field Performance Heatmap (Value F1)'
    xlabel_model_bar: str = 'Model'
    ylabel_model_bar: str = 'Value F1'
    cmap_heatmap: str = 'viridis'
    
    # Advanced plotting
    adv_plots: bool = False
    no_heatmap: bool = False
    top_fields_bar: int = 30
    bottom_fields_bar: int = 20
    no_radar: bool = False
    annotate_threshold: float = 0.3
    max_annotate: int = 25
    scatter_alpha: float = 0.75
    dpi: int = 150
    style: str = 'whitegrid'
    
    # Per-model field ranking options
    per_model_top_fields: int = 20
    per_model_bottom_fields: int = 20
    per_model_field_min_support: int = 3
    
    # Target fields configuration (from eval config style)
    target_fields: Dict[str, List[str]] = field(default_factory=dict)
    
    # Section-based evaluation
    evaluate_sections: List[str] = field(default_factory=lambda: ['protocol', 'results', 'both'])

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Config:
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert basic paths
        for key in ['index_csv', 'output_base', 'field_catalog_path', 'plot_output_dir']:
            if key in data and data[key]:
                data[key] = Path(data[key]).resolve()
        
        # Handle optional field catalog
        data.setdefault('field_catalog_path', 
                       Path("/{YOURPATH}/docs/data_definition.csv"))
        
        # Ensure models list exists
        data.setdefault('models', [])
        
        # Convert per-model path dictionaries
        for path_key in ['model_metrics_paths']:
            if path_key in data and isinstance(data[path_key], dict):
                data[path_key] = {k: Path(v).resolve() for k, v in data[path_key].items() if v}
            else:
                data[path_key] = {}

        # Optional: per-section metrics paths mapping
        if 'model_metrics_paths_by_section' in data and isinstance(data['model_metrics_paths_by_section'], dict):
            sec_map: Dict[str, Dict[str, Path]] = {}
            for sec, inner in data['model_metrics_paths_by_section'].items():
                if isinstance(inner, dict):
                    sec_map[str(sec)] = {k: Path(v).resolve() for k, v in inner.items() if v}
            data['model_metrics_paths_by_section'] = sec_map
        else:
            data['model_metrics_paths_by_section'] = {}
        
        # If models not specified but path maps exist, derive from keys
        if not data.get('models'):
            all_model_keys = set()
            for path_key in ['model_metrics_paths']:
                all_model_keys.update(data.get(path_key, {}).keys())
            if all_model_keys:
                data['models'] = sorted(list(all_model_keys))
        
        # Load target_fields if provided
        if 'target_fields' not in data or not data['target_fields']:
            data['target_fields'] = {}
        
        # Load evaluate_sections if provided
        if 'evaluate_sections' not in data:
            data['evaluate_sections'] = ['protocol', 'results', 'both']
        
        return cls(**data)



@dataclass
class FieldInfo:
    field_index: str
    piece: str
    is_target: bool
    is_leaf: bool
    type: str
    is_list: bool


# ============================================================================
# Field Catalog
# ============================================================================

class FieldCatalog:
    def __init__(self, csv_path: Path, target_fields: Optional[Dict[str, List[str]]] = None):
        """Initialize field catalog with optional target_fields override
        
        Args:
            csv_path: Path to DataDefinition CSV
            target_fields: Dict mapping section names to list of Piece names (from eval config style)
                          e.g., {'protocolSection': ['NCTId', 'BriefTitle', ...], 
                                 'resultsSection': ['BaselineGroupId', ...]}
        """
        self.df = pd.read_csv(csv_path)
        self.df['NormalizedIndex'] = self.df['FieldIndex'].str.replace(r'\[\d+\]', '', regex=True)
        
        # If target_fields provided, use it; otherwise use DataDefinition TargetYN
        if target_fields:
            self.target_fields = target_fields
            # Build target set from Piece names in target_fields
            target_pieces = set()
            for section, pieces in target_fields.items():
                target_pieces.update(pieces)
            
            # Mark fields as target if their Piece is in target_fields
            self.df['IsTarget'] = self.df['Piece'].isin(target_pieces)
            self.target_leaf_df = self.df[self.df['IsTarget'] & (self.df['IsLeaf'] == 'Y')]
            self.target_leaf_indices = set(self.target_leaf_df['FieldIndex'])
        else:
            self.target_fields = None
            self.target_leaf_df = self.df[(self.df['TargetYN'] == 'Y') & (self.df['IsLeaf'] == 'Y')]
            self.target_leaf_indices = set(self.target_leaf_df['FieldIndex'])
        
        self.field_map: Dict[str, FieldInfo] = {}
        
        for _, row in self.df.iterrows():
            self.field_map[row['FieldIndex']] = FieldInfo(
                field_index=row['FieldIndex'],
                piece=row['Piece'],
                is_target=row['TargetYN'] == 'Y',
                is_leaf=row['IsLeaf'] == 'Y',
                type=row['SourceType'],
                is_list=row['ListYN'] == 'Y'
            )
    
    def normalize(self, path: str) -> str:
        return re.sub(r'\[\d+\]', '', path)
    
    def is_target_leaf(self, path: str) -> bool:
        return self.normalize(path) in self.target_leaf_indices


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_case_list(index_csv: Path, limit: Optional[int]) -> List[Tuple[int, str]]:
    """Load case IDs and years from index CSV"""
    df = pd.read_csv(index_csv)
    # Handle missing year column
    if 'year' not in df.columns:
        df['year'] = 0
    
    sub = df[['case_id', 'year']]
    if limit:
        sub = sub.head(limit)
    return list(sub.itertuples(index=False, name=None))


def find_shard_dir(model_root: Path, case_id: int) -> Optional[Path]:
    """Find shard directory containing given case_id"""
    for sd in model_root.iterdir():
        if not sd.is_dir():
            continue
        m = re.match(r"^(\d+)_(\d+)$", sd.name)
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        if s <= case_id <= e:
            return sd
    return None


def read_metrics_shard(metrics_path: Path, case_id: int, 
                      metrics_subdir: str = 'metrics', 
                      section_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read metrics for a specific case from shard structure
    Supports both protocol and results metrics files:
    - case{case_id}_protocol_metrics.json
    - case{case_id}_results_metrics.json
    
    Args:
        metrics_path: Path to metrics directory
        case_id: Case ID
        metrics_subdir: Subdirectory name (e.g., 'metrics')
        section_filter: Filter to 'protocol', 'results', or None for both
    """
    if not metrics_path.exists():
        return None
    
    shard_dir = find_shard_dir(metrics_path, case_id)
    if not shard_dir:
        return None
    
    metrics_dir = shard_dir / metrics_subdir
    if not metrics_dir.exists():
        return None
    
    # Try different filename patterns based on section_filter
    if section_filter == 'protocol':
        patterns = [f"case{case_id}_protocol_metrics.json"]
    elif section_filter == 'results':
        patterns = [f"case{case_id}_results_metrics.json"]
    else:  # both or None
        patterns = [
            f"case{case_id}_protocol_metrics.json",
            f"case{case_id}_results_metrics.json",
            f"case{case_id}_metrics.json",
        ]
    
    all_metrics = []
    for pattern in patterns:
        fp = metrics_dir / pattern
        if fp.exists():
            try:
                data = json.loads(fp.read_text())
                # Add section info to metadata
                if '_protocol_' in pattern:
                    data['section'] = 'protocol'
                elif '_results_' in pattern:
                    data['section'] = 'results'
                all_metrics.append(data)
            except Exception as e:
                print(f"Failed to parse {fp}: {e}")
    
    if not all_metrics:
        return None
    
    # If multiple metrics files exist, merge them
    if len(all_metrics) == 1:
        return all_metrics[0]
    
    # Merge multiple metrics (protocol + results)
    merged = {
        'case_id': case_id,
        'section': 'both',
        'total': sum(m.get('total', 0) for m in all_metrics),
        'key_counts': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'EV': 0, 'NA': 0},
        'value_counts': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'EV': 0, 'NA': 0},
    }
    
    for m in all_metrics:
        for k in ['TP', 'FP', 'FN', 'TN', 'EV', 'NA']:
            merged['key_counts'][k] += m.get('key_counts', {}).get(k, 0)
            merged['value_counts'][k] += m.get('value_counts', {}).get(k, 0)
    
    # Recalculate precision, recall, F1
    kc = merged['key_counts']
    vc = merged['value_counts']
    
    merged['key_precision'] = kc['TP'] / (kc['TP'] + kc['FP']) if (kc['TP'] + kc['FP']) > 0 else 0.0
    merged['key_recall'] = kc['TP'] / (kc['TP'] + kc['FN']) if (kc['TP'] + kc['FN']) > 0 else 0.0
    merged['key_f1'] = (2 * merged['key_precision'] * merged['key_recall'] / 
                        (merged['key_precision'] + merged['key_recall'])) if (merged['key_precision'] + merged['key_recall']) > 0 else 0.0
    
    merged['value_precision'] = vc['TP'] / (vc['TP'] + vc['FP']) if (vc['TP'] + vc['FP']) > 0 else 0.0
    merged['value_recall'] = vc['TP'] / (vc['TP'] + vc['FN']) if (vc['TP'] + vc['FN']) > 0 else 0.0
    merged['value_f1'] = (2 * merged['value_precision'] * merged['value_recall'] / 
                          (merged['value_precision'] + merged['value_recall'])) if (merged['value_precision'] + merged['value_recall']) > 0 else 0.0
    
    # Merge field lists
    for field_type in ['key_TP_fields', 'key_FP_fields', 'key_FN_fields', 'key_EV_fields',
                       'val_TP_fields', 'val_FP_fields', 'val_FN_fields', 'val_EV_fields']:
        merged[field_type] = []
        for m in all_metrics:
            merged[field_type].extend(m.get(field_type, []))
    
    return merged





# ============================================================================
# Core Processing
# ============================================================================

class MetricsProcessor:
    def __init__(self, config: Config, catalog: FieldCatalog):
        self.config = config
        self.catalog = catalog

    @staticmethod
    def _merge_metrics_dicts(metrics_list: List[Optional[Dict[str, Any]]], case_id: int) -> Optional[Dict[str, Any]]:
        """Merge multiple per-case metrics dicts (e.g., protocol + results) into one.

        Expects keys like 'total', 'key_counts', 'value_counts', and field lists such as
        'key_TP_fields', 'key_FP_fields', 'key_FN_fields', 'key_EV_fields',
        'val_TP_fields', 'val_FP_fields', 'val_FN_fields', 'val_EV_fields'.
        """
        mlist = [m for m in metrics_list if m]
        if not mlist:
            return None
        if len(mlist) == 1:
            return mlist[0]

        merged: Dict[str, Any] = {
            'case_id': case_id,
            'section': 'both',
            'total': 0,
            'key_counts': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'EV': 0, 'NA': 0},
            'value_counts': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'EV': 0, 'NA': 0},
        }
        # Sum totals and counts
        for m in mlist:
            merged['total'] += int(m.get('total', 0))
            for cat in ['TP', 'FP', 'FN', 'TN', 'EV', 'NA']:
                merged['key_counts'][cat] += int(m.get('key_counts', {}).get(cat, 0))
                merged['value_counts'][cat] += int(m.get('value_counts', {}).get(cat, 0))

        # Recompute precision/recall/F1
        kc = merged['key_counts']
        vc = merged['value_counts']
        kp = kc['TP'] / (kc['TP'] + kc['FP']) if (kc['TP'] + kc['FP']) > 0 else 0.0
        kr = kc['TP'] / (kc['TP'] + kc['FN']) if (kc['TP'] + kc['FN']) > 0 else 0.0
        merged['key_precision'] = kp
        merged['key_recall'] = kr
        merged['key_f1'] = (2 * kp * kr / (kp + kr)) if (kp + kr) > 0 else 0.0

        vp = vc['TP'] / (vc['TP'] + vc['FP']) if (vc['TP'] + vc['FP']) > 0 else 0.0
        vr = vc['TP'] / (vc['TP'] + vc['FN']) if (vc['TP'] + vc['FN']) > 0 else 0.0
        merged['value_precision'] = vp
        merged['value_recall'] = vr
        merged['value_f1'] = (2 * vp * vr / (vp + vr)) if (vp + vr) > 0 else 0.0

        # Merge field lists
        list_keys = ['key_TP_fields', 'key_FP_fields', 'key_FN_fields', 'key_EV_fields',
                     'val_TP_fields', 'val_FP_fields', 'val_FN_fields', 'val_EV_fields']
        for lk in list_keys:
            acc: List[Any] = []
            for m in mlist:
                acc.extend(m.get(lk, []) or [])
            merged[lk] = acc

        return merged

    def process_variant(self, variant: str, section: str = 'both') -> Dict[str, Any]:
        """Process a single variant and return aggregated results
        
        Args:
            variant: Metrics subdirectory name
            section: 'protocol', 'results', or 'both'
        """
        section_filter = section if section != 'both' else None
        print(f"\n===== VARIANT: {variant} | SECTION: {section} =====")
        
        cases = load_case_list(self.config.index_csv, self.config.limit)
        
        # Initialize data structures
        per_case_rows: List[Dict[str, Any]] = []
        field_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        missing: Dict[str, List[int]] = {m: [] for m in self.config.models}
        case_presence: Dict[int, Dict[str, Any]] = {}
        extended_field_events: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # EV analysis data structures
        ev_per_case: Dict[str, List[int]] = {m: [] for m in self.config.models}  # EV counts per case per model
        ev_field_occurrences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # piece -> model -> count
        ev_total_evaluated: Dict[str, int] = {m: 0 for m in self.config.models}  # Total evaluated fields per model
        # Track field occurrences in reference and predictions
        ref_field_cases: Dict[str, Set[int]] = defaultdict(set)  # piece -> set of case_ids where it appears in reference
        pred_field_cases: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))  # piece -> model -> set of case_ids
        tp_field_cases: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))  # piece -> model -> set of case_ids with TP
        ev_field_cases: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))  # piece -> model -> set of case_ids with EV but no TP
        
        # Process each case
        for case_id, year in cases:
            if case_id not in case_presence:
                case_presence[case_id] = {'year': year, 'present': set()}
            
            for model in self.config.models:
                # Resolve metrics path(s) for this section/model
                data: Optional[Dict[str, Any]] = None
                by_sec = self.config.model_metrics_paths_by_section or {}
                if section == 'both' and by_sec.get('protocol', {}).get(model) and by_sec.get('results', {}).get(model):
                    mp_proto = by_sec['protocol'][model]
                    mp_res = by_sec['results'][model]
                    d_proto = read_metrics_shard(mp_proto, case_id, metrics_subdir=variant, section_filter='protocol')
                    d_res = read_metrics_shard(mp_res, case_id, metrics_subdir=variant, section_filter='results')
                    data = self._merge_metrics_dicts([d_proto, d_res], case_id)
                else:
                    # Prefer section-specific path if available
                    metrics_path = None
                    if section_filter and by_sec.get(section_filter, {}).get(model):
                        metrics_path = by_sec[section_filter][model]
                    else:
                        metrics_path = self.config.model_metrics_paths.get(model)
                    if metrics_path:
                        data = read_metrics_shard(metrics_path, case_id, metrics_subdir=variant, section_filter=section_filter)
                
                if not data:
                    missing[model].append(case_id)
                    continue
                
                case_presence[case_id]['present'].add(model)
                
                # Collect TP pieces for this case FIRST (before processing EV)
                # Use KEY level to align with requested key-only basis
                tp_pieces_this_case = set()
                for field_path in data.get('key_TP_fields', []):
                    norm = self.catalog.normalize(field_path)
                    if norm in self.catalog.field_map:
                        tp_pieces_this_case.add(self.catalog.field_map[norm].piece)
                
                # Collect EV statistics (KEY level)
                ev_fields = data.get('key_EV_fields', [])
                ev_count = len(ev_fields) if isinstance(ev_fields, list) else 0
                ev_per_case[model].append(ev_count)
                
                # Track total evaluated fields for EV rate calculation (KEY level)
                kc = data.get('key_counts', {})
                ev_total_evaluated[model] += (kc.get('TP', 0) + kc.get('FP', 0) + 
                                              kc.get('FN', 0) + kc.get('EV', 0))
                
                # Track which fields appear as EV for this model (KEY level, no target filter)
                for ev_field in ev_fields:
                    if isinstance(ev_field, str):
                        # Normalize field path and map to Piece
                        normalized = self.catalog.normalize(ev_field)
                        if normalized in self.catalog.field_map:
                            piece = self.catalog.field_map[normalized].piece
                            ev_field_occurrences[piece][model] += 1
                            # Also track pred_case for EV fields (KEY level)
                            pred_field_cases[piece][model].add(case_id)
                            # Track EV-only cases: piece has EV but no TP in this case
                            if piece not in tp_pieces_this_case:
                                ev_field_cases[piece][model].add(case_id)
                
                # Record per-case metrics
                per_case_rows.append({
                    'case_id': case_id,
                    'year': year,
                    'model': model,
                    'total': data.get('total', 0),
                    'key_precision': data.get('key_precision', 0.0),
                    'key_recall': data.get('key_recall', 0.0),
                    'key_f1': data.get('key_f1', 0.0),
                    'value_precision': data.get('value_precision', 0.0),
                    'value_recall': data.get('value_recall', 0.0),
                    'value_f1': data.get('value_f1', 0.0)
                })
                
                # Process field-level metrics (KEY-only for ref/pred case counts)
                for cat_key in ['key_TP_fields', 'key_FP_fields', 'key_FN_fields', 'key_EV_fields']:
                    cat = cat_key.split('_')[1]
                    for field_path in data.get(cat_key, []):
                        if not self.catalog.is_target_leaf(field_path):
                            continue
                        
                        norm = self.catalog.normalize(field_path)
                        piece = self.catalog.field_map.get(
                            norm, FieldInfo(norm, norm, True, True, 'TEXT', False)
                        ).piece
                        # Keep per-model field_counts accumulating KEY categories with 'k' prefix
                        kcat = 'k' + cat
                        field_counts[piece][model][kcat] += 1
                        
                        # Track field occurrences in reference and predictions (KEY level)
                        if cat in ['TP', 'FN']:  # Field key exists in reference
                            ref_field_cases[piece].add(case_id)
                        if cat == 'TP':  # Track TP separately for EV-only calculation (KEY)
                            tp_field_cases[piece][model].add(case_id)
                        if cat in ['TP', 'FP', 'EV']:  # Field key exists in prediction (including EV)
                            pred_field_cases[piece][model].add(case_id)
                        
                        if self.config.extended:
                            extended_field_events[piece][model][cat] += 1

                # Note: we already handled KEY categories above; no separate VALUE block needed.
        
        # Create dataframes
        per_case_df = pd.DataFrame(per_case_rows)
        model_df = self._create_model_summary(per_case_df)
        field_df = self._create_field_summary(field_counts)
        
        # Analyze completeness
        complete_analysis = self._analyze_completeness(
            cases, case_presence, per_case_df
        )
        
        # Per-model field metrics and rankings
        per_model_field_metrics_df = self._create_per_model_field_metrics(field_counts)
        top_df, bottom_df = self._rank_fields_per_model(per_model_field_metrics_df)
        
        # Extended analysis
        extended_results = {}
        if self.config.extended:
            extended_results = self._create_extended_analysis(extended_field_events)

        # Field occurrence counts from index (Instruction vs GT) for heatmap labels
        case_ids_used = {cid for cid, _ in cases}
        inst_counts, gt_counts = self._compute_field_occurrence_counts(case_ids_used)
        
        # Create EV analysis dataframes
        ev_model_summary_df = self._create_ev_model_summary(ev_per_case, ev_total_evaluated)
        ev_field_ranking_df = self._create_ev_field_ranking(ev_field_occurrences, ref_field_cases, pred_field_cases, ev_field_cases)
        
        return {
            'per_case_df': per_case_df,
            'model_df': model_df,
            'field_df': field_df,
            'missing': missing,
            'case_count_expected': len(cases),
            'per_model_field_metrics_df': per_model_field_metrics_df,
            'per_model_field_top_df': top_df,
            'per_model_field_bottom_df': bottom_df,
            **complete_analysis,
            **extended_results,
            'field_inst_counts': inst_counts,
            'field_gt_counts': gt_counts,
            'extended_enabled': self.config.extended,
            'ev_model_summary_df': ev_model_summary_df,
            'ev_field_ranking_df': ev_field_ranking_df
        }
    
    def _create_model_summary(self, per_case_df: pd.DataFrame) -> pd.DataFrame:
        """Create model-level summary statistics"""
        model_rows = []
        if not per_case_df.empty:
            for model in self.config.models:
                g = per_case_df[per_case_df['model'] == model]
                if g.empty:
                    continue
                model_rows.append({
                    'model': model,
                    'cases': g.shape[0],
                    'key_precision_avg': g['key_precision'].mean(),
                    'key_recall_avg': g['key_recall'].mean(),
                    'key_f1_avg': g['key_f1'].mean(),
                    'value_precision_avg': g['value_precision'].mean(),
                    'value_recall_avg': g['value_recall'].mean(),
                    'value_f1_avg': g['value_f1'].mean(),
                })
        return pd.DataFrame(model_rows)
    
    def _create_field_summary(self, field_counts: Dict) -> pd.DataFrame:
        """Create field-level summary statistics"""
        field_rows = []
        for piece, models_map in field_counts.items():
            match = self.catalog.df[self.catalog.df['Piece'] == piece].head(1)
            field_index = match['FieldIndex'].iloc[0] if not match.empty else piece
            row = {'Piece': piece, 'FieldIndex': field_index}
            
            for model in self.config.models:
                md = models_map.get(model, {})
                row[f'{model}_TP'] = md.get('TP', 0)
                row[f'{model}_FP'] = md.get('FP', 0)
                row[f'{model}_FN'] = md.get('FN', 0)
                row[f'{model}_EV'] = md.get('EV', 0)
                # Optional: per-field KEY counts if present
                row[f'{model}_kTP'] = md.get('kTP', 0)
                row[f'{model}_kFP'] = md.get('kFP', 0)
                row[f'{model}_kFN'] = md.get('kFN', 0)
            field_rows.append(row)
        
        return pd.DataFrame(field_rows)

    def _create_per_model_field_metrics(self, field_counts: Dict[str, Dict[str, Dict[str, int]]]) -> pd.DataFrame:
        """Flatten field_counts into tall per-model metrics with Precision/Recall/F1."""
        rows: List[Dict[str, Any]] = []
        for piece, models_map in field_counts.items():
            for m in self.config.models:
                cats = models_map.get(m, {})
                tp = cats.get('TP', 0)
                fp = cats.get('FP', 0)
                fn = cats.get('FN', 0)
                ev = cats.get('EV', 0)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                support = tp + fp + fn  # evaluated events contributing to P/R/F1
                rows.append({
                    'Piece': piece,
                    'model': m,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'EV': ev,
                    'Support': support,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                })
        return pd.DataFrame(rows)

    def _rank_fields_per_model(self, per_model_field_metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create top/bottom field rankings per model with support filtering."""
        if per_model_field_metrics_df is None or per_model_field_metrics_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        df = per_model_field_metrics_df.copy()
        df = df[df['Support'] >= self.config.per_model_field_min_support]
        top_rows = []
        bot_rows = []
        for m, g in df.groupby('model'):
            g_sorted = g.sort_values('F1', ascending=False)
            topn = g_sorted.head(self.config.per_model_top_fields)
            botn = g_sorted.sort_values('F1', ascending=True).head(self.config.per_model_bottom_fields)
            topn = topn.assign(model=m)
            botn = botn.assign(model=m)
            top_rows.append(topn)
            bot_rows.append(botn)
        top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
        bottom_df = pd.concat(bot_rows, ignore_index=True) if bot_rows else pd.DataFrame()
        return top_df, bottom_df
    
    def _analyze_completeness(self, cases: List, case_presence: Dict,
                             per_case_df: pd.DataFrame) -> Dict:
        """Analyze case completeness across different criteria"""
        expected_cases = len(cases)
        complete_cases = []
        per_case_missing_stats = []
        
        # Analyze each case
        for cid, info in case_presence.items():
            present_models = sorted(info['present'])
            missing_models = [m for m in self.config.models if m not in info['present']]
            
            if missing_models:
                per_case_missing_stats.append({
                    'case_id': cid,
                    'year': info['year'],
                    'missing_models': missing_models,
                    'present_models': present_models,
                    'missing_count': len(missing_models),
                    'present_count': len(present_models)
                })
            else:
                complete_cases.append(cid)
        
        # Create subset dataframes
        complete_cases = sorted(complete_cases)
        model_df_complete = self._create_subset_summary(per_case_df, complete_cases)
        
        # Calculate missing summary
        model_missing_summary = {}
        for m in self.config.models:
            miss_cnt = sum(1 for cid, info in case_presence.items() 
                          if m not in info['present'])
            model_missing_summary[m] = {
                'missing_count': miss_cnt,
                'missing_ratio': (miss_cnt / expected_cases) if expected_cases else 0.0,
                'present_count': expected_cases - miss_cnt,
                'present_ratio': ((expected_cases - miss_cnt) / expected_cases) 
                               if expected_cases else 0.0
            }
        
        return {
            'complete_cases': complete_cases,
            'complete_cases_ratio': len(complete_cases) / expected_cases 
                                   if expected_cases else 0.0,
            'model_missing_summary': model_missing_summary,
            'per_case_missing_stats': per_case_missing_stats,
            'model_df_complete': model_df_complete
        }
    
    def _create_subset_summary(self, per_case_df: pd.DataFrame, 
                               case_ids: List[int]) -> pd.DataFrame:
        """Create model summary for a subset of cases"""
        if not case_ids:
            return pd.DataFrame()
        
        subset_df = per_case_df[per_case_df['case_id'].isin(case_ids)]
        if subset_df.empty:
            return pd.DataFrame()
        
        return self._create_model_summary(subset_df)
    
    def _create_extended_analysis(self, extended_field_events: Dict) -> Dict:
        """Create extended field and EV analysis"""
        if not extended_field_events:
            return {
                'field_overall_df': pd.DataFrame(),
                'ev_summary_df': pd.DataFrame(),
                'ev_field_analysis_df': pd.DataFrame()
            }
        
        overall_rows = []
        ev_model_rows = []
        ev_field_rows = []
        ev_model_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'ev': 0, 'total': 0}
        )
        
        for piece, model_map in extended_field_events.items():
            total_tp = total_fp = total_fn = total_ev = 0
            per_model_f1 = {}
            per_model_prec = {}
            per_model_rec = {}
            
            for model, cats in model_map.items():
                tp = cats.get('TP', 0)
                fp = cats.get('FP', 0)
                fn = cats.get('FN', 0)
                evc = cats.get('EV', 0)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_ev += evc
                
                ev_model_totals[model]['ev'] += evc
                ev_model_totals[model]['total'] += (tp + fp + fn + evc)
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                
                per_model_f1[model] = f1
                per_model_prec[model] = prec
                per_model_rec[model] = rec
            
            # Overall metrics for this field
            prec_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            rec_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1_all = (2 * prec_all * rec_all / (prec_all + rec_all)) \
                     if (prec_all + rec_all) > 0 else 0.0
            
            variance_f1 = None
            if len(per_model_f1) >= 2:
                variance_f1 = float(np.var(list(per_model_f1.values())))
            
            base_row = {
                'Piece': piece,
                'TP_total': total_tp,
                'FP_total': total_fp,
                'FN_total': total_fn,
                'EV_total': total_ev,
                'Precision': prec_all,
                'Recall': rec_all,
                'F1': f1_all,
                'Model_F1_Variance': variance_f1
            }
            
            # Add per-model metrics
            for m in self.config.models:
                if m in per_model_f1:
                    base_row[f'{m}_Precision'] = per_model_prec.get(m, 0.0)
                    base_row[f'{m}_Recall'] = per_model_rec.get(m, 0.0)
                    base_row[f'{m}_F1'] = per_model_f1.get(m, 0.0)
                else:
                    base_row[f'{m}_Precision'] = 0.0
                    base_row[f'{m}_Recall'] = 0.0
                    base_row[f'{m}_F1'] = 0.0
            
            overall_rows.append(base_row)
            
            if total_ev > 0:
                ev_field_rows.append({
                    'Piece': piece,
                    'EV_total': total_ev,
                    'Total_events': total_tp + total_fp + total_fn + total_ev,
                    'EV_ratio': (total_ev / (total_tp + total_fp + total_fn + total_ev))
                               if (total_tp + total_fp + total_fn + total_ev) > 0 else 0.0
                })
        
        # Create EV model summary
        for m, stats_m in ev_model_totals.items():
            ev_pct = (stats_m['ev'] / stats_m['total'] * 100) \
                     if stats_m['total'] > 0 else 0.0
            ev_model_rows.append({
                'Model': m,
                'EV_count': stats_m['ev'],
                'Evaluated_total': stats_m['total'],
                'EV_percentage': ev_pct
            })
        
        field_overall_df = pd.DataFrame(overall_rows).sort_values('F1', ascending=False)
        ev_summary_df = pd.DataFrame(ev_model_rows).sort_values('EV_percentage', ascending=False)
        ev_field_analysis_df = pd.DataFrame(ev_field_rows).sort_values('EV_ratio', ascending=False)
        
        return {
            'field_overall_df': field_overall_df,
            'ev_summary_df': ev_summary_df,
            'ev_field_analysis_df': ev_field_analysis_df
        }

    def _compute_field_occurrence_counts(self, case_ids_used: Set[int]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Compute per-Piece instruction and GT occurrence counts from the index CSV.

        - instruction count: cases where field appears in instruction/template fields
        - GT count: cases where field appears in ground-truth fields
        """
        inst_counts: Dict[str, int] = defaultdict(int)
        gt_counts: Dict[str, int] = defaultdict(int)
        try:
            df = pd.read_csv(self.config.index_csv)
        except Exception:
            return dict(inst_counts), dict(gt_counts)

        # Restrict to cases used in this run
        if 'case_id' in df.columns:
            df = df[df['case_id'].isin(case_ids_used)]

        # Candidate columns (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        def _pick(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c.lower() in cols_lower:
                    return cols_lower[c.lower()]
            return None
        col_inst = _pick(['field_group_template', 'template_fields', 'instruction_fields', 'inst_fields'])
        col_gt = _pick(['field_group', 'gt_fields', 'gt_field_group'])
        if not col_inst and not col_gt:
            return dict(inst_counts), dict(gt_counts)

        # Build mapping token->Piece
        cdf = self.catalog.df
        piece_set = set(map(str, cdf['Piece'].dropna().astype(str).tolist()))
        idx_to_piece = {str(r['FieldIndex']): str(r['Piece']) for _, r in cdf.iterrows() if pd.notnull(r.get('FieldIndex'))}
        normidx_to_piece = {str(r['NormalizedIndex']): str(r['Piece']) for _, r in cdf.iterrows() if 'NormalizedIndex' in r}

        def _normalize_index(s: str) -> str:
            return re.sub(r'\[\d+\]', '', s)

        def _parse_list(cell: Any) -> List[str]:
            if pd.isna(cell):
                return []
            if isinstance(cell, (list, tuple)):
                return [str(x).strip() for x in cell]
            text = str(cell).strip()
            if not text:
                return []
            # JSON list
            if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
                try:
                    obj = json.loads(text)
                    if isinstance(obj, list):
                        return [str(x).strip() for x in obj]
                except Exception:
                    pass
            # Split by common delimiters
            parts = re.split(r'[;,|]', text)
            if len(parts) == 1:
                parts = text.split(',')
            return [str(x).strip() for x in parts if str(x).strip()]

        def _to_piece(tok: str) -> Optional[str]:
            if not tok:
                return None
            if tok in piece_set:
                return tok
            n = _normalize_index(tok)
            if n in normidx_to_piece:
                return normidx_to_piece[n]
            if tok in idx_to_piece:
                return idx_to_piece[tok]
            return None

        for _, r in df.iterrows():
            # Instruction
            if col_inst and col_inst in r and pd.notnull(r[col_inst]):
                toks = _parse_list(r[col_inst])
                pieces = set(filter(None, (_to_piece(t) for t in toks)))
                for p in pieces:
                    inst_counts[p] += 1
            # GT
            if col_gt and col_gt in r and pd.notnull(r[col_gt]):
                toks = _parse_list(r[col_gt])
                pieces = set(filter(None, (_to_piece(t) for t in toks)))
                for p in pieces:
                    gt_counts[p] += 1

        return dict(inst_counts), dict(gt_counts)

    def _create_ev_model_summary(self, ev_per_case: Dict[str, List[int]], 
                                  ev_total_evaluated: Dict[str, int]) -> pd.DataFrame:
        """Create per-model EV summary statistics
        
        Returns DataFrame with columns:
        - model: Model name
        - cases: Number of cases evaluated
        - total_ev_fields: Total EV fields across all cases
        - avg_ev_per_case: Average EV fields per case
        - ev_rate: Ratio of EV fields to total evaluated fields
        - cases_with_ev: Number of cases with at least one EV field
        - ev_case_rate: Percentage of cases with EV fields
        """
        rows = []
        for model in self.config.models:
            ev_counts = ev_per_case.get(model, [])
            if not ev_counts:
                continue
            
            total_ev = sum(ev_counts)
            num_cases = len(ev_counts)
            avg_ev = total_ev / num_cases if num_cases > 0 else 0.0
            cases_with_ev = sum(1 for c in ev_counts if c > 0)
            ev_case_rate = (cases_with_ev / num_cases * 100) if num_cases > 0 else 0.0
            
            # Calculate EV rate relative to total evaluated fields
            total_evaluated = ev_total_evaluated.get(model, 0)
            ev_rate = (total_ev / total_evaluated * 100) if total_evaluated > 0 else 0.0
            
            rows.append({
                'model': model,
                'cases': num_cases,
                'total_ev_fields': total_ev,
                'avg_ev_per_case': avg_ev,
                'ev_rate_pct': ev_rate,
                'cases_with_ev': cases_with_ev,
                'ev_case_rate_pct': ev_case_rate,
                'min_ev_per_case': min(ev_counts) if ev_counts else 0,
                'max_ev_per_case': max(ev_counts) if ev_counts else 0,
                'median_ev_per_case': float(np.median(ev_counts)) if ev_counts else 0.0
            })
        
        return pd.DataFrame(rows)

    def _create_ev_field_ranking(self, ev_field_occurrences: Dict[str, Dict[str, int]],
                                  ref_field_cases: Dict[str, Set[int]],
                                  pred_field_cases: Dict[str, Dict[str, Set[int]]],
                                  ev_field_cases: Dict[str, Dict[str, Set[int]]]) -> pd.DataFrame:
        """Create ranking of fields that frequently appear as EV
        
        Returns DataFrame with columns:
        - Piece: Field name
        - FieldIndex: Field index from catalog
        - total_ev_count: Total EV occurrences across all models
        - {model}_ev_count: EV count for each model
        - avg_ev_per_model: Average EV count across models
        - models_with_ev: Number of models where this field appears as EV
        - ref_case_count: Number of cases where field appears in reference
        - {model}_pred_case_count: Number of cases where field appears in model prediction
        - {model}_ev_case_count: Number of cases where field appears ONLY as EV (no TP for this field)
        - is_target: Whether this is a target field
        - field_type: Field type from catalog
        """
        rows = []
        
        # Get all pieces from both EV occurrences and field cases tracking
        all_pieces = set(ev_field_occurrences.keys()) | set(ref_field_cases.keys()) | set(pred_field_cases.keys()) | set(ev_field_cases.keys())
        
        for piece in all_pieces:
            # Get field info from catalog
            match = self.catalog.df[self.catalog.df['Piece'] == piece].head(1)
            if match.empty:
                field_index = piece
                is_target = False
                field_type = 'Unknown'
            else:
                field_index = match['FieldIndex'].iloc[0]
                is_target = match['TargetYN'].iloc[0] == 'Y'
                field_type = match['SourceType'].iloc[0]
            
            model_counts = ev_field_occurrences.get(piece, {})
            total_ev = sum(model_counts.values())
            models_with_ev = len([c for c in model_counts.values() if c > 0])
            avg_ev = total_ev / len(self.config.models) if self.config.models else 0.0
            
            # Get case counts
            ref_cases = len(ref_field_cases.get(piece, set()))
            
            row = {
                'Piece': piece,
                'FieldIndex': field_index,
                'total_ev_count': total_ev,
                'avg_ev_per_model': avg_ev,
                'models_with_ev': models_with_ev,
                'ref_case_count': ref_cases,
                'is_target': is_target,
                'field_type': field_type
            }
            
            # Add per-model EV counts
            for model in self.config.models:
                row[f'{model}_ev_count'] = model_counts.get(model, 0)
            
            # Add per-model prediction case counts
            for model in self.config.models:
                pred_cases = len(pred_field_cases.get(piece, {}).get(model, set()))
                row[f'{model}_pred_case_count'] = pred_cases
            
            # Add per-model EV case counts
            for model in self.config.models:
                ev_cases = len(ev_field_cases.get(piece, {}).get(model, set()))
                row[f'{model}_ev_case_count'] = ev_cases
            
            rows.append(row)
        
        # Sort by total EV count descending
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('total_ev_count', ascending=False)
        
        return df


# ============================================================================
# Output Functions
# ============================================================================

def _round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round all float columns to given decimals"""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    float_cols = [c for c in df2.columns if pd.api.types.is_float_dtype(df2[c])]
    for c in float_cols:
        df2[c] = df2[c].round(decimals)
    return df2


def save_outputs(agg: Dict[str, Any], config: Config, variant: str, section: str, ts: str) -> Tuple[Path, str]:
    """Save all output files for a variant and section
    
    Args:
        agg: Aggregated results dictionary
        config: Configuration object
        variant: Metrics variant name
        section: 'protocol', 'results', or 'both'
        ts: Timestamp string
    """
    # Create output directory with section suffix
    if section == 'both':
        section_suffix = ''
    else:
        section_suffix = f'_{section}'
    
    if variant != 'metrics':
        out_dir = config.output_base / f'stats_{ts}_{variant}{section_suffix}'
    else:
        out_dir = config.output_base / f'stats_{ts}{section_suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Round numeric columns
    for key in ['per_case_df', 'model_df', 'model_df_complete', 
                'field_df', 'field_overall_df', 'per_model_field_metrics_df',
                'per_model_field_top_df', 'per_model_field_bottom_df',
                'ev_model_summary_df', 'ev_field_ranking_df']:
        if key in agg and isinstance(agg[key], pd.DataFrame):
            agg[key] = _round_df(agg[key])
    
    # Save CSV files
    agg['per_case_df'].to_csv(out_dir / 'per_case_metrics.csv', index=False)
    agg['model_df'].to_csv(out_dir / f'summary_model_metrics.csv', index=False)
    
    # Save complete cases summaries with variant suffix
    if isinstance(agg.get('model_df_complete'), pd.DataFrame) and not agg['model_df_complete'].empty:
        agg['model_df_complete'].to_csv(
            out_dir / f'summary_model_metrics_complete_cases.csv', index=False
        )
    
    agg['field_df'].to_csv(out_dir / 'summary_field_value_counts.csv', index=False)
    
    # Save per-model relevant completeness and field rankings
    pm_field_df = agg.get('per_model_field_metrics_df')
    if isinstance(pm_field_df, pd.DataFrame) and not pm_field_df.empty:
        pm_field_df.to_csv(out_dir / 'per_model_field_metrics.csv', index=False)
    pm_top_df = agg.get('per_model_field_top_df')
    if isinstance(pm_top_df, pd.DataFrame) and not pm_top_df.empty:
        pm_top_df.to_csv(out_dir / f'per_model_field_top_{config.per_model_top_fields}.csv', index=False)
    pm_bottom_df = agg.get('per_model_field_bottom_df')
    if isinstance(pm_bottom_df, pd.DataFrame) and not pm_bottom_df.empty:
        pm_bottom_df.to_csv(out_dir / f'per_model_field_bottom_{config.per_model_bottom_fields}.csv', index=False)
    
    # Save EV analysis outputs
    ev_model_summary_df = agg.get('ev_model_summary_df')
    if isinstance(ev_model_summary_df, pd.DataFrame) and not ev_model_summary_df.empty:
        ev_model_summary_df.to_csv(out_dir / 'ev_model_summary.csv', index=False)
    
    ev_field_ranking_df = agg.get('ev_field_ranking_df')
    if isinstance(ev_field_ranking_df, pd.DataFrame) and not ev_field_ranking_df.empty:
        ev_field_ranking_df.to_csv(out_dir / 'ev_field_ranking.csv', index=False)
    
    # Save extended analysis
    if agg.get('extended_enabled'):
        fodf = agg.get('field_overall_df')
        if isinstance(fodf, pd.DataFrame) and not fodf.empty:
            # Enrich field_overall_metrics with FieldIndex, reference case count, and per-model EV case counts
            base = fodf.copy()
            # FieldIndex from field_df
            fdf = agg.get('field_df')
            if isinstance(fdf, pd.DataFrame) and not fdf.empty and 'FieldIndex' in fdf.columns:
                base = base.merge(fdf[['Piece','FieldIndex']].drop_duplicates(), on='Piece', how='left')
            # ref_case_count and {model}_ev_case_count from ev_field_ranking_df
            evrank = agg.get('ev_field_ranking_df')
            if isinstance(evrank, pd.DataFrame) and not evrank.empty:
                cols = ['Piece','ref_case_count']
                for m in config.models:
                    col_ev_case = f'{m}_ev_case_count'
                    if col_ev_case in evrank.columns:
                        cols.append(col_ev_case)
                base = base.merge(evrank[cols].drop_duplicates(), on='Piece', how='left')
            # Reorder to place FieldIndex after Piece
            if 'FieldIndex' in base.columns:
                cols = list(base.columns)
                cols.remove('FieldIndex')
                cols.remove('Piece')
                base = base[['Piece','FieldIndex'] + cols]
            base.to_csv(out_dir / 'field_overall_metrics.csv', index=False)
    
    # Save metadata
    meta = {
        'generated_at': ts,
        'variant': variant,
        'section': section,
        'expected_case_count': agg['case_count_expected'],
        'missing_cases': agg['missing'],
        'model_missing_summary': agg.get('model_missing_summary', {}),
        'complete_cases': {
            'count': len(agg.get('complete_cases', [])),
            'ratio': agg.get('complete_cases_ratio', 0.0),
            'case_ids': agg.get('complete_cases', [])
        },
        'cases_with_any_missing_stats': agg.get('per_case_missing_stats', []),
        'extended': agg.get('extended_enabled', False)
    }
    
    (out_dir / 'aggregate.json').write_text(json.dumps(meta, indent=2))
    
    print(f'[OK] Wrote outputs to {out_dir}')
    return out_dir, ts


# ============================================================================
# Plotting Functions
# ============================================================================

class PlotGenerator:
    def __init__(self, config: Config):
        self.config = config
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.enabled = True
        except ImportError:
            self.enabled = False
            print('[WARN] Plot libs unavailable; skipping plots.')
        # Lazy init for piece enum map
        self._piece_enum_map: Optional[Dict[str, bool]] = None
    
    def generate_plots(self, agg: Dict[str, Any], out_dir: Path, variant: str):
        """Generate all plots for a variant"""
        if not self.enabled:
            return
        
        # Create variant-specific plot directory
        if variant != 'metrics':
            plot_dir = out_dir / 'plots' / variant
        else:
            plot_dir = out_dir / 'plots'
        
        if self.config.plot_output_dir:
            # Use custom plot directory with variant subdirectory
            if variant != 'metrics':
                plot_dir = self.config.plot_output_dir / variant
            else:
                plot_dir = self.config.plot_output_dir
        
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        self.sns.set(style=self.config.style)
        
        # Generate various plots
        self._plot_model_metrics_grouped(agg, plot_dir)
        
        if self.config.extended and agg.get('extended_enabled'):
            if not self.config.no_heatmap:
                self._plot_field_heatmap(agg, plot_dir)
                # Second version: DataDefinition order + six-metric annotations per cell
                self._plot_field_heatmap_ddorder_allmetrics(agg, plot_dir)
        
        if self.config.adv_plots:
            self._plot_advanced(agg, plot_dir)
        
        # Per-model field rankings (top/bottom by F1)
        self._plot_per_model_field_rankings(agg, plot_dir)

        print(f'[OK] Plots saved to {plot_dir}')
    
    def _plot_model_metrics_grouped(self, agg: Dict[str, Any], plot_dir: Path):
        """Plot all six metrics grouped by metric (hue=model)"""
        for df_key, title, filename in [
            ('model_df', 'Model Metrics (All Cases)', 'model_metrics_all_cases_grouped.png'),
            ('model_df_complete', 'Model Metrics (Complete Cases)', 'model_metrics_complete_cases_grouped.png'),
        ]:
            df = agg.get(df_key)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            
            # Ensure model order from config
            df = df.copy()
            df['model'] = pd.Categorical(df['model'], categories=self.config.models, ordered=True)
            df = df.sort_values('model')
            
            # Melt for grouped bar chart
            metric_cols = ['key_precision_avg', 'key_recall_avg', 'key_f1_avg', 
                           'value_precision_avg', 'value_recall_avg', 'value_f1_avg']
            melted = df.melt(id_vars=['model'], value_vars=metric_cols, 
                             var_name='metric', value_name='score')
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=(14, 6))
            
            # Map model names to colors
            model_list = df['model'].unique().tolist()
            palette = [MODEL_COLORS.get(m, FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)]) 
                      for i, m in enumerate(model_list)]
            
            # Plot grouped bars
            self.sns.barplot(data=melted, x='metric', y='score', hue='model', 
                           palette=palette, ax=ax, alpha=PLOT_STYLE['alpha'],
                           edgecolor=PLOT_STYLE['edgecolor'], linewidth=PLOT_STYLE['linewidth'],
                           hue_order=self.config.models)
            
            # Formatting
            ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'], weight='bold')
            ax.set_xlabel('Metric', fontsize=PLOT_STYLE['label_fontsize'])
            ax.set_ylabel('Score', fontsize=PLOT_STYLE['label_fontsize'])
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=PLOT_STYLE['grid_alpha'])
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Annotate bars with values
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=PLOT_STYLE['value_fontsize'])
            
            self.plt.tight_layout()
            self.plt.savefig(plot_dir / filename, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
            self.plt.close()
            mdf = agg.get(df_key)
            if not isinstance(mdf, pd.DataFrame) or mdf.empty:
                continue
            
            metric_cols = ['key_precision_avg', 'key_recall_avg', 'key_f1_avg',
                          'value_precision_avg', 'value_recall_avg', 'value_f1_avg']
            if not all(c in mdf.columns for c in metric_cols):
                continue
            
            # Prepare data
            metric_map = {
                'key_precision_avg': 'Key Precision',
                'key_recall_avg': 'Key Recall',
                'key_f1_avg': 'Key F1',
                'value_precision_avg': 'Value Precision',
                'value_recall_avg': 'Value Recall',
                'value_f1_avg': 'Value F1'
            }
            # Build augmented labels for relevant_strict_* subsets with counts:
            # exp = expected total cases, rel_m = model-target cases (module F1/Recall==1), common = cases in this subset
            per_model_labels: Dict[str, str] = {}
            if df_key in ('model_df_relevant_strict_f1_complete', 'model_df_relevant_strict_recall_complete'):
                expected_cases = int(agg.get('case_count_expected', 0))
                pm_rel = agg.get('per_model_relevant_completion_df')
                f1_map: Dict[str, int] = {}
                rec_map: Dict[str, int] = {}
                if isinstance(pm_rel, pd.DataFrame) and not pm_rel.empty:
                    for _, rr in pm_rel.iterrows():
                        mname = str(rr['model'])
                        f1_map[mname] = int(rr.get('rel_module_f1_complete_count', 0))
                        rec_map[mname] = int(rr.get('rel_module_recall_complete_count', 0))
                for _, rlab in mdf.iterrows():
                    mname2 = str(rlab['model'])
                    common = int(rlab['cases'])
                    model_target = f1_map.get(mname2, 0) if df_key == 'model_df_relevant_strict_f1_complete' else rec_map.get(mname2, 0)
                    per_model_labels[mname2] = f"{mname2}\n(exp={expected_cases}, rel_m={model_target}, common={common})"

            melt_rows = []
            for _, r in mdf.iterrows():
                label_model = per_model_labels.get(r['model'], f"{r['model']}\n(n={int(r['cases'])})")
                model_raw = str(r['model'])
                for col, label in metric_map.items():
                    melt_rows.append({
                        'model': label_model,
                        'model_raw': model_raw,
                        'metric': label,
                        'value': r[col]
                    })
            mm_df = pd.DataFrame(melt_rows)
            
            # Create dynamic palette based on model_display
            palette = {}
            fallback_iter = iter(FALLBACK_PALETTE)
            unique_pairs = mm_df[['model_raw', 'model']].drop_duplicates()
            
            for _, row in unique_pairs.iterrows():
                raw = row['model_raw']
                disp = row['model']
                color = MODEL_COLORS.get(raw)
                if not color:
                    for k, v in MODEL_COLORS.items():
                        if k in raw:
                            color = v
                            break
                if not color:
                    color = next(fallback_iter, '#333333')
                palette[disp] = color

            # Create plot
            self.plt.figure(figsize=(12, 6))
            ax = self.sns.barplot(data=mm_df, x='metric', y='value', hue='model', palette=palette,
                                  alpha=PLOT_STYLE['alpha'], edgecolor=PLOT_STYLE['edgecolor'], linewidth=PLOT_STYLE['linewidth'])
            
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.text(p.get_x() + p.get_width()/2, h + 0.01, f'{h:.2f}',
                           ha='center', va='bottom', fontsize=PLOT_STYLE['value_fontsize'], rotation=0)
            
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Score (0-1)', fontsize=PLOT_STYLE['label_fontsize'], fontweight='bold')
            ax.set_xlabel('Metric', fontsize=PLOT_STYLE['label_fontsize'], fontweight='bold')
            ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'], fontweight='bold')
            ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.grid(axis='y', alpha=PLOT_STYLE['grid_alpha'])
            
            self.plt.tight_layout()
            self.plt.savefig(plot_dir / filename, dpi=PLOT_STYLE['dpi'])
            self.plt.close()
    
    def _plot_field_heatmap(self, agg: Dict[str, Any], plot_dir: Path):
        """Plot field-model F1 heatmap"""
        fodf = agg.get('field_overall_df')
        if not isinstance(fodf, pd.DataFrame) or fodf.empty:
            return
        
        base_field = agg['field_df']
        heat_records = []
        
        # Build piece->is_enum map by scanning field catalog once
        def _get_piece_enum_map() -> Dict[str, bool]:
            if self._piece_enum_map is not None:
                return self._piece_enum_map
            try:
                cdf = pd.read_csv(self.config.field_catalog_path)
                # Heuristic: treat SourceType containing 'enum'/'category' (case-insensitive) as enum
                def is_enum_row(v: Any) -> bool:
                    try:
                        s = str(v).lower()
                    except Exception:
                        return False
                    return ('enum' in s) or ('category' in s) or ('categorical' in s)
                enum_map = {}
                for _, r in cdf.iterrows():
                    piece = r.get('Piece')
                    stype = r.get('SourceType')
                    if piece is None:
                        continue
                    enum_map[str(piece)] = is_enum_row(stype)
                self._piece_enum_map = enum_map
                return enum_map
            except Exception:
                self._piece_enum_map = {}
                return self._piece_enum_map
        piece_enum_map = _get_piece_enum_map()
        
        # Compute per-row ground-truth occurrence support for labels.
        # Use max over models of (TP + FN) to approximate GT events and avoid double counting across models.
        support_totals: Dict[str, int] = {}
        for _, row in base_field.iterrows():
            piece = row['Piece']
            sup_candidates = []
            for m in self.config.models:
                tp = int(row.get(f'{m}_TP', 0) or 0)
                fn = int(row.get(f'{m}_FN', 0) or 0)
                sup_candidates.append(tp + fn)
            support_totals[piece] = int(max(sup_candidates)) if sup_candidates else 0

        for _, row in base_field.iterrows():
            piece = row['Piece']
            rec_row = {'Piece': piece}
            for m in self.config.models:
                tp = row.get(f'{m}_TP', 0)
                fp = row.get(f'{m}_FP', 0)
                fn = row.get(f'{m}_FN', 0)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec_m = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec_m / (prec + rec_m)) if (prec + rec_m) > 0 else 0.0
                rec_row[m] = f1
            heat_records.append(rec_row)
        
        heat_df = pd.DataFrame(heat_records)
        # Maintain config.models order for columns
        model_cols = [m for m in self.config.models if m in heat_df.columns]
        heat_df['avg_f1'] = heat_df[model_cols].mean(axis=1)
        
        # Build enhanced row labels using instruction/GT counts
        inst_counts: Dict[str, int] = agg.get('field_inst_counts', {}) or {}
        gt_counts: Dict[str, int] = agg.get('field_gt_counts', {}) or {}
        heat_df['__inst__'] = heat_df['Piece'].map(lambda p: int(inst_counts.get(p, 0)))
        heat_df['__gt__'] = heat_df['Piece'].map(lambda p: int(gt_counts.get(p, 0)))
        heat_df['__enum__'] = heat_df['Piece'].map(lambda p: piece_enum_map.get(p, False))
        def _mk_label(p: str, inst_v: int, gt_v: int, is_enum: bool) -> str:
            return p
        heat_df['PieceLabel'] = [
            _mk_label(p, int(inst if pd.notnull(inst) else 0), int(gt if pd.notnull(gt) else 0), bool(en))
            for p, inst, gt, en in zip(heat_df['Piece'], heat_df['__inst__'], heat_df['__gt__'], heat_df['__enum__'])
        ]
        
        # Filter out rows where all models have F1 = 0
        heat_df['max_f1'] = heat_df[model_cols].max(axis=1)
        heat_df = heat_df[heat_df['max_f1'] > 0]
        
        topn = heat_df.sort_values('avg_f1', ascending=False).head(self.config.plot_top_fields)
        plot_matrix = topn.set_index('PieceLabel')[model_cols]

        # Keep model names as-is without counts
        # No column renaming needed

        # Narrower width (2/3 of previous scaling)
        width = min(12, max(6, 0.47 * len(model_cols) + 3))
        height = max(6, 0.26 * plot_matrix.shape[0] + 4)
        self.plt.figure(figsize=(width, height))
        # Use seaborn's default colormap and annotate exact values
        ax = self.sns.heatmap(plot_matrix, annot=True, fmt='.2f', cmap=CUSTOM_HEATMAP_CMAP,
                        vmin=0, vmax=1, cbar_kws={'label': 'F1'}, annot_kws={'linespacing': 1.1})
        # Improve readability of x tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Bold the maximum value(s) per row
        try:
            texts = ax.texts
            nrows, ncols = plot_matrix.shape
            for i in range(nrows):
                row_vals = plot_matrix.iloc[i, :].astype(float).values
                if row_vals.size == 0:
                    continue
                # Use rounded values for tie detection to match displayed annotations (fmt='.2f')
                row_vals_r = np.round(row_vals, 2)
                maxr = np.nanmax(row_vals_r)
                for j in range(ncols):
                    t = texts[i * ncols + j]
                    if np.isfinite(row_vals_r[j]) and row_vals_r[j] == maxr:
                        t.set_fontweight('bold')
            # Draw light border for all cells
            for i in range(nrows):
                for j in range(ncols):
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="#FFFFFF", linewidth=1.1))
        except Exception:
            pass
        # Include overall target file (case) count in the title label
        n_cases = int(agg.get('case_count_expected', 0))
        self.plt.title(f"{self.config.title_field_heatmap} (cases={n_cases})")
        self.plt.xlabel('Model')
        self.plt.ylabel('Field (Piece)')
        # Add a bit more top margin to prevent title clipping
        self.plt.tight_layout(rect=[0, 0, 1, 0.98])
        self.plt.savefig(plot_dir / 'field_model_f1_heatmap.png', dpi=self.config.dpi, bbox_inches='tight')
        self.plt.close()

    def _plot_field_heatmap_ddorder_allmetrics(self, agg: Dict[str, Any], plot_dir: Path):
        """Second heatmap variant:
        - Row order follows DataDefinition (catalog) order of fields (Piece)
        - Color encodes value F1 (per field, per model)
        - Each cell annotation includes six metrics computed per (field×model):
          (kP, kR, kF, vP, vR, vF) where
            k*: computed from per-field key counts (kTP/kFP/kFN)
            v*: computed from per-field value counts (TP/FP/FN)
        """
        # Base dataframes
        base_field = agg.get('field_df')
        mdf = agg.get('model_df')
        if not isinstance(base_field, pd.DataFrame) or base_field.empty:
            return
        if not isinstance(mdf, pd.DataFrame) or mdf.empty:
            return

        # Compute per-field×model key metrics from kTP/kFP/kFN in field_df
        # field_df should include columns like '{model}_kTP', '{model}_kFP', '{model}_kFN'
        def _safe(val: Any) -> float:
            try:
                return float(val or 0)
            except Exception:
                return 0.0

        # Compute per-field per-model value metrics from TP/FP/FN
        pieces: List[str] = list(base_field['Piece'].astype(str)) if 'Piece' in base_field.columns else []
        if not pieces:
            return
        models: List[str] = list(self.config.models)
        # Build matrices
        val_prec: Dict[Tuple[str, str], float] = {}
        val_rec: Dict[Tuple[str, str], float] = {}
        val_f1: Dict[Tuple[str, str], float] = {}
        key_prec: Dict[Tuple[str, str], float] = {}
        key_rec: Dict[Tuple[str, str], float] = {}
        key_f1: Dict[Tuple[str, str], float] = {}
        for _, row in base_field.iterrows():
            piece = str(row['Piece'])
            for m in models:
                tp = float(row.get(f'{m}_TP', 0) or 0)
                fp = float(row.get(f'{m}_FP', 0) or 0)
                fn = float(row.get(f'{m}_FN', 0) or 0)
                p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                val_prec[(piece, m)] = p
                val_rec[(piece, m)] = r
                val_f1[(piece, m)] = f

                ktp = _safe(row.get(f'{m}_kTP'))
                kfp = _safe(row.get(f'{m}_kFP'))
                kfn = _safe(row.get(f'{m}_kFN'))
                kp = (ktp / (ktp + kfp)) if (ktp + kfp) > 0 else 0.0
                kr = (ktp / (ktp + kfn)) if (ktp + kfn) > 0 else 0.0
                kf = (2 * kp * kr / (kp + kr)) if (kp + kr) > 0 else 0.0
                key_prec[(piece, m)] = kp
                key_rec[(piece, m)] = kr
                key_f1[(piece, m)] = kf

        # Determine DataDefinition order for Pieces
        try:
            cdf = pd.read_csv(self.config.field_catalog_path)
            order_list: List[str] = []
            if 'Piece' in cdf.columns:
                for x in cdf['Piece']:
                    if pd.isnull(x):
                        continue
                    sx = str(x)
                    if sx not in order_list:
                        order_list.append(sx)
            else:
                # Fallback to FieldIndex -> Piece mapping if available in agg
                order_list = list(dict.fromkeys(pieces))
        except Exception:
            order_list = list(dict.fromkeys(pieces))

        # Filter to pieces present and preserve order
        pieces_ordered = [p for p in order_list if p in set(pieces)]
        # Apply top limit if configured; <=0 means show all
        topN = getattr(self.config, 'plot_top_fields', 40)
        if isinstance(topN, int) and topN > 0:
            pieces_ordered = pieces_ordered[:topN]

        if not pieces_ordered:
            return

        # Build plot matrix for color (value F1) and annotation strings
        # Row labels: include inst/gt counts and enum marker
        # Prepare piece -> enum map
        def _get_piece_enum_map() -> Dict[str, bool]:
            if self._piece_enum_map is not None:
                return self._piece_enum_map
            try:
                cdf2 = pd.read_csv(self.config.field_catalog_path)
                def is_enum_row(v: Any) -> bool:
                    try:
                        s = str(v).lower()
                    except Exception:
                        return False
                    return ('enum' in s) or ('category' in s) or ('categorical' in s)
                enum_map2 = {}
                for _, r in cdf2.iterrows():
                    piece = r.get('Piece')
                    stype = r.get('SourceType')
                    if piece is None:
                        continue
                    enum_map2[str(piece)] = is_enum_row(stype)
                self._piece_enum_map = enum_map2
                return enum_map2
            except Exception:
                self._piece_enum_map = {}
                return self._piece_enum_map
        piece_enum_map = _get_piece_enum_map()

        inst_counts: Dict[str, int] = agg.get('field_inst_counts', {}) or {}
        gt_counts: Dict[str, int] = agg.get('field_gt_counts', {}) or {}
        def _mk_label(piece: str) -> str:
            inst_v = int(inst_counts.get(piece, 0))
            gt_v = int(gt_counts.get(piece, 0))
            suffix = ', E' if piece_enum_map.get(piece, False) else ''
            return f"{piece} (inst={inst_v}, gt={gt_v}{suffix})"

        color_data = []
        annot_data = []
        row_labels: List[str] = []
        for piece in pieces_ordered:
            row_color = []
            row_annot = []
            for m in models:
                vf = float(val_f1.get((piece, m), 0.0))
                vp = float(val_prec.get((piece, m), 0.0))
                vr = float(val_rec.get((piece, m), 0.0))
                kp = float(key_prec.get((piece, m), 0.0))
                kr = float(key_rec.get((piece, m), 0.0))
                kf = float(key_f1.get((piece, m), 0.0))
                row_color.append(vf)
                row_annot.append(
                    f"kP {kp:.2f} kR {kr:.2f} kF {kf:.2f}\n"
                    f"vP {vp:.2f} vR {vr:.2f} vF {vf:.2f}"
                )
            color_data.append(row_color)
            annot_data.append(row_annot)
            row_labels.append(_mk_label(piece))

        color_df = pd.DataFrame(color_data, index=row_labels, columns=models)

        # Augment x labels with counts (cases)
        counts_per_model: Dict[str, int] = {}
        if {'model','cases'} <= set(mdf.columns):
            counts_per_model = {str(r['model']): int(r['cases']) for _, r in mdf.iterrows()}
        def _alias_label(name: str) -> str:
            n = counts_per_model.get(name, 0)
            alias = name if len(name) <= 18 else name.replace('_', '\n')
            return f"{alias}\n(n={n})"
        color_df = color_df.rename(columns={c: _alias_label(c) for c in color_df.columns})

        # Figure size; 2/3 width ratio
        width = min(21, max(9, 0.9 * len(models) + 5))
        height = max(9, 0.52 * len(pieces_ordered) + 5)
        self.plt.figure(figsize=(width, height))
        ax = self.sns.heatmap(
            color_df,
            annot=np.array(annot_data),
            fmt='',
            cmap=CUSTOM_HEATMAP_CMAP,
            vmin=0, vmax=1,
            cbar_kws={'label': 'Value F1'},
            annot_kws={'fontsize': 12, 'ha': 'center', 'va': 'center'}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Slight line spacing for readability, but keep horizontal compactness
        try:
            for t in ax.texts:
                t.set_linespacing(1.1)
        except Exception:
            pass
        # Add base border for ALL cells and bold maxima (ties included)
        try:
            from matplotlib.patches import Rectangle
            texts = ax.texts
            nrows, ncols = color_df.shape
            # Draw light border for all cells
            for i in range(nrows):
                for j in range(ncols):
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="#FFFFFF", linewidth=0.9))
            # Bold maxima per row (ties included)
            for i in range(nrows):
                row_vals = color_df.iloc[i, :].astype(float).values
                if row_vals.size == 0:
                    continue
                row_vals_r = np.round(row_vals, 2)
                maxr = np.nanmax(row_vals_r)
                for j in range(ncols):
                    if np.isfinite(row_vals_r[j]) and row_vals_r[j] == maxr:
                        t = texts[i * ncols + j]
                        t.set_fontweight('bold')
        except Exception:
            pass
        self.plt.title('Field Metrics Heatmap (DD order)\nAnnot: kP kR kF / vP vR vF')
        self.plt.xlabel('Model')
        self.plt.ylabel('Field (Piece) — DataDefinition order')
        self.plt.tight_layout(rect=[0, 0, 1, 0.98])
        self.plt.savefig(plot_dir / 'field_model_metrics_heatmap_ddorder.png', dpi=self.config.dpi, bbox_inches='tight')
        self.plt.close()
    
    def _plot_advanced(self, agg: Dict[str, Any], plot_dir: Path):
        """Generate advanced analytical plots"""
        
        # Field analysis plots
        fodf = agg.get('field_overall_df')
        if self.config.extended and isinstance(fodf, pd.DataFrame) and not fodf.empty:
            self._plot_top_bottom_fields(fodf, plot_dir)
        
        # Missing metrics stacked bar
        self._plot_missing_metrics(agg, plot_dir)
        
    def _plot_top_bottom_fields(self, fodf: pd.DataFrame, plot_dir: Path):
        """Plot top and bottom performing fields"""
        # Top fields
        topn = fodf.sort_values('F1', ascending=False).head(self.config.top_fields_bar)
        self.plt.figure(figsize=(9, max(4, 0.25 * len(topn))))
        self.plt.barh(topn['Piece'][::-1], topn['F1'][::-1], color='teal')
        self.plt.xlabel('F1')
        self.plt.title(f'Top {len(topn)} Fields by F1')
        
        for i, (f1v, name) in enumerate(zip(topn['F1'][::-1], topn['Piece'][::-1])):
            self.plt.text(f1v + 0.005, i, f'{f1v:.3f}', va='center', fontsize=8)
        
        self.plt.xlim(0, 1)
        self.plt.tight_layout()
        self.plt.savefig(plot_dir / 'top_fields_f1_barh.png', dpi=self.config.dpi)
        self.plt.close()
        
        # Bottom fields
        botn = fodf.sort_values('F1', ascending=True).head(self.config.bottom_fields_bar)
        self.plt.figure(figsize=(9, max(4, 0.25 * len(botn))))
        self.plt.barh(botn['Piece'][::-1], botn['F1'][::-1], color='maroon')
        self.plt.xlabel('F1')
        self.plt.title(f'Bottom {len(botn)} Fields by F1')
        
        for i, (f1v, name) in enumerate(zip(botn['F1'][::-1], botn['Piece'][::-1])):
            self.plt.text(f1v + 0.005, i, f'{f1v:.3f}', va='center', fontsize=8)
        
        self.plt.xlim(0, 1)
        self.plt.tight_layout()
        self.plt.savefig(plot_dir / 'bottom_fields_f1_barh.png', dpi=self.config.dpi)
        self.plt.close()
    
    def _plot_missing_metrics(self, agg: Dict[str, Any], plot_dir: Path):
        """Plot missing metrics stacked bar"""
        mms = agg.get('model_missing_summary')
        if not mms:
            return
        
        miss_rows = []
        # Maintain config.models order
        for model in self.config.models:
            if model in mms:
                stats_m = mms[model]
                miss_rows.append({
                    'model': model,
                    'present': stats_m['present_count'],
                    'missing': stats_m['missing_count']
                })
        
        miss_df = pd.DataFrame(miss_rows)
        if miss_df['missing'].sum() == 0:
            return
        
        self.plt.figure(figsize=(7, 4))
        bottom = None
        for col, color in [('present', '#2ca02c'), ('missing', '#d62728')]:
            self.plt.bar(miss_df['model'], miss_df[col], bottom=bottom,
                        label=col.capitalize(), color=color)
            bottom = miss_df[col] if bottom is None else bottom + miss_df[col]
        
        for i, r in miss_df.iterrows():
            total = r['present'] + r['missing']
            self.plt.text(i, r['present']/2, f"{r['present']}",
                         ha='center', va='center', color='white', fontsize=8)
            if r['missing'] > 0:
                self.plt.text(i, r['present'] + r['missing']/2, f"{r['missing']}",
                             ha='center', va='center', color='white', fontsize=8)
            self.plt.text(i, total + 0.5, f'n={total}', ha='center', fontsize=8)
        
        self.plt.ylabel('Cases')
        self.plt.title('Present vs Missing Metrics Files by Model')
        self.plt.legend()
        self.plt.tight_layout()
        self.plt.savefig(plot_dir / 'missing_metrics_stacked_bar.png', dpi=self.config.dpi)
        self.plt.close()

    def _plot_per_model_field_rankings(self, agg: Dict[str, Any], plot_dir: Path):
        """Plot top and bottom fields by F1 per model using computed rankings."""
        top_df = agg.get('per_model_field_top_df')
        bot_df = agg.get('per_model_field_bottom_df')
        if not (isinstance(top_df, pd.DataFrame) and not top_df.empty) and not (isinstance(bot_df, pd.DataFrame) and not bot_df.empty):
            return

        out_dir = plot_dir / 'per_model_fields'
        out_dir.mkdir(parents=True, exist_ok=True)

        def _plot_barh(df: pd.DataFrame, model: str, kind: str, n: int):
            dd = df[df['model'] == model]
            if dd.empty:
                return
            dd = dd.copy()
            # Order for barh
            if kind == 'top':
                dd = dd.sort_values('F1', ascending=True).tail(n)
            else:
                dd = dd.sort_values('F1', ascending=True).head(n)
            self.plt.figure(figsize=(10, max(4, 0.35 * len(dd))))
            color = '#2a9d8f' if kind == 'top' else '#d62828'
            self.plt.barh(dd['Piece'], dd['F1'], color=color)
            for i, (f1v, sup) in enumerate(zip(dd['F1'], dd['Support'])):
                try:
                    f1f = float(f1v)
                except Exception:
                    f1f = 0.0
                self.plt.text(f1f + 0.005, i, f"{f1f:.3f} (sup={int(sup)})", va='center', fontsize=8)
            self.plt.xlabel('F1')
            self.plt.xlim(0, 1)
            title = 'Top' if kind == 'top' else 'Bottom'
            self.plt.title(f"{model}: {title} fields by F1 (n={len(dd)})")
            self.plt.tight_layout()
            self.plt.savefig(out_dir / f"{model}_{kind}_fields_f1_barh.png", dpi=self.config.dpi)
            self.plt.close()

        # Use config.models order
        models_present = set()
        if isinstance(top_df, pd.DataFrame) and not top_df.empty:
            models_present.update(top_df['model'].unique())
        if isinstance(bot_df, pd.DataFrame) and not bot_df.empty:
            models_present.update(bot_df['model'].unique())
        models = [m for m in self.config.models if m in models_present]

        for m in models:
            if isinstance(top_df, pd.DataFrame) and not top_df.empty:
                _plot_barh(top_df, m, 'top', self.config.per_model_top_fields)
            if isinstance(bot_df, pd.DataFrame) and not bot_df.empty:
                _plot_barh(bot_df, m, 'bottom', self.config.per_model_bottom_fields)

# ============================================================================
# Main Functions
# ============================================================================

def print_summary(agg: Dict[str, Any]):
    """Print summary statistics to console"""
    if not agg['model_df'].empty:
        print('\nModel summary (all available metrics):')
        disp_df = agg['model_df'].copy()
        disp_df['model'] = disp_df.apply(lambda r: f"{r['model']} (n={int(r['cases'])})", axis=1)
        print(disp_df.drop(columns=['cases']).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    if isinstance(agg.get('model_df_complete'), pd.DataFrame) and not agg['model_df_complete'].empty:
        print('\nModel summary (complete cases only):')
        disp_df_c = agg['model_df_complete'].copy()
        disp_df_c['model'] = disp_df_c.apply(lambda r: f"{r['model']} (n={int(r['cases'])})", axis=1)
        print(disp_df_c.drop(columns=['cases']).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"Complete cases: {len(agg.get('complete_cases', []))} / {agg['case_count_expected']} "
              f"(ratio={agg.get('complete_cases_ratio', 0):.3f})")
    
    # Print missing statistics
    missing_any = {k: v for k, v in agg['missing'].items() if v}
    if missing_any:
        print('\nMissing metrics files (first 10 ids each):')
        for m, ids in missing_any.items():
            print(f'  {m}: {len(ids)} missing -> {sorted(ids)[:10]}{"..." if len(ids) > 10 else ""}')
    

    # Show a quick peek of per-model strongest/weakest fields
    pm_top = agg.get('per_model_field_top_df')
    pm_bot = agg.get('per_model_field_bottom_df')
    if isinstance(pm_top, pd.DataFrame) and not pm_top.empty:
        print(f"\nTop fields per model (top {pm_top.groupby('model').size().max() if not pm_top.empty else 0} by F1, min support={agg.get('per_model_field_metrics_df', pd.DataFrame()).get('Support').min() if isinstance(agg.get('per_model_field_metrics_df'), pd.DataFrame) and not agg.get('per_model_field_metrics_df').empty else 'NA'}):")
        for m, g in pm_top.groupby('model'):
            print(f"  [{m}] top {len(g)} fields: ")
            print(g[['Piece','Support','F1']].head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if isinstance(pm_bot, pd.DataFrame) and not pm_bot.empty:
        print(f"\nBottom fields per model (min support filter applied):")
        for m, g in pm_bot.groupby('model'):
            print(f"  [{m}] bottom {len(g)} fields: ")
            print(g[['Piece','Support','F1']].head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Show EV analysis summary
    ev_model_summary = agg.get('ev_model_summary_df')
    if isinstance(ev_model_summary, pd.DataFrame) and not ev_model_summary.empty:
        print('\nEV Field Summary by Model:')
        print(ev_model_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    ev_field_ranking = agg.get('ev_field_ranking_df')
    if isinstance(ev_field_ranking, pd.DataFrame) and not ev_field_ranking.empty:
        print(f"\nTop 10 Most Frequent EV Fields:")
        display_cols = ['Piece', 'total_ev_count', 'models_with_ev', 'is_target', 'field_type']
        print(ev_field_ranking[display_cols].head(10).to_string(index=False))




def main():
    parser = argparse.ArgumentParser(description='Generate evaluation statistics from metrics')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f'[ERROR] Configuration file not found: {config_path}')
        return 1
    
    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        print(f'[ERROR] Failed to load configuration: {e}')
        return 1
    
    # Validate required paths
    if not config.index_csv.exists():
        print(f'[ERROR] Index CSV not found: {config.index_csv}')
        return 1
    
    if not config.field_catalog_path.exists():
        print(f'[ERROR] Field catalog not found: {config.field_catalog_path}')
        return 1
    
    print(f'Loading configuration from: {config_path}')
    print(f'Processing variants: {config.variants}')
    print(f'Evaluating sections: {config.evaluate_sections}')
    
    # Initialize components
    catalog = FieldCatalog(config.field_catalog_path, target_fields=config.target_fields)
    processor = MetricsProcessor(config, catalog)
    plotter = PlotGenerator(config)
    
    # Process each variant
    shared_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for variant in config.variants:
        for section in config.evaluate_sections:
            # Process variant for this section
            agg = processor.process_variant(variant, section=section)
            
            # Save outputs
            out_dir, ts = save_outputs(agg, config, variant, section, shared_ts)
            shared_ts = ts  # Use same timestamp for all variants
            
            # Generate plots
            if config.plots:
                plotter.generate_plots(agg, out_dir, variant)
            
            # Print summary
            print_summary(agg)
    
    return 0


if __name__ == '__main__':
    exit(main())