"""
Clinical Trial Evaluation Statistics Analyzer

This module analyzes evaluation metrics from clinical trial data extraction models.
It processes metrics files, generates statistical summaries, and creates visualizations
to compare model performance across different fields and metrics.

Main outputs:
- Field-level performance statistics
- Model comparison metrics
- Target leaf field analysis
- Extra Values (EV) analysis
- Various visualizations (heatmaps, bar charts, etc.)
"""

import json
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
EVAL_METRICS_OUTPUT = BASE_DIR / "DATA" / "EVAL_METRICS_OUTPUT"
FIELD_CATALOG_PATH = BASE_DIR / "CTG_DOCUMENT" / "DataDefinition.csv"
EVAL_STATS_OUTPUT = BASE_DIR / "DATA" / "EVAL_STATS_OUTPUT"
CTG_DATA_PATH = BASE_DIR / "DATA" / "CTG"

# Visualization settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class FieldInfo:
    """Field metadata from the catalog"""
    field_index: str
    piece: str
    is_target: bool
    is_leaf: bool
    type: str
    is_list: bool


@dataclass
class FieldStats:
    """Statistics for a single field across all evaluations"""
    piece: str
    field_index: str
    total_count: int = 0
    tp_count: int = 0
    fp_count: int = 0
    fn_count: int = 0
    tn_count: int = 0
    ev_count: int = 0
    models: Dict[str, Dict[str, int]] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = defaultdict(lambda: defaultdict(int))


@dataclass
class ModelStats:
    """Aggregate statistics for a single model"""
    model: str
    key_precision_sum: float = 0.0
    key_recall_sum: float = 0.0
    key_f1_sum: float = 0.0
    value_precision_sum: float = 0.0
    value_recall_sum: float = 0.0
    value_f1_sum: float = 0.0
    case_count: int = 0
    field_stats: Dict[str, FieldStats] = None
    
    def __post_init__(self):
        if self.field_stats is None:
            self.field_stats = {}


# ============================================================================
# FIELD CATALOG MANAGEMENT
# ============================================================================

class FieldCatalog:
    """Manages field definitions and metadata from the catalog CSV"""
    
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)
        self.df.sort_values("No", inplace=True)
        
        # Normalize field indices by removing array brackets
        self.df['NormalizedIndex'] = self.df['FieldIndex'].str.replace(r'\[\d+\]', '', regex=True)
        
        # Filter target leaf fields
        self.target_leaf_df = self.df[(self.df['TargetYN'] == 'Y') & (self.df['IsLeaf'] == 'Y')]
        self.target_leaf_indices = set(self.target_leaf_df['FieldIndex'])
        
        # Build lookup dictionary
        self.field_map = {}
        for _, row in self.df.iterrows():
            self.field_map[row['FieldIndex']] = FieldInfo(
                field_index=row['FieldIndex'],
                piece=row['Piece'],
                is_target=row['TargetYN'] == 'Y',
                is_leaf=row['IsLeaf'] == 'Y',
                type=row['SourceType'],
                is_list=row['ListYN'] == 'Y'
            )
    
    def normalize_field_path(self, field_path: str) -> str:
        """Remove array indices from field path"""
        return re.sub(r'\[\d+\]', '', field_path)
    
    def get_field_info(self, field_path: str) -> FieldInfo:
        """Get field information by normalized path"""
        normalized = self.normalize_field_path(field_path)
        return self.field_map.get(normalized)
    
    def is_target_leaf(self, field_path: str) -> bool:
        """Check if field is a target leaf field"""
        normalized = self.normalize_field_path(field_path)
        return normalized in self.target_leaf_indices


# ============================================================================
# DATA ANALYSIS
# ============================================================================

class CTGDataAnalyzer:
    """Analyzes CTG source data to count field occurrences"""
    
    def __init__(self, ctg_path: Path, field_catalog: FieldCatalog):
        self.ctg_path = ctg_path
        self.catalog = field_catalog
        self.field_case_counts = defaultdict(set)
        self.field_total_counts = defaultdict(int)
        
    def flatten_json(self, data: Any, parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested JSON structure"""
        items = []
        
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.extend(self.flatten_json(v, new_key).items())
                else:
                    items.append((new_key, v))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                new_key = f"{parent_key}[{i}]"
                if isinstance(v, (dict, list)):
                    items.extend(self.flatten_json(v, new_key).items())
                else:
                    items.append((new_key, v))
        else:
            items.append((parent_key, data))
            
        return dict(items)
    
    def analyze_ctg_files(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Analyze all CTG files and count field occurrences"""
        ctg_files = list(self.ctg_path.glob("case_*.json"))
        print(f"Found {len(ctg_files)} CTG files")
        
        for file_path in ctg_files:
            # Extract case ID from filename
            match = re.search(r'case_(\d+)_', file_path.name)
            if not match:
                continue
            case_id = int(match.group(1))
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                flat_data = self.flatten_json(data)
                
                for flat_key, value in flat_data.items():
                    if value is not None:
                        normalized_key = self.catalog.normalize_field_path(flat_key)
                        field_info = self.catalog.get_field_info(normalized_key)
                        
                        if field_info and field_info.is_target and field_info.is_leaf:
                            self.field_case_counts[normalized_key].add(case_id)
                            self.field_total_counts[normalized_key] += 1
                            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        case_counts = {field: len(cases) for field, cases in self.field_case_counts.items()}
        return case_counts, self.field_total_counts


class MetricsProcessor:
    """Processes evaluation metrics files and aggregates statistics"""
    
    def __init__(self, field_catalog: FieldCatalog):
        self.catalog = field_catalog
        self.all_stats = defaultdict(FieldStats)
        self.model_stats = defaultdict(ModelStats)
        self.field_model_metrics = defaultdict(lambda: defaultdict(dict))
        
    def extract_model_from_filename(self, filename: str) -> str:
        """Extract model name from metrics filename"""
        match = re.search(r'case\d+_(.+?)_metrics\.json', filename)
        return match.group(1) if match else "unknown"
    
    def process_metrics_file(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Load and parse a single metrics file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            model = self.extract_model_from_filename(file_path.name)
            return model, data
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None
    
    def update_field_stats(self, model: str, metrics: Dict[str, Any]):
        """Update field-level statistics from metrics data"""
        categories = ['val_TP_fields', 'val_FP_fields', 'val_FN_fields', 
                     'val_TN_fields', 'val_EV_fields']
        
        for category in categories:
            if category not in metrics:
                continue
                
            fields = metrics[category]
            cat_type = category.split('_')[1]  # Extract: TP, FP, FN, TN, EV
            
            for field_path in fields:
                normalized = self.catalog.normalize_field_path(field_path)
                field_info = self.catalog.get_field_info(normalized)
                
                if not field_info or not (field_info.is_target and field_info.is_leaf):
                    continue
                
                piece = field_info.piece
                
                if piece not in self.all_stats:
                    self.all_stats[piece] = FieldStats(
                        piece=piece, 
                        field_index=field_info.field_index
                    )
                
                stats = self.all_stats[piece]
                stats.total_count += 1
                
                # Update category counts
                if cat_type == 'TP':
                    stats.tp_count += 1
                elif cat_type == 'FP':
                    stats.fp_count += 1
                elif cat_type == 'FN':
                    stats.fn_count += 1
                elif cat_type == 'TN':
                    stats.tn_count += 1
                elif cat_type == 'EV':
                    stats.ev_count += 1
                
                stats.models[model][cat_type] += 1
    
    def update_model_stats(self, model: str, metrics: Dict[str, Any]):
        """Update model-level aggregate statistics"""
        if model not in self.model_stats:
            self.model_stats[model] = ModelStats(model=model)
        
        stats = self.model_stats[model]
        stats.key_precision_sum += metrics.get('key_precision', 0)
        stats.key_recall_sum += metrics.get('key_recall', 0)
        stats.key_f1_sum += metrics.get('key_f1', 0)
        stats.value_precision_sum += metrics.get('value_precision', 0)
        stats.value_recall_sum += metrics.get('value_recall', 0)
        stats.value_f1_sum += metrics.get('value_f1', 0)
        stats.case_count += 1
    
    def process_all_files(self, base_dir: Path, eval_dir: Path):
        """Process all metrics files in the evaluation directory"""
        all_files = []
        
        print(f"\nChecking directory structure...")
        if eval_dir.exists():
            files = list(eval_dir.glob("case*_*_metrics.json"))
            all_files.extend(files)
            print(f"Found {len(files)} files in {eval_dir}")
        else:
            print(f"Directory not found: {eval_dir}")
        
        if all_files:
            print("Sample files:")
            for f in all_files[:3]:
                print(f"  - {f}")
        
        for file_path in all_files:
            model, metrics = self.process_metrics_file(file_path)
            if model and metrics:
                self.update_field_stats(model, metrics)
                self.update_model_stats(model, metrics)
        
        print(f"Successfully processed {len(all_files)} files")


class StatsCalculator:
    """Calculates derived statistics from raw counts"""
    
    @staticmethod
    def calculate_field_metrics(stats: FieldStats) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for a field"""
        tp = stats.tp_count
        fp = stats.fp_count
        fn = stats.fn_count
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': stats.total_count
        }
    
    @staticmethod
    def calculate_model_averages(stats: ModelStats) -> Dict[str, float]:
        """Calculate average metrics across all cases for a model"""
        if stats.case_count == 0:
            return {
                'key_precision': 0.0,
                'key_recall': 0.0,
                'key_f1': 0.0,
                'value_precision': 0.0,
                'value_recall': 0.0,
                'value_f1': 0.0,
                'case_count': 0
            }
        
        return {
            'key_precision': stats.key_precision_sum / stats.case_count,
            'key_recall': stats.key_recall_sum / stats.case_count,
            'key_f1': stats.key_f1_sum / stats.case_count,
            'value_precision': stats.value_precision_sum / stats.case_count,
            'value_recall': stats.value_recall_sum / stats.case_count,
            'value_f1': stats.value_f1_sum / stats.case_count,
            'case_count': stats.case_count
        }


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generates summary reports and CSV outputs"""
    
    def __init__(self, field_stats: Dict[str, FieldStats], 
                 model_stats: Dict[str, ModelStats],
                 catalog: FieldCatalog):
        self.field_stats = field_stats
        self.model_stats = model_stats
        self.catalog = catalog
        self.calc = StatsCalculator()
    
    def generate_field_summary(self, output_path: Path) -> pd.DataFrame:
        """Generate field-level performance summary"""
        rows = []
        
        for piece, stats in sorted(self.field_stats.items(), 
                                  key=lambda x: x[1].total_count, 
                                  reverse=True):
            metrics = self.calc.calculate_field_metrics(stats)
            
            row = {
                'Piece': piece,
                'FieldIndex': stats.field_index,
                'TotalCount': stats.total_count,
                'TP': stats.tp_count,
                'FP': stats.fp_count,
                'FN': stats.fn_count,
                'TN': stats.tn_count,
                'EV': stats.ev_count,
                'Precision': round(metrics['precision'], 3),
                'Recall': round(metrics['recall'], 3),
                'F1': round(metrics['f1'], 3)
            }
            rows.append(row)
        
        if not rows:
            print("No field statistics to generate")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path / 'field_summary.csv', index=False)
        
        print("\n=== TOP 10 MOST FREQUENT FIELDS ===")
        print(df.head(10)[['Piece', 'TotalCount', 'F1']].to_string(index=False))
        
        print("\n=== BOTTOM 10 FIELDS BY F1 SCORE (min 10 occurrences) ===")
        df_filtered = df[df['TotalCount'] >= 10].sort_values('F1')
        if len(df_filtered) > 0:
            print(df_filtered.head(10)[['Piece', 'TotalCount', 'F1']].to_string(index=False))
        
        return df
    
    def generate_model_summary(self, output_path: Path) -> pd.DataFrame:
        """Generate model-level performance summary"""
        rows = []
        
        for model, stats in sorted(self.model_stats.items()):
            averages = self.calc.calculate_model_averages(stats)
            
            row = {
                'Model': model,
                'Cases': averages['case_count'],
                'Key_Precision': round(averages['key_precision'], 3),
                'Key_Recall': round(averages['key_recall'], 3),
                'Key_F1': round(averages['key_f1'], 3),
                'Value_Precision': round(averages['value_precision'], 3),
                'Value_Recall': round(averages['value_recall'], 3),
                'Value_F1': round(averages['value_f1'], 3)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path / 'model_summary.csv', index=False)
        
        print("\n=== MODEL PERFORMANCE SUMMARY ===")
        print(df.to_string(index=False))
        
        return df


class TargetLeafSummaryGenerator:
    """Generates comprehensive analysis of target leaf fields"""
    
    def __init__(self, field_catalog: FieldCatalog, ctg_analyzer: CTGDataAnalyzer):
        self.catalog = field_catalog
        self.ctg_analyzer = ctg_analyzer
        
    def generate_target_leaf_summary(self, processor: MetricsProcessor, output_path: Path) -> pd.DataFrame:
        """Generate comprehensive summary for all target leaf fields"""
        print("\nAnalyzing CTG data for field occurrences...")
        ctg_case_counts, ctg_total_counts = self.ctg_analyzer.analyze_ctg_files()
        
        target_leaf_df = self.catalog.target_leaf_df.copy()
        rows = []
        calc = StatsCalculator()
        
        for _, field_row in target_leaf_df.iterrows():
            field_index = field_row['FieldIndex']
            piece = field_row['Piece']
            
            row_data = {
                'FieldIndex': field_index,
                'Piece': piece,
                'ListYN': field_row['ListYN'],
                'SourceType': field_row['SourceType'],
                'CTG_CaseCount': ctg_case_counts.get(field_index, 0),
                'CTG_TotalCount': ctg_total_counts.get(field_index, 0),
            }
            
            # Calculate overall metrics across all models
            total_model_count = 0
            overall_tp = 0
            overall_fp = 0
            overall_fn = 0
            
            models = sorted(processor.model_stats.keys())
            
            for model in models:
                if piece in processor.all_stats:
                    field_stats = processor.all_stats[piece]
                    model_data = field_stats.models.get(model, {})
                    
                    tp = model_data.get('TP', 0)
                    fp = model_data.get('FP', 0)
                    fn = model_data.get('FN', 0)
                    
                    overall_tp += tp
                    overall_fp += fp
                    overall_fn += fn
                    total_model_count += (tp + fp + fn)
            
            # Calculate overall metrics
            if (overall_tp + overall_fp + overall_fn) > 0:
                overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
                overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
                overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            else:
                overall_precision = overall_recall = overall_f1 = 0.0
            
            # Add overall metrics
            row_data['Overall_key_precision'] = round(overall_precision, 3)
            row_data['Overall_key_recall'] = round(overall_recall, 3)
            row_data['Overall_key_f1'] = round(overall_f1, 3)
            row_data['Overall_value_precision'] = round(overall_precision, 3)
            row_data['Overall_value_recall'] = round(overall_recall, 3)
            row_data['Overall_value_f1'] = round(overall_f1, 3)
            
            # Add per-model metrics
            for model in models:
                if piece in processor.all_stats:
                    field_stats = processor.all_stats[piece]
                    model_data = field_stats.models.get(model, {})
                    
                    tp = model_data.get('TP', 0)
                    fp = model_data.get('FP', 0)
                    fn = model_data.get('FN', 0)
                    
                    if (tp + fp + fn) > 0:
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    else:
                        precision = recall = f1 = 0.0
                else:
                    precision = recall = f1 = 0.0
                
                row_data[f'{model}_key_precision'] = round(precision, 3)
                row_data[f'{model}_key_recall'] = round(recall, 3)
                row_data[f'{model}_key_f1'] = round(f1, 3)
                row_data[f'{model}_value_precision'] = round(precision, 3)
                row_data[f'{model}_value_recall'] = round(recall, 3)
                row_data[f'{model}_value_f1'] = round(f1, 3)
            
            row_data['Total_ModelCount'] = total_model_count
            rows.append(row_data)
        
        df = pd.DataFrame(rows)
        df.index = range(1, len(df) + 1)
        
        output_file = output_path / 'target_leaf_summary.csv'
        df.to_csv(output_file, index=True, index_label='Index')
        
        print(f"\nTarget Leaf Summary saved to: {output_file}")
        print(f"Total Target Leaf fields: {len(df)}")
        
        print("\n=== TARGET LEAF FIELDS STATISTICS ===")
        print(f"Fields appearing in CTG: {len(df[df['CTG_CaseCount'] > 0])}")
        print(f"Fields not in CTG: {len(df[df['CTG_CaseCount'] == 0])}")
        print(f"Fields evaluated by models: {len(df[df['Total_ModelCount'] > 0])}")
        print(f"Fields not evaluated: {len(df[df['Total_ModelCount'] == 0])}")
        
        return df


class EVAnalysisGenerator:
    """Analyzes Extra Values (EV) classifications"""
    
    def __init__(self, field_catalog: FieldCatalog):
        self.catalog = field_catalog
        
    def generate_ev_analysis(self, processor: MetricsProcessor, output_path: Path) -> Dict[str, Any]:
        """Generate comprehensive EV analysis"""
        print("\n" + "="*60)
        print("EV (EXTRA VALUES) ANALYSIS")
        print("="*60)
        
        ev_stats = self._collect_ev_statistics(processor)
        model_ev_analysis = self._analyze_ev_by_model(processor)
        field_ev_analysis = self._analyze_ev_by_field(processor)
        ev_field_characteristics = self._analyze_ev_field_characteristics(processor)
        
        self._save_ev_analysis(ev_stats, model_ev_analysis, field_ev_analysis, 
                              ev_field_characteristics, output_path)
        
        self._print_ev_summary(ev_stats, model_ev_analysis, field_ev_analysis)
        
        return {
            'ev_stats': ev_stats,
            'model_analysis': model_ev_analysis,
            'field_analysis': field_ev_analysis,
            'characteristics': ev_field_characteristics
        }
    
    def _collect_ev_statistics(self, processor: MetricsProcessor) -> Dict[str, Any]:
        """Collect overall EV statistics"""
        total_ev_count = 0
        total_all_count = 0
        model_ev_counts = {}
        
        for model, stats in processor.model_stats.items():
            model_ev_count = 0
            model_total_count = 0
            
            for piece, field_stats in processor.all_stats.items():
                if model in field_stats.models:
                    model_data = field_stats.models[model]
                    ev_count = model_data.get('EV', 0)
                    model_ev_count += ev_count
                    
                    total_for_field = sum(model_data.values())
                    model_total_count += total_for_field
            
            model_ev_counts[model] = {
                'ev_count': model_ev_count,
                'total_count': model_total_count,
                'ev_percentage': (model_ev_count / model_total_count * 100) if model_total_count > 0 else 0
            }
            
            total_ev_count += model_ev_count
            total_all_count += model_total_count
        
        overall_ev_percentage = (total_ev_count / total_all_count * 100) if total_all_count > 0 else 0
        
        return {
            'total_ev_count': total_ev_count,
            'total_evaluated_count': total_all_count,
            'overall_ev_percentage': overall_ev_percentage,
            'model_breakdown': model_ev_counts
        }
    
    def _analyze_ev_by_model(self, processor: MetricsProcessor) -> Dict[str, Any]:
        """Analyze EV occurrences by model"""
        model_analysis = {}
        
        for model, stats in processor.model_stats.items():
            ev_fields = []
            ev_count = 0
            total_count = 0
            
            for piece, field_stats in processor.all_stats.items():
                if model in field_stats.models:
                    model_data = field_stats.models[model]
                    ev_for_field = model_data.get('EV', 0)
                    
                    if ev_for_field > 0:
                        field_info = self.catalog.get_field_info(field_stats.field_index)
                        ev_fields.append({
                            'piece': piece,
                            'field_index': field_stats.field_index,
                            'ev_count': ev_for_field,
                            'source_type': field_info.type if field_info else 'UNKNOWN',
                            'is_list': field_info.is_list if field_info else False
                        })
                    
                    ev_count += ev_for_field
                    total_count += sum(model_data.values())
            
            ev_fields.sort(key=lambda x: x['ev_count'], reverse=True)
            
            model_analysis[model] = {
                'total_ev_count': ev_count,
                'total_evaluated': total_count,
                'ev_percentage': (ev_count / total_count * 100) if total_count > 0 else 0,
                'ev_fields': ev_fields,
                'top_ev_fields': ev_fields[:10]
            }
        
        return model_analysis
    
    def _analyze_ev_by_field(self, processor: MetricsProcessor) -> Dict[str, Any]:
        """Analyze EV occurrences by field"""
        field_analysis = []
        
        for piece, field_stats in processor.all_stats.items():
            total_ev = field_stats.ev_count
            total_occurrences = field_stats.total_count
            
            if total_ev > 0:
                field_info = self.catalog.get_field_info(field_stats.field_index)
                
                model_breakdown = {}
                for model, model_data in field_stats.models.items():
                    ev_count = model_data.get('EV', 0)
                    if ev_count > 0:
                        model_total = sum(model_data.values())
                        model_breakdown[model] = {
                            'ev_count': ev_count,
                            'total_for_model': model_total,
                            'ev_percentage': (ev_count / model_total * 100) if model_total > 0 else 0
                        }
                
                field_analysis.append({
                    'piece': piece,
                    'field_index': field_stats.field_index,
                    'total_ev_count': total_ev,
                    'total_occurrences': total_occurrences,
                    'ev_percentage': (total_ev / total_occurrences * 100) if total_occurrences > 0 else 0,
                    'source_type': field_info.type if field_info else 'UNKNOWN',
                    'is_list': field_info.is_list if field_info else False,
                    'is_target': field_info.is_target if field_info else False,
                    'is_leaf': field_info.is_leaf if field_info else False,
                    'model_breakdown': model_breakdown
                })
        
        field_analysis.sort(key=lambda x: x['ev_percentage'], reverse=True)
        
        return {
            'total_fields_with_ev': len(field_analysis),
            'field_details': field_analysis,
            'high_ev_fields': [f for f in field_analysis if f['ev_percentage'] > 50],
            'medium_ev_fields': [f for f in field_analysis if 20 < f['ev_percentage'] <= 50],
            'low_ev_fields': [f for f in field_analysis if f['ev_percentage'] <= 20]
        }
    
    def _analyze_ev_field_characteristics(self, processor: MetricsProcessor) -> Dict[str, Any]:
        """Analyze characteristics of fields classified as EV"""
        type_distribution = defaultdict(int)
        list_vs_nonlist = {'list': 0, 'non_list': 0}
        target_vs_nontarget = {'target': 0, 'non_target': 0}
        leaf_vs_nonleaf = {'leaf': 0, 'non_leaf': 0}
        
        for piece, field_stats in processor.all_stats.items():
            if field_stats.ev_count > 0:
                field_info = self.catalog.get_field_info(field_stats.field_index)
                
                if field_info:
                    type_distribution[field_info.type] += field_stats.ev_count
                    
                    if field_info.is_list:
                        list_vs_nonlist['list'] += field_stats.ev_count
                    else:
                        list_vs_nonlist['non_list'] += field_stats.ev_count
                    
                    if field_info.is_target:
                        target_vs_nontarget['target'] += field_stats.ev_count
                    else:
                        target_vs_nontarget['non_target'] += field_stats.ev_count
                    
                    if field_info.is_leaf:
                        leaf_vs_nonleaf['leaf'] += field_stats.ev_count
                    else:
                        leaf_vs_nonleaf['non_leaf'] += field_stats.ev_count
        
        total_ev = sum(type_distribution.values())
        
        return {
            'total_ev_analyzed': total_ev,
            'type_distribution': dict(type_distribution),
            'type_percentages': {k: (v/total_ev*100) for k, v in type_distribution.items()} if total_ev > 0 else {},
            'list_distribution': list_vs_nonlist,
            'list_percentages': {k: (v/total_ev*100) for k, v in list_vs_nonlist.items()} if total_ev > 0 else {},
            'target_distribution': target_vs_nontarget,
            'target_percentages': {k: (v/total_ev*100) for k, v in target_vs_nontarget.items()} if total_ev > 0 else {},
            'leaf_distribution': leaf_vs_nonleaf,
            'leaf_percentages': {k: (v/total_ev*100) for k, v in leaf_vs_nonleaf.items()} if total_ev > 0 else {}
        }
    
    def _save_ev_analysis(self, ev_stats: Dict, model_analysis: Dict, 
                         field_analysis: Dict, characteristics: Dict, output_path: Path):
        """Save EV analysis results to files"""
        # Model-level EV summary
        ev_summary_rows = []
        for model, data in ev_stats['model_breakdown'].items():
            ev_summary_rows.append({
                'Model': model,
                'EV_Count': data['ev_count'],
                'Total_Evaluated': data['total_count'],
                'EV_Percentage': round(data['ev_percentage'], 2)
            })
        
        ev_summary_df = pd.DataFrame(ev_summary_rows)
        ev_summary_df.to_csv(output_path / 'ev_summary_by_model.csv', index=False)
        
        # Field-level EV analysis
        field_ev_rows = []
        for field_data in field_analysis['field_details']:
            field_ev_rows.append({
                'Piece': field_data['piece'],
                'FieldIndex': field_data['field_index'],
                'Total_EV_Count': field_data['total_ev_count'],
                'Total_Occurrences': field_data['total_occurrences'],
                'EV_Percentage': round(field_data['ev_percentage'], 2),
                'SourceType': field_data['source_type'],
                'IsList': field_data['is_list'],
                'IsTarget': field_data['is_target'],
                'IsLeaf': field_data['is_leaf']
            })
        
        field_ev_df = pd.DataFrame(field_ev_rows)
        field_ev_df.to_csv(output_path / 'ev_analysis_by_field.csv', index=False)
        
        # Save characteristics and full analysis as JSON
        with open(output_path / 'ev_characteristics.json', 'w', encoding='utf-8') as f:
            json.dump(characteristics, f, indent=2, ensure_ascii=False)
        
        full_analysis = {
            'summary': ev_stats,
            'model_analysis': model_analysis,
            'field_analysis': field_analysis,
            'characteristics': characteristics
        }
        
        with open(output_path / 'ev_full_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(full_analysis, f, indent=2, ensure_ascii=False)
    
    def _print_ev_summary(self, ev_stats: Dict, model_analysis: Dict, field_analysis: Dict):
        """Print EV analysis summary to console"""
        print(f"\n>> Overall EV Statistics:")
        print(f"   • Total EV count: {ev_stats['total_ev_count']:,}")
        print(f"   • Total evaluated items: {ev_stats['total_evaluated_count']:,}")
        print(f"   • Overall EV percentage: {ev_stats['overall_ev_percentage']:.2f}%")
        
        print(f"\n>> EV Percentage by Model:")
        for model, data in ev_stats['model_breakdown'].items():
            print(f"   • {model}: {data['ev_percentage']:.2f}% ({data['ev_count']:,}/{data['total_count']:,})")
        
        print(f"\n>> Field-level EV Analysis:")
        print(f"   • Fields with EV: {field_analysis['total_fields_with_ev']:,}")
        print(f"   • High EV fields (>50%): {len(field_analysis['high_ev_fields']):,}")
        print(f"   • Medium EV fields (20-50%): {len(field_analysis['medium_ev_fields']):,}")
        print(f"   • Low EV fields (≤20%): {len(field_analysis['low_ev_fields']):,}")
        
        if field_analysis['high_ev_fields']:
            print(f"\n>> Top 5 Fields with Highest EV Percentage:")
            for i, field in enumerate(field_analysis['high_ev_fields'][:5], 1):
                print(f"   {i}. {field['piece']}: {field['ev_percentage']:.1f}% ({field['total_ev_count']}/{field['total_occurrences']})")


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Creates visualizations for analysis results"""
    
    @staticmethod
    def plot_field_frequency(field_stats: Dict[str, FieldStats], top_n: int = 30):
        """Plot most frequent fields"""
        sorted_fields = sorted(field_stats.items(), 
                             key=lambda x: x[1].total_count, 
                             reverse=True)[:top_n]
        
        pieces = [item[0] for item in sorted_fields]
        counts = [item[1].total_count for item in sorted_fields]
        
        plt.figure(figsize=(12, 8))
        plt.barh(pieces, counts)
        plt.xlabel('Total Count')
        plt.ylabel('Field (Piece)')
        plt.title(f'Top {top_n} Most Frequent Fields')
        plt.tight_layout()
        plt.savefig(EVAL_STATS_OUTPUT / 'field_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_field_performance(field_stats: Dict[str, FieldStats], top_n: int = 30):
        """Plot fields with lowest F1 scores"""
        calc = StatsCalculator()
        
        field_metrics = []
        for piece, stats in field_stats.items():
            metrics = calc.calculate_field_metrics(stats)
            if stats.total_count > 10:
                field_metrics.append((piece, metrics['f1'], stats.total_count))
        
        field_metrics.sort(key=lambda x: x[1])
        field_metrics = field_metrics[:top_n]
        
        pieces = [item[0] for item in field_metrics]
        f1_scores = [item[1] for item in field_metrics]
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if f1 < 0.5 else 'yellow' if f1 < 0.8 else 'green' 
                 for f1 in f1_scores]
        plt.barh(pieces, f1_scores, color=colors)
        plt.xlabel('F1 Score')
        plt.ylabel('Field (Piece)')
        plt.title(f'Bottom {top_n} Fields by F1 Score (min 10 occurrences)')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(EVAL_STATS_OUTPUT / 'field_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(model_stats: Dict[str, ModelStats]):
        """Plot model performance comparison"""
        calc = StatsCalculator()

        models = []
        metrics_data = {
            'key_precision': [],
            'key_recall': [],
            'key_f1': [],
            'value_precision': [],
            'value_recall': [],
            'value_f1': []
        }

        for model, stats in sorted(model_stats.items()):
            models.append(model)
            averages = calc.calculate_model_averages(stats)
            for metric in metrics_data:
                metrics_data[metric].append(averages[metric])

        x = np.arange(len(models))
        width = 0.22

        colors = ['#FF6F61', '#6FCF97', '#56CCF2']  # coral, mint green, sky blue

        # Key metrics plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        bars1 = ax1.bar(x - width, metrics_data['key_precision'], width, label='Precision', color=colors[0])
        bars2 = ax1.bar(x, metrics_data['key_recall'], width, label='Recall', color=colors[1])
        bars3 = ax1.bar(x + width, metrics_data['key_f1'], width, label='F1', color=colors[2])
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Key-level Metrics by Model', pad=30)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(EVAL_STATS_OUTPUT / 'model_comparison_key.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Value metrics plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars4 = ax2.bar(x - width, metrics_data['value_precision'], width, label='Precision', color=colors[0])
        bars5 = ax2.bar(x, metrics_data['value_recall'], width, label='Recall', color=colors[1])
        bars6 = ax2.bar(x + width, metrics_data['value_f1'], width, label='F1', color=colors[2])
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Score')
        ax2.set_title('Value-level Metrics by Model', pad=30)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        for bars in [bars4, bars5, bars6]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(EVAL_STATS_OUTPUT / 'model_comparison_value.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_model_field_heatmap(field_stats: Dict[str, FieldStats], 
                               models: List[str], catalog: FieldCatalog, top_n: int = 49):
        """Create heatmap of model performance by field"""
        calc = StatsCalculator()
        
        catalog_ordered_fields = []
        for _, row in catalog.target_leaf_df.iterrows():
            piece = row['Piece']
            if piece in field_stats and field_stats[piece].total_count > 0:
                catalog_ordered_fields.append((piece, field_stats[piece]))
        
        selected_fields = catalog_ordered_fields[:top_n]
        heatmap_data = []
        field_names = []
        
        for piece, stats in selected_fields:
            field_names.append(piece)
            row = []
            for model in models:
                model_stats = stats.models.get(model, {})
                tp = model_stats.get('TP', 0)
                fp = model_stats.get('FP', 0)
                fn = model_stats.get('FN', 0)
                
                if (tp + fp + fn) > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    f1 = np.nan
                row.append(f1)
            heatmap_data.append(row)
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(heatmap_data, 
                   xticklabels=models, 
                   yticklabels=field_names,
                   cmap='RdYlGn',
                   vmin=0, 
                   vmax=1,
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': 'F1 Score'})
        plt.title(f'Model Performance Heatmap (Ordered by DataDefinition.csv)')
        plt.xlabel('Model')
        plt.ylabel('Field (Piece)')
        plt.tight_layout()
        plt.savefig(EVAL_STATS_OUTPUT / 'model_field_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_ev_analysis(ev_analysis_results: Dict[str, Any], output_path: Path):
        """Create visualizations for EV analysis"""
        # Model comparison bar chart
        plt.figure(figsize=(10, 6))
        model_data = ev_analysis_results['ev_stats']['model_breakdown']
        models = list(model_data.keys())
        ev_percentages = [model_data[model]['ev_percentage'] for model in models]
        ev_counts = [model_data[model]['ev_count'] for model in models]
        
        bars = plt.bar(models, ev_percentages, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], width=0.5)
        plt.ylabel('EV Percentage (%)')
        plt.title('EV (Extra Values) Ratio by Model')
        plt.ylim(0, max(ev_percentages) * 1.2)
        
        for i, (bar, count, pct) in enumerate(zip(bars, ev_counts, ev_percentages)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{pct:.1f}%\n({count:,})', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path / 'ev_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # EV type distribution pie chart
        plt.figure(figsize=(10, 8))
        characteristics = ev_analysis_results['characteristics']
        type_data = characteristics['type_distribution']
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        wedges, texts, autotexts = plt.pie(type_data.values(), 
                                          labels=type_data.keys(), 
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        
        plt.title('EV Fields Distribution by Source Type', fontsize=14, fontweight='bold')
        
        legend_labels = [f'{label}: {count:,}' for label, count in type_data.items()]
        plt.legend(wedges, legend_labels, title="Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(output_path / 'ev_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Top EV fields horizontal bar chart
        plt.figure(figsize=(12, 8))
        field_analysis = ev_analysis_results['field_analysis']
        top_ev_fields = field_analysis['field_details'][:15]
        
        pieces = [field['piece'] for field in top_ev_fields]
        ev_percentages = [field['ev_percentage'] for field in top_ev_fields]
        ev_counts = [field['total_ev_count'] for field in top_ev_fields]
        
        colors = []
        for pct in ev_percentages:
            if pct == 100:
                colors.append('#ff4757')
            elif pct >= 90:
                colors.append('#ff6348')
            elif pct >= 50:
                colors.append('#ffa502')
            else:
                colors.append('#70a1ff')
        
        bars = plt.barh(range(len(pieces)), ev_percentages, color=colors)
        plt.yticks(range(len(pieces)), pieces)
        plt.xlabel('EV Percentage (%)')
        plt.title('Top 15 Fields by EV Percentage')
        plt.xlim(0, 105)
        
        for i, (bar, count, pct) in enumerate(zip(bars, ev_counts, ev_percentages)):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}% ({count})', 
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / 'ev_top_fields.png', dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("CLINICAL TRIAL EVALUATION STATISTICS ANALYZER")
    print("="*80)
    
    # Setup output directory
    EVAL_STATS_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {EVAL_STATS_OUTPUT}")
    
    # Load field catalog
    print("\n1. Loading field catalog...")
    if not FIELD_CATALOG_PATH.exists():
        print(f"ERROR: Field catalog not found at {FIELD_CATALOG_PATH}")
        return
        
    catalog = FieldCatalog(FIELD_CATALOG_PATH)
    print(f"   Loaded {len(catalog.field_map)} fields")
    print(f"   Target leaf fields: {len(catalog.target_leaf_indices)}")
    
    # Process metrics files
    print("\n2. Processing metrics files...")
    processor = MetricsProcessor(catalog)
    processor.process_all_files(BASE_DIR, EVAL_METRICS_OUTPUT)
    
    if not processor.all_stats:
        print("\nNo data was processed. Please check:")
        print("1. Directory paths are correct")
        print("2. Metrics files exist in expected format")
        print("3. File naming pattern: case*_*_metrics.json")
        return
    
    # Generate reports
    print("\n3. Generating statistics...")
    report_gen = ReportGenerator(processor.all_stats, processor.model_stats, catalog)
    field_df = report_gen.generate_field_summary(EVAL_STATS_OUTPUT)
    model_df = report_gen.generate_model_summary(EVAL_STATS_OUTPUT)
    
    # Generate Target Leaf Summary
    print("\n4. Generating Target Leaf Summary...")
    ctg_analyzer = CTGDataAnalyzer(CTG_DATA_PATH, catalog)
    target_leaf_generator = TargetLeafSummaryGenerator(catalog, ctg_analyzer)
    target_leaf_df = target_leaf_generator.generate_target_leaf_summary(processor, EVAL_STATS_OUTPUT)
    
    # Generate EV Analysis
    print("\n5. Generating EV (Extra Values) Analysis...")
    ev_analyzer = EVAnalysisGenerator(catalog)
    ev_analysis_results = ev_analyzer.generate_ev_analysis(processor, EVAL_STATS_OUTPUT)
    
    # Create visualizations
    if len(processor.all_stats) > 0 and len(processor.model_stats) > 0:
        print("\n6. Creating visualizations...")
        visualizer = Visualizer()
        
        try:
            visualizer.plot_field_frequency(processor.all_stats, top_n=30)
            visualizer.plot_field_performance(processor.all_stats, top_n=30)
            visualizer.plot_model_comparison(processor.model_stats)
            
            models = sorted(processor.model_stats.keys())
            visualizer.plot_model_field_heatmap(processor.all_stats, models, catalog, top_n=49)
            
            visualizer.plot_ev_analysis(ev_analysis_results, EVAL_STATS_OUTPUT)
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        print("\n6. Skipping visualizations (insufficient data)")
    
    # Summary
    print("\n7. Analysis complete!")
    print(f"   Results saved to: {EVAL_STATS_OUTPUT}")
    print("   Generated files:")
    print("   - field_summary.csv")
    print("   - model_summary.csv")
    print("   - target_leaf_summary.csv")
    print("   - ev_summary_by_model.csv")
    print("   - ev_analysis_by_field.csv")
    print("   - ev_characteristics.json")
    print("   - ev_full_analysis.json")
    print("   - Visualization PNG files")
    
    # Key insights
    if len(field_df) > 0 and len(model_df) > 0:
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        challenging = field_df[(field_df['F1'] < 0.5) & (field_df['TotalCount'] >= 20)]
        print(f"\nChallenging Fields (F1 < 0.5, min 20 occurrences): {len(challenging)}")
        
        if len(model_df) > 0:
            best_key_model = model_df.loc[model_df['Key_F1'].idxmax(), 'Model']
            best_val_model = model_df.loc[model_df['Value_F1'].idxmax(), 'Model']
            print(f"\nBest Performance:")
            print(f"   Key-level: {best_key_model}")
            print(f"   Value-level: {best_val_model}")


if __name__ == "__main__":
    main()