import json
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
import re

###############################################################################
# CONFIGURATION
###############################################################################
load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
EVAL_METRICS_OUTPUT = BASE_DIR / "DATA" / "EVAL_METRICS_OUTPUT"
FIELD_CATALOG_PATH = BASE_DIR / "CTG_DOCUMENT" / "DataDefinition.csv"
EVAL_STATS_OUTPUT = BASE_DIR / "DATA" / "EVAL_STATS_OUTPUT"

# Visualization settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

###############################################################################
# DATA STRUCTURES
###############################################################################

@dataclass
class FieldInfo:
    """Field information structure"""
    field_index: str
    piece: str
    is_target: bool
    is_leaf: bool
    type: str
    is_list: bool

@dataclass
class FieldStats:
    """Field-level statistics structure"""
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
    """Model-level statistics structure"""
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

###############################################################################
# FIELD CATALOG LOADER
###############################################################################

class FieldCatalog:
    """Field catalog management class"""
    
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)
        self.df.sort_values("No", inplace=True)
        
        # Normalize index (remove list indices)
        self.df['NormalizedIndex'] = self.df['FieldIndex'].str.replace(r'\[\d+\]', '', regex=True)
        
        # Filter only Target Leaf fields
        self.target_leaf_df = self.df[(self.df['TargetYN'] == 'Y') & (self.df['IsLeaf'] == 'Y')]
        self.target_leaf_indices = set(self.target_leaf_df['FieldIndex'])
        
        # Dictionary for fast lookup
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
        """Normalize field path (remove list indices)"""
        return re.sub(r'\[\d+\]', '', field_path)
    
    def get_field_info(self, field_path: str) -> FieldInfo:
        """Get field information by normalized field path"""
        normalized = self.normalize_field_path(field_path)
        # Try exact matching
        if normalized in self.field_map:
            return self.field_map[normalized]
        # Return None if not found
        return None
    
    def is_target_leaf(self, field_path: str) -> bool:
        """Check if field is Target Leaf"""
        normalized = self.normalize_field_path(field_path)
        return normalized in self.target_leaf_indices

###############################################################################
# METRICS FILE PROCESSOR
###############################################################################

class MetricsProcessor:
    """Metrics file processing class"""
    
    def __init__(self, field_catalog: FieldCatalog):
        self.catalog = field_catalog
        self.all_stats = defaultdict(FieldStats)
        self.model_stats = defaultdict(ModelStats)
        
    def extract_model_from_filename(self, filename: str) -> str:
        """Extract model name from filename"""
        # case1_gpt-4o_metrics.json -> gpt-4o
        match = re.search(r'case\d+_(.+?)_metrics\.json', filename)
        return match.group(1) if match else "unknown"
    
    def process_metrics_file(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process single metrics file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            model = self.extract_model_from_filename(file_path.name)
            return model, data
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None
    
    def update_field_stats(self, model: str, metrics: Dict[str, Any]):
        """Update field-level statistics"""
        # Field lists by category
        categories = ['val_TP_fields', 'val_FP_fields', 'val_FN_fields', 
                     'val_TN_fields', 'val_EV_fields']
        
        for category in categories:
            if category not in metrics:
                continue
                
            fields = metrics[category]
            cat_type = category.split('_')[1]  # TP, FP, FN, TN, EV
            
            for field_path in fields:
                normalized = self.catalog.normalize_field_path(field_path)
                field_info = self.catalog.get_field_info(normalized)
                
                if not field_info:
                    continue
                
                # Process only Target Leaf fields
                if not (field_info.is_target and field_info.is_leaf):
                    continue
                
                piece = field_info.piece
                
                # Update overall statistics
                if piece not in self.all_stats:
                    self.all_stats[piece] = FieldStats(
                        piece=piece, 
                        field_index=field_info.field_index
                    )
                
                stats = self.all_stats[piece]
                stats.total_count += 1
                
                # Update category-specific counts
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
                
                # Update model-specific statistics
                stats.models[model][cat_type] += 1
    
    def update_model_stats(self, model: str, metrics: Dict[str, Any]):
        """Update model-level statistics"""
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
    
    def process_all_files(self, base_dir: Path, folders: List[str]):
        """Process all metrics files"""
        all_files = []
        
        # Check actual directory structure
        print(f"\nChecking directory structure...")
        
        # Check inside EVALUATION folder
        eval_dir = EVAL_METRICS_OUTPUT
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

###############################################################################
# STATISTICS CALCULATOR
###############################################################################

class StatsCalculator:
    """Statistics calculation class"""
    
    @staticmethod
    def calculate_field_metrics(stats: FieldStats) -> Dict[str, float]:
        """Calculate field-level metrics"""
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
        """Calculate model-level average metrics"""
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

###############################################################################
# VISUALIZATION
###############################################################################

class Visualizer:
    """Visualization class"""
    
    @staticmethod
    def plot_field_frequency(field_stats: Dict[str, FieldStats], top_n: int = 20):
        """Visualize field frequency"""
        # Create output directory
        EVAL_STATS_OUTPUT.mkdir(exist_ok=True)
        
        # Sort by frequency
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
    def plot_field_performance(field_stats: Dict[str, FieldStats], top_n: int = 20):
        """Visualize field performance (F1 score)"""
        calc = StatsCalculator()
        
        # Calculate and sort F1 scores
        field_metrics = []
        for piece, stats in field_stats.items():
            metrics = calc.calculate_field_metrics(stats)
            if stats.total_count > 10:  # Only fields appearing more than 10 times
                field_metrics.append((piece, metrics['f1'], stats.total_count))
        
        # Sort by F1 score
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
        """Visualize model performance comparison (separated by key/value with numerical values and adjusted title position)"""
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
        width = 0.2
        
        # Key metrics plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        bars1 = ax1.bar(x - width, metrics_data['key_precision'], width, label='Precision')
        bars2 = ax1.bar(x, metrics_data['key_recall'], width, label='Recall')
        bars3 = ax1.bar(x + width, metrics_data['key_f1'], width, label='F1')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Key-level Metrics by Model', pad=30)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Annotate values
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
        bars4 = ax2.bar(x - width, metrics_data['value_precision'], width, label='Precision')
        bars5 = ax2.bar(x, metrics_data['value_recall'], width, label='Recall')
        bars6 = ax2.bar(x + width, metrics_data['value_f1'], width, label='F1')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Score')
        ax2.set_title('Value-level Metrics by Model', pad=30)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Annotate values
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
                               models: List[str], top_n: int = 30):
        """Model-field F1 score heatmap"""
        calc = StatsCalculator()
        
        # Select top fields by frequency
        sorted_fields = sorted(field_stats.items(), 
                             key=lambda x: x[1].total_count, 
                             reverse=True)[:top_n]
        
        # Prepare heatmap data
        heatmap_data = []
        field_names = []
        
        for piece, stats in sorted_fields:
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
        
        # Draw heatmap
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
        plt.title(f'Model Performance Heatmap (Top {top_n} Fields by Frequency)')
        plt.xlabel('Model')
        plt.ylabel('Field (Piece)')
        plt.tight_layout()
        plt.savefig(EVAL_STATS_OUTPUT / 'model_field_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

###############################################################################
# REPORT GENERATOR
###############################################################################

class ReportGenerator:
    """Report generation class"""
    
    def __init__(self, field_stats: Dict[str, FieldStats], 
                 model_stats: Dict[str, ModelStats],
                 catalog: FieldCatalog):
        self.field_stats = field_stats
        self.model_stats = model_stats
        self.catalog = catalog
        self.calc = StatsCalculator()
    
    def generate_field_summary(self, output_path: Path):
        """Generate field-level summary statistics"""
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
        
        # Print top/bottom fields
        print("\n=== TOP 10 MOST FREQUENT FIELDS ===")
        if len(df) > 0:
            print(df.head(10)[['Piece', 'TotalCount', 'F1']].to_string(index=False))
        else:
            print("No data available")
        
        print("\n=== BOTTOM 10 FIELDS BY F1 SCORE (min 10 occurrences) ===")
        if len(df) > 0:
            df_filtered = df[df['TotalCount'] >= 10].sort_values('F1')
            if len(df_filtered) > 0:
                print(df_filtered.head(10)[['Piece', 'TotalCount', 'F1']].to_string(index=False))
            else:
                print("No fields with sufficient occurrences")
        else:
            print("No data available")
        
        return df
    
    def generate_model_summary(self, output_path: Path):
        """Generate model-level summary statistics"""
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
    
    def generate_model_field_summary(self, output_path: Path):
        """Generate detailed model-field statistics"""
        rows = []
        
        for piece, field_stats in self.field_stats.items():
            for model in sorted(self.model_stats.keys()):
                model_data = field_stats.models.get(model, {})
                
                tp = model_data.get('TP', 0)
                fp = model_data.get('FP', 0)
                fn = model_data.get('FN', 0)
                tn = model_data.get('TN', 0)
                ev = model_data.get('EV', 0)
                
                total = tp + fp + fn + tn + ev
                
                if total > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    row = {
                        'Model': model,
                        'Piece': piece,
                        'FieldIndex': field_stats.field_index,
                        'Total': total,
                        'TP': tp,
                        'FP': fp,
                        'FN': fn,
                        'TN': tn,
                        'EV': ev,
                        'Precision': round(precision, 3),
                        'Recall': round(recall, 3),
                        'F1': round(f1, 3)
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path / 'model_field_summary.csv', index=False)
        
        # Worst performing fields by model
        print("\n=== WORST PERFORMING FIELDS BY MODEL ===")
        for model in sorted(self.model_stats.keys()):
            print(f"\n{model}:")
            model_df = df[df['Model'] == model]
            worst = model_df[model_df['Total'] >= 5].sort_values('F1').head(5)
            if not worst.empty:
                print(worst[['Piece', 'Total', 'F1']].to_string(index=False))
            else:
                print("No fields with sufficient data")
        
        return df

###############################################################################
# MAIN EXECUTION
###############################################################################

def main():
    """메인 실행 함수"""
    print("="*80)
    print("EVALUATION RESULTS STATISTICS ANALYZER")
    print("="*80)
    
    print("\n1. Loading field catalog...")
    if not FIELD_CATALOG_PATH.exists():
        print(f"ERROR: Field catalog not found at {FIELD_CATALOG_PATH}")
        return
        
    catalog = FieldCatalog(FIELD_CATALOG_PATH)
    print(f"   Loaded {len(catalog.field_map)} fields")
    print(f"   Target leaf fields: {len(catalog.target_leaf_indices)}")
    
    print("\n2. Processing metrics files...")
    processor = MetricsProcessor(catalog)
    processor.process_all_files(BASE_DIR, EVAL_METRICS_OUTPUT)
    
    if not processor.all_stats:
        print("\nNo data was processed. Please check:")
        print("1. The directory paths are correct")
        print("2. The metrics files exist in the expected format")
        print("3. The file naming pattern matches: case*_*_metrics.json")
        return
    
    print("\n3. Generating statistics...")
    
    report_gen = ReportGenerator(processor.all_stats, 
                                processor.model_stats, 
                                catalog)
    
    field_df = report_gen.generate_field_summary(EVAL_STATS_OUTPUT)
    model_df = report_gen.generate_model_summary(EVAL_STATS_OUTPUT)
    model_field_df = report_gen.generate_model_field_summary(EVAL_STATS_OUTPUT)
    
    if len(processor.all_stats) > 0 and len(processor.model_stats) > 0:
        print("\n4. Creating visualizations...")
        visualizer = Visualizer()
        
        try:
            visualizer.plot_field_frequency(processor.all_stats, top_n=20)
            
            visualizer.plot_field_performance(processor.all_stats, top_n=20)
            
            visualizer.plot_model_comparison(processor.model_stats)
            
            models = sorted(processor.model_stats.keys())
            visualizer.plot_model_field_heatmap(processor.all_stats, models, top_n=30)
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        print("\n4. Skipping visualizations (insufficient data)")
    
    print("\n5. Analysis complete!")
    print(f"   Results saved to: {EVAL_STATS_OUTPUT}")
    print(f"   - field_summary.csv")
    print(f"   - model_summary.csv")
    print(f"   - model_field_summary.csv")
    if len(processor.all_stats) > 0:
        print(f"   - Visualization PNG files")
    
    if len(field_df) > 0 and len(model_df) > 0:
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        print("\nMost Challenging Fields (F1 < 0.5, min 20 occurrences):")
        challenging = field_df[(field_df['F1'] < 0.5) & (field_df['TotalCount'] >= 20)]
        print(f"   Found {len(challenging)} challenging fields")
        
        print("\nModel Strengths/Weaknesses:")
        if len(model_df) > 0:
            best_key_model = model_df.loc[model_df['Key_F1'].idxmax(), 'Model']
            best_val_model = model_df.loc[model_df['Value_F1'].idxmax(), 'Model']
            print(f"   Best Key-level performance: {best_key_model}")
            print(f"   Best Value-level performance: {best_val_model}")
        
        list_fields = [piece for piece, stats in processor.all_stats.items() 
                       if stats.total_count > 100]
        print(f"\nHigh-frequency fields (likely lists/structs): {len(list_fields)}")
        if list_fields:
            print(f"   Examples: {list_fields[:5]}")

if __name__ == "__main__":
    main()