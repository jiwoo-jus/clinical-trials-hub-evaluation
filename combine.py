"""
Module Extraction and Restructuring Pipeline

1. Extract modules from existing JSON files
2. Save as CSV (one file per model, columns = modules)
3. Restructure JSON files according to config specification
"""

import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class ModuleExtractor:
    """Extract modules from JSON files and save to CSV"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = config['input']['models']
        self.input_dir = Path(config['input']['model_outputs_dir'])
        self.csv_dir = Path(config['output']['csv_dir'])
        self.modules = config['modules']
        self.module_overrides = config.get('module_overrides', {})
        
        # Create output directory
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Initialized ModuleExtractor")
        print(f"   Input: {self.input_dir}")
        print(f"   CSV Output: {self.csv_dir}")
        print(f"   Models: {self.models}")
        print(f"   Modules to extract: {len(self.modules)}")
        if self.module_overrides:
            print(f"   Module overrides configured: {len(self.module_overrides)} model(s)")
    
    def find_module_in_json(self, data: Dict[str, Any], module_name: str) -> Optional[Dict[str, Any]]:
        """Recursively search for a module in JSON structure"""
        if not isinstance(data, dict):
            return None
        
        # Direct match
        if module_name in data:
            return data[module_name]
        
        # Search in nested structures
        for key, value in data.items():
            if isinstance(value, dict):
                result = self.find_module_in_json(value, module_name)
                if result is not None:
                    return result
        
        return None
    
    def extract_modules_from_file(self, json_path: Path) -> Dict[str, Any]:
        """Extract all modules from a single JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract case_id from filename (e.g., "42_GPT-5.1.json" -> 42)
            case_id = json_path.stem.split('_')[0]
            
            modules_data = {'case_id': case_id}
            
            # Extract each module
            for module_name in self.modules:
                module_content = self.find_module_in_json(data, module_name)
                # Store as JSON string for CSV
                if module_content is not None:
                    modules_data[module_name] = json.dumps(module_content, ensure_ascii=False)
                else:
                    modules_data[module_name] = None
            
            return modules_data
        
        except Exception as e:
            print(f"❌ Error extracting from {json_path}: {e}")
            return None
    
    def load_override_data(self, model_name: str) -> Dict[str, Dict[str, str]]:
        """Load override CSV data for a model"""
        override_data = {}
        
        if model_name not in self.module_overrides:
            return override_data
        
        model_overrides = self.module_overrides[model_name]
        
        for module_name, csv_path_str in model_overrides.items():
            csv_path = Path(csv_path_str)
            if not csv_path.exists():
                print(f"   ⚠️  Override CSV not found: {csv_path}")
                continue
            
            print(f"   📝 Loading override for {module_name} from {csv_path.name}")
            
            # Read override CSV
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Index by case_id
            override_data[module_name] = {}
            for row in rows:
                if 'case_id' in row and module_name in row and row[module_name]:
                    override_data[module_name][row['case_id']] = row[module_name]
            
            print(f"      Loaded {len(override_data[module_name])} override(s)")
        
        return override_data
    
    def extract_model_to_csv(self, model_name: str):
        """Extract all cases for one model to CSV"""
        print(f"\n📊 Processing model: {model_name}")
        
        # Find all JSON files for this model
        model_dir = self.input_dir / model_name
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return
        
        json_files = sorted(model_dir.glob("*_*.json"))
        # Filter out timing files
        json_files = [f for f in json_files if not f.stem.endswith('_timing')]
        
        print(f"   Found {len(json_files)} JSON files")
        
        if not json_files:
            print(f"   No files to process")
            return
        
        # Load override data
        override_data = self.load_override_data(model_name)
        
        # Extract modules from all files
        all_rows = []
        for json_file in json_files:
            modules_data = self.extract_modules_from_file(json_file)
            if modules_data:
                # Apply overrides
                case_id = modules_data['case_id']
                for module_name in self.modules:
                    if module_name in override_data and case_id in override_data[module_name]:
                        # Override with data from override CSV
                        modules_data[module_name] = override_data[module_name][case_id]
                        
                all_rows.append(modules_data)
        
        if not all_rows:
            print(f"   No data extracted")
            return
        
        # Sort by case_id
        all_rows.sort(key=lambda x: int(x['case_id']))
        
        # Write to CSV
        csv_path = self.csv_dir / f"{model_name}.csv"
        fieldnames = ['case_id'] + self.modules
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"   ✅ Saved to {csv_path}")
        print(f"   Total cases: {len(all_rows)}")
        if override_data:
            print(f"   ✨ Overrides applied from custom CSV(s)")
    
    def run(self):
        """Extract modules for all models"""
        print("="*80)
        print("MODULE EXTRACTION TO CSV")
        print("="*80)
        
        for model_name in self.models:
            self.extract_model_to_csv(model_name)
        
        print("\n" + "="*80)
        print("✅ Module extraction completed")
        print("="*80)


class ModuleRestructurer:
    """Restructure JSON files from CSV according to config"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = config['input']['models']
        self.csv_dir = Path(config['output']['csv_dir'])
        self.output_dir = Path(config['output']['restructured_dir'])
        self.structure = config['output_structure']
        self.modules = config['modules']
        self.include_metadata = config['options'].get('include_metadata', True)
        self.module_overrides = config.get('module_overrides', {})
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for model_name in self.models:
            (self.output_dir / model_name).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Initialized ModuleRestructurer")
        print(f"   CSV Input: {self.csv_dir}")
        print(f"   JSON Output: {self.output_dir}")
        print(f"   Include metadata: {self.include_metadata}")
        if self.module_overrides:
            print(f"   Module overrides configured: {len(self.module_overrides)} model(s)")
    
    def restructure_from_csv_row(self, row: Dict[str, str], override_data: Dict[str, Dict[str, str]] = None) -> Dict[str, Any]:
        """Build structured JSON from CSV row with optional module overrides"""
        result = {}
        case_id = row.get('case_id', 'unknown')
        
        if override_data is None:
            override_data = {}
        
        # Build each section according to config
        for section_name, module_list in self.structure.items():
            section_data = {}
            
            for module_name in module_list:
                # Check for override first
                module_value = None
                if module_name in override_data and case_id in override_data[module_name]:
                    module_value = override_data[module_name][case_id]
                    # Use override value
                elif module_name in row and row[module_name]:
                    module_value = row[module_name]
                
                if module_value:
                    try:
                        # Parse JSON string from CSV
                        module_content = json.loads(module_value)
                        section_data[module_name] = module_content
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error for {module_name} in case {case_id}: {e}")
                        section_data[module_name] = None
                else:
                    section_data[module_name] = None
            
            result[section_name] = section_data
        
        return result
    
    def load_override_csvs(self, model_name: str) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load module override CSVs for a specific model"""
        override_data = {}
        
        if model_name not in self.module_overrides:
            return override_data
        
        model_overrides = self.module_overrides[model_name]
        
        for module_name, csv_path_str in model_overrides.items():
            csv_path = Path(csv_path_str)
            if not csv_path.exists():
                print(f"   ⚠️  Override CSV not found: {csv_path}")
                continue
            
            print(f"   📝 Loading override for {module_name} from {csv_path.name}")
            
            # Read override CSV
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Index by case_id
            override_data[module_name] = {}
            for row in rows:
                if 'case_id' in row and module_name in row:
                    override_data[module_name][row['case_id']] = row[module_name]
            
            print(f"      Loaded {len(override_data[module_name])} override(s)")
        
        return override_data
    
    def restructure_model(self, model_name: str):
        """Restructure all cases for one model"""
        print(f"\n🔨 Restructuring model: {model_name}")
        
        csv_path = self.csv_dir / f"{model_name}.csv"
        if not csv_path.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"   Found {len(rows)} cases")
        
        # Load override CSVs
        override_data = self.load_override_csvs(model_name)
        
        # Restructure each case
        output_model_dir = self.output_dir / model_name
        
        for row in rows:
            case_id = row['case_id']
            
            # Build structured JSON (with overrides if available)
            structured_data = self.restructure_from_csv_row(row, override_data)
            
            # Add metadata if requested
            if self.include_metadata:
                structured_data['_metadata'] = {
                    'case_id': case_id,
                    'model_name': model_name,
                    'restructured': True,
                    'source': 'module_csv'
                }
            
            # Save to JSON
            output_path = output_model_dir / f"{case_id}_{model_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=self.config['options']['json_indent'])
        
        print(f"   ✅ Restructured {len(rows)} files to {output_model_dir}")
    
    def run(self):
        """Restructure all models"""
        print("="*80)
        print("JSON RESTRUCTURING FROM CSV")
        print("="*80)
        
        for model_name in self.models:
            self.restructure_model(model_name)
        
        print("\n" + "="*80)
        print("✅ Restructuring completed")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract modules to CSV and restructure JSON files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="combine_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['extract', 'restructure', 'both'],
        default='both',
        help="Operation mode: extract (to CSV), restructure (from CSV), or both"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("MODULE EXTRACTION AND RESTRUCTURING PIPELINE")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Mode: {args.mode}")
    print("="*80)
    
    # Run extraction
    if args.mode in ['extract', 'both']:
        extractor = ModuleExtractor(config)
        extractor.run()
    
    # Run restructuring
    if args.mode in ['restructure', 'both']:
        restructurer = ModuleRestructurer(config)
        restructurer.run()
    
    print("\n" + "="*80)
    print("✅ Pipeline completed successfully")
    print("="*80)


if __name__ == "__main__":
    main()
