#!/usr/bin/env python3
"""
Script to extract specific modules from PMC cases using LLM inference.
Processes CSV files containing "True" values and generates extracted module data.
"""

import os
import csv
import json
import logging
import yaml
import openai
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModuleExtractor:
    """Handles extraction of clinical trial modules from PMC XML files using LLM."""
    
    def __init__(self, config_path: str = 'extract_config.yaml'):
        """Initialize the extractor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI client for litellm proxy
        self.client = openai.OpenAI(
            base_url=self.config['litellm_api']['base_url'],
            api_key=self.config['litellm_api'].get('api_key', "dummy")
        )
        
        self.module_mapping = self.config['module_prompt_mapping']
        self.prompts_cache = {}
    
    def load_pmc_file(self, case_id: str) -> Optional[str]:
        """Load and parse PMC XML file."""
        pmc_dir = Path(self.config['paths']['pmc_dir'])
        
        # Find the PMC file for this case_id
        pmc_files = list(pmc_dir.glob(f"{case_id}_*.xml"))
        
        if not pmc_files:
            logger.warning(f"No PMC file found for case_id: {case_id}")
            return None
        
        pmc_file = pmc_files[0]
        logger.info(f"Loading PMC file: {pmc_file}")
        
        try:
            with open(pmc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Optionally truncate if too long
            max_len = self.config['processing']['max_pmc_length']
            if len(content) > max_len:
                logger.warning(f"PMC content truncated from {len(content)} to {max_len} chars")
                content = content[:max_len]
            
            return content
        except Exception as e:
            logger.error(f"Error loading PMC file {pmc_file}: {e}")
            return None
    
    def load_prompt(self, prompt_path: str) -> Optional[str]:
        """Load prompt template from file with caching."""
        if prompt_path in self.prompts_cache:
            return self.prompts_cache[prompt_path]
        
        full_path = Path(self.config['paths']['prompts_dir']) / prompt_path
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.prompts_cache[prompt_path] = content
            return content
        except Exception as e:
            logger.error(f"Error loading prompt {full_path}: {e}")
            return None
    
    def extract_module(self, case_id: str, module_name: str, pmc_content: str) -> Optional[Dict]:
        """Extract a specific module using LLM."""
        # Get prompt template
        prompt_path = self.module_mapping.get(module_name)
        if not prompt_path:
            logger.error(f"No prompt mapping found for module: {module_name}")
            return None
        
        prompt_template = self.load_prompt(prompt_path)
        if not prompt_template:
            return None
        
        # Replace placeholder in prompt
        prompt = prompt_template.replace('{{pmc_text}}', pmc_content)
        
        logger.info(f"Extracting {module_name} for case {case_id}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.config['litellm_api']['model'],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=self.config['litellm_api']['timeout'],
                temperature=0
            )
            
            content = response.choices[0].message.content
            logger.info(f"Received response for {module_name} ({response.usage.total_tokens} tokens)")
            
            # Parse JSON from response (handle markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            # Some prompts return multiple modules (e.g., description_and_conditions)
            # Extract only the requested module
            if module_name in result:
                return {module_name: result[module_name]}
            else:
                # If the module is in the nested structure, extract it
                for key, value in result.items():
                    if key == module_name:
                        return {module_name: value}
                
                # Return full result if we can't isolate the module
                return result
            
        except Exception as e:
            logger.error(f"Error extracting {module_name} for case {case_id}: {e}")
            return None
    
    def process_csv(self):
        """Main processing function to read input CSV and generate output CSV."""
        input_path = Path(self.config['paths']['input_csv'])
        output_path = Path(self.config['paths']['output_csv'])
        
        # Add timestamp to output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_stem = output_path.stem
        output_suffix = output_path.suffix
        output_path = output_path.parent / f"{output_stem}_{timestamp}{output_suffix}"
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input CSV
        logger.info(f"Reading input CSV: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
        
        logger.info(f"Found {len(rows)} rows in input CSV")
        
        # Initialize output rows with empty values
        output_rows = []
        for row in rows:
            case_id = row['case_id']
            output_row = {'case_id': case_id}
            
            # Initialize all columns with empty strings
            for col in fieldnames:
                if col != 'case_id':
                    output_row[col] = ''
            
            # Check if this row needs processing
            needs_processing = any(row.get(col, '').strip() == 'True' for col in fieldnames if col != 'case_id')
            if needs_processing:
                output_rows.append(output_row)
        
        # Create output CSV file with headers immediately
        logger.info(f"Creating output CSV: {output_path}")
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        
        # Process each row and update CSV after each module extraction
        row_index = 0
        for row in rows:
            case_id = row['case_id']
            
            # Load PMC content once per case
            pmc_content = None
            
            # Check if this row needs processing
            needs_processing = False
            for module_name in fieldnames:
                if module_name == 'case_id':
                    continue
                
                # Check if this cell has "True"
                if row[module_name].strip() == 'True':
                    needs_processing = True
                    break
            
            if not needs_processing:
                continue
            
            # Process each module for this case
            for module_name in fieldnames:
                if module_name == 'case_id':
                    continue
                
                # Check if this cell has "True"
                if row[module_name].strip() == 'True':
                    # Load PMC content if not already loaded
                    if pmc_content is None:
                        pmc_content = self.load_pmc_file(case_id)
                        if pmc_content is None:
                            logger.warning(f"Skipping case {case_id} - no PMC content")
                            break
                    
                    # Extract the module
                    result = self.extract_module(case_id, module_name, pmc_content)
                    
                    if result:
                        # Convert result to JSON string for CSV
                        output_rows[row_index][module_name] = json.dumps(result.get(module_name, result), ensure_ascii=False)
                    else:
                        output_rows[row_index][module_name] = ''
                    
                    # Update CSV file after each module extraction
                    with open(output_path, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(output_rows)
                    
                    logger.info(f"Updated CSV - Case {case_id}, Module {module_name}")
            
            row_index += 1
            logger.info(f"Completed case {case_id}")
        
        logger.info(f"Completed! Processed {row_index} cases")
        logger.info(f"Output saved to: {output_path}")


def main():
    """Main entry point."""
    extractor = ModuleExtractor()
    extractor.process_csv()


if __name__ == "__main__":
    main()
