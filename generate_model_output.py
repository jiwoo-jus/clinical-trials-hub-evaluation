import os
import re
import json
import time
import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import openai

load_dotenv()

PROCESS_MODE = "range"
START_CASE = 1
END_CASE = 2
SPECIFIC_CASES = []

MAX_WORKERS = 3
MAX_CASES = 100

BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
SAMPLE_IDS_FILE = BASE_DIR / "DATA" / "sample_ids.txt"
PROMPT_TEMPLATE_DIR = BASE_DIR / "PROMPT_TEMPLATE"
OUTPUT_DIR = BASE_DIR / "DATA" / "PROMPT"
MODEL_OUTPUT_DIR = BASE_DIR / "DATA" / "MODEL_OUTPUT"

MODELS = {
    "gpt-4o": "GPT-4o",
    "claude-4-sonnet": "Anthropic Claude 4 Sonnet",
    "llama-70b": "Meta Llama 3.3 70B Instruct"
}

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_TOOL_NAME = os.getenv("NCBI_TOOL_NAME")

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """API request rate limiter"""
    def __init__(self, requests_per_second: int):
        self.requests_per_second = requests_per_second
        self.request_times = []
    
    def wait_if_needed(self):
        """Wait if necessary"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        
        if len(self.request_times) >= self.requests_per_second:
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_times = []
        
        self.request_times.append(now)

ncbi_limiter = RateLimiter(10)
litellm_limiter = RateLimiter(3)

class NCBIFetcher:
    """Fetch PMC XML from NCBI"""
    
    def __init__(self):
        self.base_params = {
            "tool": "ClinicalTrialsHub",
            "email": NCBI_EMAIL,
            "api_key": NCBI_API_KEY
        }
    
    def fetch_pmc_xml(self, pmcid: str) -> Optional[str]:
        """Fetch XML by PMC ID"""
        if pmcid.startswith("PMC"):
            pmcid = pmcid[3:]
        
        ncbi_limiter.wait_if_needed()
        
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            **self.base_params,
            "db": "pmc",
            "id": pmcid,
            "retmode": "xml"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if response.text.strip():
                return response.text
            else:
                logger.warning(f"Empty response for PMC{pmcid}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching PMC{pmcid}: {e}")
            return None
    
    def extract_text_from_xml(self, xml_content: str) -> str:
        """Extract text from XML"""
        try:
            root = ET.fromstring(xml_content)
            
            text_parts = []
            
            front = root.find(".//front")
            if front is not None:
                title = front.find(".//article-title")
                if title is not None and title.text:
                    text_parts.append(f"Title: {' '.join(title.itertext())}")
                
                abstract = front.find(".//abstract")
                if abstract is not None:
                    abstract_text = ' '.join(abstract.itertext())
                    if abstract_text:
                        text_parts.append(f"Abstract: {abstract_text}")
            
            # Body
            body = root.find(".//body")
            if body is not None:
                body_text = ' '.join(body.itertext())
                if body_text:
                    text_parts.append(f"Body: {body_text}")
            
            # Back matter (references, etc)
            back = root.find(".//back")
            if back is not None:
                back_text = ' '.join(back.itertext())
                if back_text:
                    text_parts.append(f"References and Supplementary: {back_text}")
            
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            return ""

class PromptGenerator:
    """Prompt generator"""
    
    def __init__(self):
        self.template_groups = {
            "1_protocol_section": [
                "1_identification.md",
                "2_description_and_conditions.md",
                "3_design.md",
                "4_arms_interventions.md",
                "5_outcomes.md",
                "6_eligibility.md"
            ]
            # ,"2_results_section": [
            #     "1_participantflow.md",
            #     "2_baselinecharacteristics.md",
            #     "3_outcomemeasures.md",
            #     "4_adverse_events.md",
            #     "5_more_info.md"
            # ]
            # ,"3_derived_section": [
            #     "1_conditionbrowse_interventionbrowse.md"
            # ]
        }
        
        logger.info(f"Checking template directory: {PROMPT_TEMPLATE_DIR}")
        if not PROMPT_TEMPLATE_DIR.exists():
            logger.error(f"Template directory does not exist: {PROMPT_TEMPLATE_DIR}")
        else:
            for group, templates in self.template_groups.items():
                group_dir = PROMPT_TEMPLATE_DIR / group
                if not group_dir.exists():
                    logger.error(f"Template group directory missing: {group_dir}")
                else:
                    for template in templates:
                        template_path = group_dir / template
                        if not template_path.exists():
                            logger.error(f"Template file missing: {template_path}")
                        else:
                            logger.info(f"Template found: {template_path}")
    
    def load_template(self, template_path: Path) -> str:
        """Load template file"""
        try:
            logger.debug(f"Loading template from: {template_path}")
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Template loaded successfully, length: {len(content)} chars")
                return content
        except FileNotFoundError:
            logger.error(f"Template file not found: {template_path}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Absolute path attempted: {template_path.absolute()}")
            return ""
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {type(e).__name__}: {e}")
            return ""
    
    def generate_prompts(self, pmc_text: str, case_info: Dict[str, str]) -> Dict[str, str]:
        """Generate prompts for all templates"""
        prompts = {}
        
        logger.info(f"Generating prompts for case {case_info['case_no']}")
        logger.debug(f"PMC text length: {len(pmc_text)} chars")
        
        for group, templates in self.template_groups.items():
            for template_name in templates:
                template_path = PROMPT_TEMPLATE_DIR / group / template_name
                
                logger.debug(f"Processing template: {template_path}")
                
                if not template_path.exists():
                    logger.warning(f"Template not found: {template_path}")
                    logger.warning(f"Expected location: {template_path.absolute()}")
                    continue
                
                template = self.load_template(template_path)
                
                if not template:
                    logger.error(f"Empty template loaded from {template_path}")
                    continue
                
                prompt = template.replace("{{pmc_text}}", pmc_text)
                
                if "{{pmc_text}}" in prompt:
                    logger.warning(f"Variable substitution failed for {template_path}")
                
                metadata = f"""
# Metadata
- Case No: {case_info['case_no']}
- NCT ID: {case_info['nct_id']}
- PMID: {case_info['pmid']}
- PMCID: {case_info['pmcid']}
- Timestamp: {datetime.now().isoformat()}

"""
                prompt = metadata + prompt
                
                key = f"{group}/{template_name.replace('.md', '')}"
                prompts[key] = prompt
                
                logger.debug(f"Generated prompt for {key}, length: {len(prompt)} chars")
        
        logger.info(f"Generated {len(prompts)} prompts for case {case_info['case_no']}")
        return prompts


class ModelCaller:
    """Model calling through LiteLLM"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=LITELLM_API_KEY,
            base_url=LITELLM_BASE_URL
        )
    
    async def call_model_async(self, model_key: str, prompt: str, case_info: Dict[str, str], 
                              prompt_key: str) -> Dict[str, Any]:
        """Asynchronous model call (save results immediately)"""
        model_name = MODELS[model_key]
        
        try:
            litellm_limiter.wait_if_needed()
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant trained to extract structured data in JSON format from clinical trial articles."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4096
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"raw_response": content, "error": "JSON parsing failed"}
            
            output = {
                "model": model_key,
                "prompt_key": prompt_key,
                "case_info": case_info,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            pipeline = Pipeline()
            pipeline.save_incremental_result(case_info['case_no'], model_key, prompt_key, output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error calling {model_key} for {prompt_key}: {e}")
            error_output = {
                "model": model_key,
                "prompt_key": prompt_key,
                "case_info": case_info,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            pipeline = Pipeline()
            pipeline.save_incremental_result(case_info['case_no'], model_key, prompt_key, error_output)
            
            return error_output
    
    def call_model_sync(self, model_key: str, prompt: str, case_info: Dict[str, str], 
                       prompt_key: str, pipeline_instance=None) -> Dict[str, Any]:
        """Synchronous model call (save results immediately)"""
        model_name = MODELS[model_key]
        
        logger.debug(f"Calling {model_key} for {prompt_key}")
        logger.debug(f"Prompt length: {len(prompt)} chars")
        logger.debug(f"First 200 chars of prompt: {prompt[:200]}...")
        
        try:
            litellm_limiter.wait_if_needed()
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant trained to extract structured data in JSON format from clinical trial articles."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4096
            )
            
            content = response.choices[0].message.content
            
            if content is None:
                logger.error(f"Received None response from {model_key}")
                content = "{}"
            
            logger.debug(f"Response length: {len(content)} chars")
            
            try:
                result = json.loads(content)
                logger.debug(f"JSON parsed successfully, keys: {list(result.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.debug(f"Raw response: {content[:500]}...")
                result = {"raw_response": content, "error": f"JSON parsing failed: {str(e)}"}
            
            output = {
                "model": model_key,
                "prompt_key": prompt_key,
                "case_info": case_info,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            if pipeline_instance:
                pipeline_instance.save_incremental_result(case_info['case_no'], model_key, prompt_key, output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error calling {model_key} for {prompt_key}: {type(e).__name__}: {e}")
            if hasattr(e, 'response'):
                logger.error(f"API response: {e.response}")
            
            error_output = {
                "model": model_key,
                "prompt_key": prompt_key,
                "case_info": case_info,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            
            if pipeline_instance:
                pipeline_instance.save_incremental_result(case_info['case_no'], model_key, prompt_key, error_output)
            
            return error_output

class Pipeline:
    """Complete pipeline"""
    
    def __init__(self):
        self.ncbi_fetcher = NCBIFetcher()
        self.prompt_generator = PromptGenerator()
        self.model_caller = ModelCaller()
        
        logger.info("=== Pipeline Initialization ===")
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Template directory: {PROMPT_TEMPLATE_DIR}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info(f"Model output directory: {MODEL_OUTPUT_DIR}")
        logger.info(f"Sample IDs file: {SAMPLE_IDS_FILE}")
        
        if not BASE_DIR.exists():
            logger.error(f"Base directory does not exist: {BASE_DIR}")
        
        if not PROMPT_TEMPLATE_DIR.exists():
            logger.error(f"Template directory does not exist: {PROMPT_TEMPLATE_DIR}")
            logger.info("Expected template structure:")
            for group in ["1_protocol_section"]:
                logger.info(f"  {PROMPT_TEMPLATE_DIR / group}/")
        
        if not Path(SAMPLE_IDS_FILE).exists():
            logger.error(f"Sample IDs file not found: {SAMPLE_IDS_FILE}")
        
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            
        try:
            MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model output directory ready: {MODEL_OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create model output directory: {e}")
        
        if not LITELLM_API_KEY:
            logger.error("LiteLLM API key not found in environment variables")
        else:
            logger.info("LiteLLM API key found")
            
        logger.info("=== Initialization Complete ===\n")
    
    def load_sample_ids(self, limit: int = 100, start_case: int = None, end_case: int = None) -> List[Dict[str, str]]:
        """Load sample IDs (can specify case range)"""
        samples = []
        
        pattern = re.compile(r'\[(\d+)\]\s*NCTID:\s*(NCT\d+),\s*PMID:\s*(\d+),\s*PMCID:\s*(\d+)')
        
        with open(SAMPLE_IDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    case_no = int(match.group(1))
                    
                    if start_case and case_no < start_case:
                        continue
                    if end_case and case_no > end_case:
                        continue
                    
                    samples.append({
                        "case_no": str(case_no),
                        "nct_id": match.group(2),
                        "pmid": match.group(3),
                        "pmcid": match.group(4)
                    })
                    
                    if len(samples) >= limit:
                        break
        
        if start_case or end_case:
            logger.info(f"Loaded {len(samples)} samples from case {start_case or 'start'} to {end_case or 'end'}")
        else:
            logger.info(f"Loaded {len(samples)} samples (limit: {limit})")
            
        return samples
    
    def save_prompts(self, case_info: Dict[str, str], prompts: Dict[str, str]):
        """Save prompts"""
        case_dir = OUTPUT_DIR / f"case_{case_info['case_no']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        for prompt_key, prompt_content in prompts.items():
            parts = prompt_key.split('/')
            filename = f"{parts[-1]}.md"
            
            filepath = case_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(prompt_content)
        
        metadata_file = case_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(case_info, f, indent=2)
    
    def merge_module_results(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge module results to create final structure"""
        merged = {}
        
        for module_name, module_data in module_results.items():
            if isinstance(module_data, dict):
                for key, value in module_data.items():
                    merged[key] = value
        
        return merged
    
    def save_incremental_result(self, case_no: str, model_key: str, prompt_key: str, result: Dict[str, Any]):
        """Save single result immediately (incremental save)"""
        case_dir = MODEL_OUTPUT_DIR / f"case_{case_no}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = case_dir / f"{model_key}_results.json"
        
        logger.debug(f"Saving result to: {output_file}")
        
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    ctg_output = json.load(f)
                logger.debug(f"Loaded existing file with sections: {list(ctg_output.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in existing file {output_file}: {e}")
                ctg_output = {"protocolSection": {}}
            except Exception as e:
                logger.error(f"Error reading existing file {output_file}: {type(e).__name__}: {e}")
                ctg_output = {"protocolSection": {}}
        else:
            ctg_output = {"protocolSection": {}}
            logger.debug("Creating new output file")
        
        module_result = result.get('result', {})
        
        logger.debug(f"Module result keys: {list(module_result.keys()) if isinstance(module_result, dict) else 'Not a dict'}")
        
        if result.get('error'):
            logger.error(f"Error result for {prompt_key}: {result['error']}")
            error_file = case_dir / f"{model_key}_errors.json"
            errors = []
            if error_file.exists():
                try:
                    with open(error_file, 'r', encoding='utf-8') as f:
                        errors = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading error file: {e}")
            
            errors.append({
                "prompt_key": prompt_key,
                "error": result['error'],
                "timestamp": result.get('timestamp'),
                "full_result": result
            })
            
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(errors, f, indent=2, ensure_ascii=False)
                logger.debug(f"Error saved to {error_file}")
            except Exception as e:
                logger.error(f"Failed to save error file: {e}")
            return
        
        updated = False
        
        if prompt_key.startswith("1_protocol_section"):
            if "1_identification" in prompt_key:
                if "identificationModule" in module_result:
                    ctg_output["protocolSection"]["identificationModule"] = module_result["identificationModule"]
                    updated = True
                elif "nctId" in module_result:
                    ctg_output["protocolSection"]["identificationModule"] = module_result
                    updated = True
                    logger.debug(f"Wrapped bare identification data in identificationModule")
                else:
                    logger.warning(f"Expected 'identificationModule' or identification fields not found in result for {prompt_key}")
                    logger.debug(f"Available keys: {list(module_result.keys())}")
            
            elif "2_description_and_conditions" in prompt_key:
                if "descriptionModule" in module_result:
                    ctg_output["protocolSection"]["descriptionModule"] = module_result["descriptionModule"]
                    updated = True
                elif "briefSummary" in module_result or "detailedDescription" in module_result:
                    ctg_output["protocolSection"]["descriptionModule"] = {
                        k: v for k, v in module_result.items() 
                        if k in ["briefSummary", "detailedDescription"]
                    }
                    updated = True
                    
                if "conditionsModule" in module_result:
                    ctg_output["protocolSection"]["conditionsModule"] = module_result["conditionsModule"]
                    updated = True
                elif "conditions" in module_result or "keywords" in module_result:
                    ctg_output["protocolSection"]["conditionsModule"] = {
                        k: v for k, v in module_result.items() 
                        if k in ["conditions", "keywords"]
                    }
                    updated = True
                    
                if not updated:
                    logger.warning(f"Expected modules not found in result for {prompt_key}")
                    logger.debug(f"Available keys: {list(module_result.keys())}")
            
            elif "3_design" in prompt_key:
                if "designModule" in module_result:
                    ctg_output["protocolSection"]["designModule"] = module_result["designModule"]
                    updated = True
                elif "studyType" in module_result or "phases" in module_result or "designInfo" in module_result:
                    ctg_output["protocolSection"]["designModule"] = module_result
                    updated = True
                    logger.debug(f"Wrapped bare design data in designModule")
                else:
                    logger.warning(f"Expected 'designModule' or design fields not found in result for {prompt_key}")
            
            elif "4_arms_interventions" in prompt_key:
                if "armsInterventionsModule" in module_result:
                    ctg_output["protocolSection"]["armsInterventionsModule"] = module_result["armsInterventionsModule"]
                    updated = True
                elif "armGroups" in module_result or "interventions" in module_result:
                    ctg_output["protocolSection"]["armsInterventionsModule"] = module_result
                    updated = True
                    logger.debug(f"Wrapped bare arms/interventions data in armsInterventionsModule")
                else:
                    logger.warning(f"Expected 'armsInterventionsModule' or arms/interventions fields not found in result for {prompt_key}")
            
            elif "5_outcomes" in prompt_key:
                if "outcomesModule" in module_result:
                    ctg_output["protocolSection"]["outcomesModule"] = module_result["outcomesModule"]
                    updated = True
                elif "primaryOutcomes" in module_result or "secondaryOutcomes" in module_result:
                    ctg_output["protocolSection"]["outcomesModule"] = module_result
                    updated = True
                    logger.debug(f"Wrapped bare outcomes data in outcomesModule")
                else:
                    logger.warning(f"Expected 'outcomesModule' or outcomes fields not found in result for {prompt_key}")
            
            elif "6_eligibility" in prompt_key:
                if "eligibilityModule" in module_result:
                    ctg_output["protocolSection"]["eligibilityModule"] = module_result["eligibilityModule"]
                    updated = True
                elif "eligibilityCriteria" in module_result or "sex" in module_result or "minimumAge" in module_result:
                    ctg_output["protocolSection"]["eligibilityModule"] = module_result
                    updated = True
                    logger.debug(f"Wrapped bare eligibility data in eligibilityModule")
                else:
                    logger.warning(f"Expected 'eligibilityModule' or eligibility fields not found in result for {prompt_key}")
        
        if not updated:
            logger.warning(f"No updates made for {prompt_key}")
            
            unexpected_file = case_dir / f"{model_key}_unexpected.json"
            unexpected_results = []
            if unexpected_file.exists():
                try:
                    with open(unexpected_file, 'r', encoding='utf-8') as f:
                        unexpected_results = json.load(f)
                except:
                    pass
            
            unexpected_results.append({
                "prompt_key": prompt_key,
                "result": module_result,
                "timestamp": result.get('timestamp')
            })
            
            with open(unexpected_file, 'w', encoding='utf-8') as f:
                json.dump(unexpected_results, f, indent=2, ensure_ascii=False)
        
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ctg_output, f, indent=4, ensure_ascii=False)
            logger.info(f"Incrementally saved {model_key} result for case {case_no}, module {prompt_key}")
        except Exception as e:
            logger.error(f"Failed to save output file {output_file}: {type(e).__name__}: {e}")
    
    async def process_case_async(self, case_info: Dict[str, str]) -> List[Dict[str, Any]]:
        logger.info(f"Processing case {case_info['case_no']}: PMCID={case_info['pmcid']}")
        
        xml_content = self.ncbi_fetcher.fetch_pmc_xml(case_info['pmcid'])
        if not xml_content:
            logger.error(f"Failed to fetch PMC XML for case {case_info['case_no']}")
            return []
        
        pmc_text = self.ncbi_fetcher.extract_text_from_xml(xml_content)
        if not pmc_text:
            logger.error(f"Failed to extract text from PMC XML for case {case_info['case_no']}")
            return []
        
        prompts = self.prompt_generator.generate_prompts(pmc_text, case_info)
        
        self.save_prompts(case_info, prompts)
        
        results = []
        
        
        section_order = [
            ("1_protocol_section", [
                "1_identification",
                "2_description_and_conditions", 
                "3_design",
                "4_arms_interventions",
                "5_outcomes",
                "6_eligibility"
            ])
        ]
        
        
        for section, modules in section_order:
            for module in modules:
                prompt_key = f"{section}/{module}"
                if prompt_key in prompts:
                    prompt_content = prompts[prompt_key]
                    
                    
                    model_tasks = []
                    for model_key in MODELS:
                        task = self.model_caller.call_model_async(
                            model_key, prompt_content, case_info, prompt_key
                        )
                        model_tasks.append(task)
                    
                    
                    module_results = await asyncio.gather(*model_tasks, return_exceptions=True)
                    
                    for result in module_results:
                        if not isinstance(result, Exception):
                            results.append(result)
                        else:
                            logger.error(f"Task exception: {result}")
        
        return results
    
    def run_parallel(self, max_workers: int = 3, limit: int = 100, start_case: int = None, end_case: int = None):
        """병렬 처리 실행 (증분 저장, 케이스 범위 지정 가능)"""
        
        samples = self.load_sample_ids(limit=limit, start_case=start_case, end_case=end_case)
        
        
        completed_cases = self.check_completed_cases(samples)
        remaining_samples = [s for s in samples if s['case_no'] not in completed_cases]
        
        logger.info(f"Total samples: {len(samples)}, Completed: {len(completed_cases)}, Remaining: {len(remaining_samples)}")
        logger.info(f"Starting parallel processing with {max_workers} workers")
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            future_to_case = {}
            
            for case_info in remaining_samples:
                future = executor.submit(self.process_case_sync, case_info)
                future_to_case[future] = case_info
            
            
            for future in as_completed(future_to_case):
                case_info = future_to_case[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Completed case {case_info['case_no']}")
                except Exception as e:
                    logger.error(f"Error processing case {case_info['case_no']}: {e}")
        
        logger.info(f"Pipeline completed. Total results: {len(all_results)}")
    
    def check_completed_cases(self, samples: List[Dict[str, str]]) -> Set[str]:
        completed = set()
        
        for sample in samples:
            case_no = sample['case_no']
            progress_file = MODEL_OUTPUT_DIR / f"case_{case_no}" / "progress.json"
            
            if progress_file.exists():
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress = json.load(f)
                    if progress.get('status') == 'completed':
                        completed.add(case_no)
                except Exception as e:
                    logger.warning(f"Error reading progress file for case {case_no}: {e}")
        
        return completed
    
    def process_case_sync(self, case_info: Dict[str, str]) -> List[Dict[str, Any]]:
        logger.info(f"Processing case {case_info['case_no']}: PMCID={case_info['pmcid']}")
        
        
        progress_file = MODEL_OUTPUT_DIR / f"case_{case_info['case_no']}" / "progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        progress = {
            "case_info": case_info,
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
            "completed_modules": []
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
        
        xml_content = self.ncbi_fetcher.fetch_pmc_xml(case_info['pmcid'])
        if not xml_content:
            logger.error(f"Failed to fetch PMC XML for case {case_info['case_no']}")
            progress["status"] = "failed"
            progress["error"] = "Failed to fetch PMC XML"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
            return []
        
        pmc_text = self.ncbi_fetcher.extract_text_from_xml(xml_content)
        if not pmc_text:
            logger.error(f"Failed to extract text from PMC XML for case {case_info['case_no']}")
            progress["status"] = "failed"
            progress["error"] = "Failed to extract text from PMC XML"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
            return []
        
        prompts = self.prompt_generator.generate_prompts(pmc_text, case_info)
        
        self.save_prompts(case_info, prompts)
        
        results = []
        
        
        section_order = [
            ("1_protocol_section", [
                "1_identification",
                "2_description_and_conditions", 
                "3_design",
                "4_arms_interventions",
                "5_outcomes",
                "6_eligibility"
            ])
        ]
        
        
        for section, modules in section_order:
            for module in modules:
                prompt_key = f"{section}/{module}"
                if prompt_key in prompts:
                    prompt_content = prompts[prompt_key]
                    
                    
                    for model_key in MODELS:
                        result = self.model_caller.call_model_sync(
                            model_key, prompt_content, case_info, prompt_key, 
                            pipeline_instance=self
                        )
                        results.append(result)
                    
                    
                    progress["completed_modules"].append(prompt_key)
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress, f, indent=2)
        
        
        progress["status"] = "completed"
        progress["end_time"] = datetime.now().isoformat()
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
        
        return results
    
    async def run_async(self):
        """비동기 실행 (선택적)"""
        
        samples = self.load_sample_ids(limit=100)
        
        logger.info("Starting async processing")
        
        
        tasks = []
        for case_info in samples:
            task = self.process_case_async(case_info)
            tasks.append(task)
        
        
        all_results = []
        for i in range(0, len(tasks), 5):
            batch = tasks[i:i+5]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for results in batch_results:
                if isinstance(results, Exception):
                    logger.error(f"Batch exception: {results}")
                else:
                    all_results.extend(results)
        
        
        self.save_model_output(all_results)
        
        logger.info(f"Pipeline completed. Total results: {len(all_results)}")

def main():
    """메인 함수"""
    pipeline = Pipeline()
    
    logger.info(f"Process mode: {PROCESS_MODE}")
    
    if PROCESS_MODE == "specific":
        
        logger.info(f"Processing specific cases: {SPECIFIC_CASES}")
        
        for case_no in sorted(SPECIFIC_CASES):
            samples = pipeline.load_sample_ids(limit=1, start_case=case_no, end_case=case_no)
            if samples:
                logger.info(f"\n=== Processing case {case_no} ===")
                pipeline.process_case_sync(samples[0])
            else:
                logger.warning(f"Case {case_no} not found in sample file")
                
    elif PROCESS_MODE == "range":
        
        logger.info(f"Processing cases from {START_CASE} to {END_CASE}")
        pipeline.run_parallel(
            max_workers=MAX_WORKERS,
            limit=MAX_CASES,
            start_case=START_CASE,
            end_case=END_CASE
        )
    else:
        
        logger.info(f"Processing all cases (limit: {MAX_CASES})")
        pipeline.run_parallel(
            max_workers=MAX_WORKERS,
            limit=MAX_CASES
        )

if __name__ == "__main__":
    main()