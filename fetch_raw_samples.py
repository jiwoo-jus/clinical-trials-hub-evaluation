#!/usr/bin/env python3

import os
import re
import json
import time
import argparse
import io
import logging
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import threading
from datetime import datetime

# -----------------------------
# Helpers and config
# -----------------------------
@dataclass
class SampleRecord:
    index: int
    nct_id: str
    pmid: str
    pmcid: str  # normalized to start with 'PMC'

    @classmethod
    def from_line(cls, line: str) -> Optional["SampleRecord"]:
        try:
            pattern = r'\[(\d+)\]\s+NCTID:\s*([^,]+),\s*PMID:\s*(\d+),\s*PMCID:\s*(\d+)'
            m = re.match(pattern, line.strip())
            if not m:
                return None
            index, nct_id, pmid, pmcid = m.groups()
            nct_id = nct_id.strip()
            if not re.match(r'^NCT\d{8}$', nct_id):
                return None
            pmcid = pmcid if str(pmcid).startswith("PMC") else f"PMC{pmcid}"
            return cls(
                index=int(index),
                nct_id=nct_id,
                pmid=str(pmid),
                pmcid=str(pmcid),
            )
        except Exception:
            return None

@dataclass
class DownloadProgress:
    total_records: int = 0
    processed_records: int = 0
    successful_pm: int = 0
    successful_pmc: int = 0
    successful_ctg: int = 0
    last_processed_index: int = 0
    start_time: str = ""
    last_update: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DownloadProgress":
        return cls(**data)

    def save(self, filepath: Path):
        self.last_update = datetime.now().isoformat()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> Optional["DownloadProgress"]:
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except Exception:
            return None

class RateLimiter:
    def __init__(self, rps: int):
        self.rps = max(1, int(rps))
        self.request_times: List[float] = []
        import threading
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 1.0]
            if len(self.request_times) >= self.rps:
                sleep_time = 1.0 - (now - self.request_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.request_times.append(time.time())

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_bucket_name(index: int, bucket_size: int) -> str:
    start = ((max(1, index) - 1) // bucket_size) * bucket_size + 1
    end = start + bucket_size - 1
    return f"{start}_{end}"



# -----------------------------
# PMC XML pruning helpers
# -----------------------------
PRUNE_SECTION_TITLE_PATTERNS = [
    r"^acknowledg(e)?ments?$",
    r"^conflicts? of interest$",
    r"^funding$",
    r"^author contributions$",
    r"^abbreviations$",
    r"^supplementary material(s)?$",
    r"^data availability$",
    r"^references$",
]

PRUNE_TAGS = {"ref-list", "ack", "fn-group"}

def _local_tag(tag: str) -> str:
    if not isinstance(tag, str):
        return str(tag)
    if tag.startswith("{"):
        try:
            return tag.rsplit('}', 1)[1]
        except Exception:
            return tag
    return tag

def _sec_title_text(elem) -> str:
    for child in list(elem):
        if _local_tag(child.tag) == "title":
            try:
                return (child.text or "").strip().lower()
            except Exception:
                return ""
    return ""

def prune_pmc_xml_str(xml_bytes: bytes,
                      title_patterns: Optional[List[str]] = None,
                      remove_tags: Optional[List[str]] = None) -> bytes:
    def _contains_table(elem) -> bool:
        # Returns True if element contains table-wrap or table
        for d in elem.iter():
            if _local_tag(d.tag) in {"table-wrap", "table"}:
                return True
        return False

    try:
        tree = ET.ElementTree(ET.fromstring(xml_bytes))
    except Exception:
        return xml_bytes
    root = tree.getroot()

    patterns = [re.compile(pat, flags=re.IGNORECASE) for pat in (title_patterns or PRUNE_SECTION_TITLE_PATTERNS)]
    tag_set = set(remove_tags or PRUNE_TAGS)

    # 1) Remove <back> section, but preserve if it contains tables
    for parent in root.iter():
        for child in list(parent):
            if _local_tag(child.tag) == "back":
                if _contains_table(child):
                    # Preserve (can be extended to remove only non-table content inside)
                    continue
                parent.remove(child)

    # 2) Remove specific tags (<ref-list>, <ack>, <fn-group>, etc.)
    for parent in root.iter():
        for child in list(parent):
            if _local_tag(child.tag) in tag_set:
                # Preserve if tag block contains tables
                if _contains_table(child):
                    continue
                parent.remove(child)

    # 3) Preserve tables when matching section title patterns
    for parent in root.iter():
        for child in list(parent):
            if _local_tag(child.tag) == "sec":
                title = _sec_title_text(child)
                if title:
                    for pat in patterns:
                        if pat.match(title):
                            if _contains_table(child):
                                # Preserve
                                break
                            parent.remove(child)
                            break

    try:
        try:
            ET.indent(tree, space="  ", level=0)  # Py3.9+
        except Exception:
            pass
        out = io.BytesIO()
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return out.getvalue()
    except Exception:
        return xml_bytes


def clean_ctg_json(data: dict, variant: Optional[str] = None) -> dict:
    """
    Clean CTG JSON by removing unwanted sections that only contain NCTId.
    
    For multi-variant mode:
    - If variant is 'protocolSection', keep only protocolSection, remove resultsSection if it only has nctId
    - If variant is 'resultsSection', keep only resultsSection, remove protocolSection if it only has nctId
    
    Args:
        data: The CTG JSON data
        variant: The variant name (e.g., 'protocolSection', 'resultsSection')
    
    Returns:
        Cleaned CTG JSON data
    """
    if not isinstance(data, dict):
        return data
    
    def is_minimal_section(section: dict) -> bool:
        """Check if a section only contains identificationModule with nctId"""
        if not isinstance(section, dict):
            return False
        
        # Check if it only has identificationModule
        if len(section) == 1 and "identificationModule" in section:
            id_module = section["identificationModule"]
            # Check if identificationModule only has nctId
            if isinstance(id_module, dict) and len(id_module) == 1 and "nctId" in id_module:
                return True
        
        return False
    
    cleaned_data = data.copy()
    
    # For multi-variant mode, remove the other section if it's minimal
    if variant == "protocolSection":
        if "resultsSection" in cleaned_data and is_minimal_section(cleaned_data["resultsSection"]):
            del cleaned_data["resultsSection"]
    elif variant == "resultsSection":
        if "protocolSection" in cleaned_data and is_minimal_section(cleaned_data["protocolSection"]):
            del cleaned_data["protocolSection"]
    
    return cleaned_data


def build_output_path(
    out_root: Path,
    data_type: str,   # "pm", "pmc", "ctg"
    case_index: int,
    bucket_size: int,
    nct_id: str,
    pmid: str,
    pmcid: str,
    ctg_variant: Optional[str] = None,
) -> Path:
    # Base structure: {data_type}/...
    base = out_root / data_type
    
    # For CTG in multi-variant mode, insert variant subfolder AFTER data_type
    if data_type == "ctg" and ctg_variant:
        base = base / ctg_variant
    
    # Add bucket folder
    base = base / get_bucket_name(case_index, bucket_size)
    
    # Generate filename based on data type
    if data_type == "ctg":
        fname = f"{case_index}_{nct_id}.json"
    elif data_type == "pmc":
        norm_pmcid = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
        fname = f"{case_index}_{norm_pmcid}.xml"
    elif data_type == "pm":
        fname = f"{case_index}_{pmid}.xml"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    return base / fname

# -----------------------------
# Downloaders
# -----------------------------
class NCBIDownloader:
    def __init__(self, rate_limit: int, base_params: Dict[str, str], timeout: int, max_retries: int):
        self.rate = RateLimiter(rate_limit)
        self.base_params = base_params
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, url: str, params: Dict[str, str]) -> Optional[requests.Response]:
        for attempt in range(self.max_retries):
            try:
                self.rate.wait_if_needed()
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except Exception as e:
                is_429 = "429" in str(e) or "Too Many Requests" in str(e)
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) * (2 if is_429 else 1))
                else:
                    logging.error(f"NCBI request failed after {self.max_retries} attempts: {e}")
        return None

    def download_pubmed_xml(self, pmid: str) -> Optional[bytes]:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {**self.base_params, "db": "pubmed", "id": pmid, "retmode": "xml"}
        resp = self._get(url, params)
        if not resp:
            return None
        try:
            ET.fromstring(resp.content)
        except ET.ParseError as e:
            logging.error(f"Invalid PubMed XML for PMID {pmid}: {e}")
            return None
        return resp.content

    def download_pmc_xml(self, pmcid: str) -> Optional[bytes]:
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {**self.base_params, "db": "pmc", "id": pmcid, "retmode": "xml"}
        resp = self._get(url, params)
        if not resp:
            return None
        if len(resp.content) < 100:
            logging.warning(f"Very short PMC response for {pmcid} ({len(resp.content)} bytes)")
            return None
        try:
            ET.fromstring(resp.content)
        except ET.ParseError as e:
            logging.error(f"Invalid PMC XML for {pmcid}: {e}")
            return None
        return resp.content

class CTGDownloader:
    def __init__(self, rate_limit: int, timeout: int, max_retries: int, fields_param: Optional[str] = None):
        self.rate = RateLimiter(rate_limit)
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.fields_param = fields_param

    def download_ctg_json(self, nct_id: str, fields_param_override: Optional[str] = None) -> Optional[dict]:
        url = f"{self.base_url}/{nct_id}"
        params = {}
        # Prefer per-call override, fall back to default
        eff_fields = fields_param_override if fields_param_override is not None else self.fields_param
        if eff_fields:
            params["fields"] = eff_fields
        for attempt in range(self.max_retries):
            try:
                self.rate.wait_if_needed()
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                # Basic validation
                if not isinstance(data, dict) or ("protocolSection" not in data and "studies" not in data):
                    logging.warning(f"CTG response shape unexpected for {nct_id}")
                return data
            except Exception as e:
                is_429 = "429" in str(e) or "Too Many Requests" in str(e)
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) * (2 if is_429 else 1))
                else:
                    logging.error(f"CTG request failed for {nct_id} after {self.max_retries} attempts: {e}")
        return None

# -----------------------------
# Orchestrator
# -----------------------------
class FetchRawSamples:
    def __init__(self, cfg: dict, resume: bool, start_index: Optional[int], skip_existing: bool, workers: int):
        self.cfg = cfg
        self.resume = resume
        self.start_index = start_index
        self.skip_existing = skip_existing
        self.workers = max(1, int(workers))
        self.out_root = Path(cfg["paths"]["out_root"]).resolve()
        self.sample_ids_path = Path(cfg["paths"]["sample_ids"]).resolve()

        # logging
        log_file = Path(cfg["paths"].get("log_file", self.out_root / "pipeline/fetch_raw_samples/download_log.txt"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        # API params from config (no .env dependency)
        api_cfg = cfg.get("api", {}) or {}
        ncbi_cfg = api_cfg.get("ncbi", {}) or {}
        
        ncbi_base_params = {
            "tool": ncbi_cfg.get("tool_name", "ClinicalTrialSampleCollector"),
            "email": ncbi_cfg.get("email", ""),
        }
        api_key = ncbi_cfg.get("api_key", "")
        if api_key:
            ncbi_base_params["api_key"] = api_key

        dl = cfg.get("download", {})
        self.bucket_size = int(dl.get("bucket_size", 100))
        self.max_retries = int(dl.get("max_retries", 3))
        self.request_timeout = int(dl.get("request_timeout", 30))
        self.ncbi_rate_limit = int(dl.get("ncbi_rate_limit", 2 if not api_key else 5))
        self.ctg_rate_limit = int(dl.get("ctg_rate_limit", 5))

        # Enable/disable which data types to download
        types_cfg = dl.get("types", {}) or {}
        self.enable_types: Dict[str, bool] = {
            "pm": bool(types_cfg.get("pm", True)),
            "pmc": bool(types_cfg.get("pmc", True)),
            "ctg": bool(types_cfg.get("ctg", True)),
        }

        # Downloaders
        self.ncbi = NCBIDownloader(
            rate_limit=self.ncbi_rate_limit,
            base_params=ncbi_base_params,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )
        # CTG config: support single (fields: ...) or multi-variant
        ctg_cfg = cfg.get("ctg", {}) or {}
        fields_param = None
        self.ctg_fields_map: Optional[Dict[str, str]] = None  # variant -> fields
        if isinstance(ctg_cfg, dict):
            if "fields" in ctg_cfg:
                # single mode
                fields_param = ctg_cfg.get("fields")
            elif ctg_cfg:
                # multi-variant mode: {variant: {fields: ...}, ...}
                tmp_map: Dict[str, str] = {}
                for variant, sub in ctg_cfg.items():
                    if isinstance(sub, dict) and "fields" in sub and sub["fields"]:
                        tmp_map[str(variant)] = str(sub["fields"])
                if tmp_map:
                    self.ctg_fields_map = tmp_map
        # Initialize downloader with default fields (may be None). For multi, we'll pass per-call overrides.
        self.ctg = CTGDownloader(
            rate_limit=self.ctg_rate_limit,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            fields_param=fields_param,
        )

        # PMC pruning config
        pmc_prune_cfg = cfg.get("pmc_prune", {}) or {}
        self.pmc_prune_enabled = bool(pmc_prune_cfg.get("enabled", False))
        self.pmc_prune_title_patterns = pmc_prune_cfg.get("title_patterns") or PRUNE_SECTION_TITLE_PATTERNS
        self.pmc_prune_remove_tags = pmc_prune_cfg.get("remove_tags") or list(PRUNE_TAGS)

        # Progress/failed
        self.progress_file = Path(cfg["paths"].get("progress_file", self.out_root / "pipeline/fetch_raw_samples/progress.json"))
        self.failed_report = Path(cfg["paths"].get("failed_report", self.out_root / "pipeline/fetch_raw_samples/failed_downloads.txt"))
        self.progress: Optional[DownloadProgress] = None
        self.failed_downloads: List[Dict] = []



    def load_records(self) -> List[SampleRecord]:
        records: List[SampleRecord] = []
        with open(self.sample_ids_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = SampleRecord.from_line(line)
                if rec:
                    records.append(rec)
                else:
                    self.logger.warning(f"Skipped invalid line {line_num}")
        self.logger.info(f"Loaded {len(records)} records from {self.sample_ids_path}")
        return records

    def apply_filters(self, records: List[SampleRecord]) -> List[SampleRecord]:
        filtered = records
        # index
        if self.start_index:
            filtered = [r for r in filtered if r.index >= self.start_index]
            self.logger.info(f"Filter index >= {self.start_index}: {len(filtered)} remain")
        return filtered

    def ensure_dir(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def check_existing(self, record: SampleRecord) -> Dict[str, bool]:
        exists = {"pm": False, "pmc": False, "ctg": False}
        if not self.skip_existing:
            return exists
        pm_path = build_output_path(self.out_root, "pm", record.index, self.bucket_size, record.nct_id, record.pmid, record.pmcid)
        pmc_path = build_output_path(self.out_root, "pmc", record.index, self.bucket_size, record.nct_id, record.pmid, record.pmcid)
        exists["pm"] = pm_path.exists()
        exists["pmc"] = pmc_path.exists()
        # CTG: single vs multi-variant
        if self.ctg_fields_map:
            # all variants must exist to consider CTG as existing
            all_exist = True
            for variant in self.ctg_fields_map.keys():
                v_path = build_output_path(self.out_root, "ctg", record.index, self.bucket_size, record.nct_id, record.pmid, record.pmcid, ctg_variant=variant)
                if not v_path.exists():
                    all_exist = False
                    break
            exists["ctg"] = all_exist
        else:
            ctg_path = build_output_path(self.out_root, "ctg", record.index, self.bucket_size, record.nct_id, record.pmid, record.pmcid)
            exists["ctg"] = ctg_path.exists()
        return exists

    def save_bytes(self, path: Path, content: bytes) -> bool:
        try:
            self.ensure_dir(path)
            with open(path, "wb") as f:
                f.write(content)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save bytes {path}: {e}")
            return False

    def save_json(self, path: Path, obj: dict) -> bool:
        try:
            self.ensure_dir(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save json {path}: {e}")
            return False

    def process_record(self, r: SampleRecord) -> Tuple[int, Dict[str, bool]]:
        # Only track results for enabled types
        results: Dict[str, bool] = {}
        exist = self.check_existing(r)

        # PM (PubMed)
        if self.enable_types.get("pm", True):
            if exist["pm"]:
                results["pm"] = True
            else:
                pm_bytes = self.ncbi.download_pubmed_xml(r.pmid)
                if pm_bytes:
                    pm_path = build_output_path(self.out_root, "pm", r.index, self.bucket_size, r.nct_id, r.pmid, r.pmcid)
                    results["pm"] = self.save_bytes(pm_path, pm_bytes)
                    time.sleep(0.1)

        # PMC
        if self.enable_types.get("pmc", True):
            if exist["pmc"]:
                results["pmc"] = True
            else:
                pmc_bytes = self.ncbi.download_pmc_xml(r.pmcid)
                if pmc_bytes:
                    # Optionally prune
                    if self.pmc_prune_enabled:
                        try:
                            pmc_bytes = prune_pmc_xml_str(
                                pmc_bytes,
                                title_patterns=self.pmc_prune_title_patterns,
                                remove_tags=self.pmc_prune_remove_tags,
                            )
                        except Exception as _:
                            pass
                    pmc_path = build_output_path(self.out_root, "pmc", r.index, self.bucket_size, r.nct_id, r.pmid, r.pmcid)
                    results["pmc"] = self.save_bytes(pmc_path, pmc_bytes)
                    time.sleep(0.1)

        # CTG
        if self.enable_types.get("ctg", True):
            if exist["ctg"]:
                results["ctg"] = True
            else:
                # Single vs multi-variant
                if self.ctg_fields_map:
                    variant_success_all = True
                    for variant, fields in self.ctg_fields_map.items():
                        ctg_path = build_output_path(
                            self.out_root,
                            "ctg",
                            r.index,
                            self.bucket_size,
                            r.nct_id,
                            r.pmid,
                            r.pmcid,
                            ctg_variant=variant,
                        )
                        # Skip per-variant if file exists and skipping is enabled
                        if self.skip_existing and ctg_path.exists():
                            continue
                        ctg_data = self.ctg.download_ctg_json(r.nct_id, fields_param_override=fields)
                        if ctg_data is None:
                            variant_success_all = False
                            self.logger.warning(f"CTG download failed for {r.nct_id} variant '{variant}'")
                            continue
                        
                        # Clean the JSON: remove unwanted sections that only contain NCTId
                        ctg_data = clean_ctg_json(ctg_data, variant=variant)
                        
                        ok = self.save_json(ctg_path, ctg_data)
                        if not ok:
                            variant_success_all = False
                    results["ctg"] = variant_success_all
                else:
                    ctg_data = self.ctg.download_ctg_json(r.nct_id)
                    if ctg_data is not None:
                        # Clean the JSON (no variant specified for single mode)
                        ctg_data = clean_ctg_json(ctg_data, variant=None)
                        
                        ctg_path = build_output_path(self.out_root, "ctg", r.index, self.bucket_size, r.nct_id, r.pmid, r.pmcid)
                        results["ctg"] = self.save_json(ctg_path, ctg_data)

        # Update progress pointers
        if self.progress:
            self.progress.last_processed_index = r.index

        # Track failures
        failed = [k for k, v in results.items() if not v]
        if failed:
            self.failed_downloads.append({
                "index": r.index,
                "nct_id": r.nct_id,
                "pmid": r.pmid,
                "pmcid": r.pmcid,
                "failed_types": failed
            })
            self.logger.warning(f"[{r.index}] Failed: {', '.join(failed)}")

        return r.index, results

    def save_failed_report(self):
        if not self.failed_downloads:
            self.logger.info("No failed downloads")
            return
        self.failed_report.parent.mkdir(parents=True, exist_ok=True)
        with open(self.failed_report, "w", encoding="utf-8") as f:
            f.write("# Failed Downloads Report\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total failed records: {len(self.failed_downloads)}\n")
            f.write("# Format: [Index] NCT_ID, PMID, PMCID\n\n")
            for fd in self.failed_downloads:
                f.write(f"[{fd['index']}] {fd['nct_id']}, {fd['pmid']}, {fd['pmcid']}\n")
        self.logger.info(f"Failed report saved: {self.failed_report}")

    def run(self):
        self.logger.info("Starting fetch_raw_samples")

        # Load or init progress
        if self.resume:
            self.progress = DownloadProgress.load(self.progress_file)
            if self.progress:
                self.logger.info(f"Resuming: {self.progress.processed_records} processed previously")
        if not self.progress:
            self.progress = DownloadProgress()
            self.progress.start_time = datetime.now().isoformat()

        # Load & filter records
        all_records = self.load_records()
        if not all_records:
            self.logger.error("No records found")
            return
        self.progress.total_records = len(all_records)
        records = self.apply_filters(all_records)
        if not records:
            self.logger.info("No records to process after filters")
            return

        self.logger.info(
            f"Enabled types -> pm: {self.enable_types.get('pm', True)}, pmc: {self.enable_types.get('pmc', True)}, ctg: {self.enable_types.get('ctg', True)}"
        )

        # Apply resume based on last processed index
        start_pos = 0
        if self.resume and self.progress and self.progress.last_processed_index:
            min_index = self.progress.last_processed_index + 1
            start_pos = sum(1 for r in records if r.index < min_index)
            self.logger.info(
                f"Resume from index: skipping {start_pos} items with index < {min_index}"
            )

        records_to_process = records[start_pos:]
        total_to_process = len(records_to_process)
        if total_to_process == 0:
            self.logger.info("Nothing to process after resume and filters.")
            self.progress.save(self.progress_file)
            self.save_failed_report()
            self.logger.info(f"Progress file: {self.progress_file}")
            return

        self.logger.info(
            f"Processing {total_to_process}/{len(records)} records this session with {self.workers} workers"
        )

        completed = 0
        batch_succ = {"pm": 0, "pmc": 0, "ctg": 0}

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(self.process_record, r): r for r in records_to_process}
            for fut in as_completed(futures):
                try:
                    _, res = fut.result()
                    completed += 1
                    self.progress.processed_records += 1
                    for k, ok in res.items():
                        if ok:
                            batch_succ[k] += 1
                            if k == "pm":
                                self.progress.successful_pm += 1
                            elif k == "pmc":
                                self.progress.successful_pmc += 1
                            elif k == "ctg":
                                self.progress.successful_ctg += 1

                    # periodic save
                    if completed % 50 == 0:
                        self.progress.save(self.progress_file)

                    if completed % 100 == 0 or completed == total_to_process:
                        self.logger.info(
                            f"Progress: {completed}/{total_to_process} in session "
                            f"({self.progress.processed_records}/{self.progress.total_records} total)"
                        )
                        if completed > 0:
                            self.logger.info(
                                f"  Session success - PM: {batch_succ['pm']}/{completed} "
                                f"({batch_succ['pm']/completed*100:.1f}%), "
                                f"PMC: {batch_succ['pmc']}/{completed} "
                                f"({batch_succ['pmc']/completed*100:.1f}%), "
                                f"CTG: {batch_succ['ctg']}/{completed} "
                                f"({batch_succ['ctg']/completed*100:.1f}%)"
                            )
                except Exception as e:
                    self.logger.error(f"Error processing record: {e}")

        # Finalize
        self.progress.save(self.progress_file)
        self.logger.info("Done.")
        self.logger.info(f"Session processed: {completed}")
        self.logger.info(f"Batch PM/PMC/CTG: {batch_succ['pm']}/{batch_succ['pmc']}/{batch_succ['ctg']}")
        self.logger.info(f"Overall PM/PMC/CTG: {self.progress.successful_pm}/{self.progress.successful_pmc}/{self.progress.successful_ctg}")
        self.logger.info(f"Failures this session: {len(self.failed_downloads)}")
        self.save_failed_report()
        self.logger.info(f"Progress file: {self.progress_file}")

# -----------------------------
# CLI
# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch raw samples (PM/PMC/CTG) with bucketing and results split")
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    
    # Get all settings from config
    runtime_cfg = cfg.get("runtime", {}) or {}
    download_cfg = cfg.get("download", {}) or {}
    
    app = FetchRawSamples(
        cfg=cfg,
        resume=bool(runtime_cfg.get("resume", False)),
        start_index=runtime_cfg.get("start_index"),
        skip_existing=bool(runtime_cfg.get("skip_existing", True)),
        workers=int(download_cfg.get("workers", 2)),
    )
    try:
        app.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        if app.progress:
            app.progress.save(app.progress_file)
            logging.info(f"Progress saved to {app.progress_file}")

if __name__ == "__main__":
    main()
