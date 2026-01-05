import os
import re
import time
import json
import datetime
import logging
import requests
import psycopg2
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pathlib import Path

# ------------------- Load environment variables -------------------
load_dotenv(dotenv_path="/{YOURPATH}/fetch_sample_ids_config.env")
BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
output_directory = BASE_DIR / "DATA"
sample_id_file = output_directory / 'sample_ids.txt'
sample_id_history_file = output_directory / 'sample_id_history.txt'
pmid_log_file = output_directory / 'pmid_log.txt'

ncbi_api_key = os.getenv("NCBI_API_KEY", "")
tool_email = os.getenv("NCBI_EMAIL", "")
tool_name = os.getenv("NCBI_TOOL_NAME", "")
max_new_records = int(os.getenv("MAX_NEW_RECORDS"))
case_start_index = os.getenv("CASE_START_INDEX")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
NCBI_MAX_IDS_PER_REQUEST = int(os.getenv("NCBI_MAX_IDS_PER_REQUEST"))
CTG_MAX_PAGE_SIZE = int(os.getenv("CTG_MAX_PAGE_SIZE"))
CTG_MAX_IDS_PER_REQUEST = int(os.getenv("CTG_MAX_IDS_PER_REQUEST"))

PUBMED_QUERY = '''
(("randomized controlled trial"[Publication Type] OR "controlled clinical trial"[Publication Type] OR 
"randomized"[Title/Abstract] OR "placebo"[Title/Abstract] OR "clinical trials as topic"[MeSH Terms:noexp] OR 
"randomly"[Title/Abstract] OR "trial"[Title]) NOT ("animals"[MeSH Terms] NOT "humans"[MeSH Terms])) 
AND ("english"[Language] OR "English"[lang]) AND "pubmed pmc open access"[Filter] 
AND "clinicaltrials gov"[Secondary Source ID] AND "2021/01/01"[Date - Publication] : "3000"[Date - Publication]
'''.replace('\n', ' ').strip()

LOG_FILE_PATH = output_directory / 'execution_log.txt'
os.makedirs(output_directory, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger()

class RateLimiter:
    def __init__(self, requests_per_second: int):
        self.requests_per_second = requests_per_second
        self.request_times = []
    
    def wait_if_needed(self):
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        
        if len(self.request_times) >= self.requests_per_second:
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_times = []
        
        self.request_times.append(now)

# Rate limiters
ncbi_limiter = RateLimiter(10 if ncbi_api_key else 3)
ctg_limiter = RateLimiter(10)

class DatabaseManager:
    def __init__(self):
        self.conn_params = {
            "host": DB_HOST,
            "port": DB_PORT,
            "user": DB_USER,
            "dbname": DB_NAME
        }
    
    def get_connection(self):
        return psycopg2.connect(**self.conn_params)
    
    def check_ctg_references(self, nct_pmid_pairs: List[Tuple[str, str]]) -> Dict[str, bool]:
        if not nct_pmid_pairs:
            return {}
        
        nct_ids = [pair[0] for pair in nct_pmid_pairs]
        
        sql = """
            SELECT 
                sr.nct_id,
                sr.pmid,
                COUNT(sr.pmid) OVER (PARTITION BY sr.nct_id) as pmid_count
            FROM ctgov.study_references sr
            WHERE sr.nct_id = ANY(%s)
            AND sr.pmid IS NOT NULL
        """
        
        results = {}
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (nct_ids,))
                    rows = cur.fetchall()
                    
                    nct_references = {}
                    for nct_id, pmid, pmid_count in rows:
                        if nct_id not in nct_references:
                            nct_references[nct_id] = {
                                'pmids': set(),
                                'count': pmid_count
                            }
                        if pmid:
                            nct_references[nct_id]['pmids'].add(str(pmid))
                    
                    for nct_id, target_pmid in nct_pmid_pairs:
                        if nct_id in nct_references:
                            ref_info = nct_references[nct_id]
                            is_one_to_one = (
                                ref_info['count'] == 1 and 
                                target_pmid in ref_info['pmids']
                            )
                            results[nct_id] = is_one_to_one
                        else:
                            results[nct_id] = False
                    
            logger.info(f"Checked {len(nct_pmid_pairs)} NCT-PMID pairs in DB, {sum(results.values())} are 1:1 mappings")
            return results
            
        except Exception as e:
            logger.error(f"Database error checking CTG references: {e}")
            return {nct_id: False for nct_id, _ in nct_pmid_pairs}

class CTGAPIClient:
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.db_manager = DatabaseManager()
    
    def fetch_studies_with_results(self, nct_ids: List[str]) -> Dict[str, bool]:
        results = {}
        
        for i in range(0, len(nct_ids), CTG_MAX_IDS_PER_REQUEST):
            chunk = nct_ids[i:i + CTG_MAX_IDS_PER_REQUEST]
            
            ctg_limiter.wait_if_needed()
            
            query_ids = ",".join(chunk)
            params = {
                "query.id": query_ids,
                "fields": "NCTId,HasResults",
                "filter.overallStatus": "COMPLETED",
                "pageSize": len(chunk),
                "format": "json"
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for study in data.get("studies", []):
                    try:
                        nct_id = study["protocolSection"]["identificationModule"]["nctId"]
                        has_results = study.get("hasResults", False)
                        results[nct_id] = has_results
                    except KeyError:
                        continue
                
                logger.info(f"Fetched hasResults for {len(chunk)} NCT IDs from CTG API")
                
            except Exception as e:
                logger.error(f"Error fetching CTG data for chunk: {e}")
                for nct_id in chunk:
                    results[nct_id] = False
        
        return results

class NCBIClient:
    def __init__(self):
        self.base_params = {
            "tool": tool_name,
            "email": tool_email
        }
        if ncbi_api_key:
            self.base_params["api_key"] = ncbi_api_key

    def _clean_json_response(self, response_text: str) -> str:
        import string
        printable = set(string.printable)
        cleaned = ''.join(filter(lambda x: x in printable, response_text))
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        return cleaned

    def _safe_json_parse(self, response) -> dict:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error, trying to clean response: {e}")
            cleaned_text = self._clean_json_response(response.text)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON even after cleaning: {e2}")
                logger.error(f"Response preview: {response.text[:300]}...")
                ultra_clean = re.sub(r'[^\x20-\x7E\s]', '', response.text)
                try:
                    return json.loads(ultra_clean)
                except json.JSONDecodeError as e3:
                    logger.error(f"Ultra-clean also failed: {e3}")
                    raise e3

    def search_all_pmids_multiple_queries(self) -> List[str]:
        all_pmids = []

        years = list(range(2021, 2025))
        
        for year in years:
            year_query = PUBMED_QUERY.replace(
                '"2021/01/01"[Date - Publication] : "3000"[Date - Publication]',
                f'"{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication]'
            )
            
            logger.info(f"Searching PMIDs for year {year}...")
            
            ncbi_limiter.wait_if_needed()
            
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                **self.base_params,
                "db": "pubmed",
                "term": year_query,
                "retmode": "json",
                "retmax": 9999,
                "usehistory": "n"
            }
            
            try:
                resp = requests.get(search_url, params=params, timeout=30)
                resp.raise_for_status()
                
                data = self._safe_json_parse(resp)["esearchresult"]
                year_count = int(data["count"])
                year_pmids = data.get("idlist", [])
                
                all_pmids.extend(year_pmids)
                logger.info(f"Year {year}: Found {year_count} PMIDs, retrieved {len(year_pmids)}")
                
                if year_count > 9999:
                    logger.warning(f"Year {year} has {year_count} PMIDs (>9999), only got first 9999")
                
            except Exception as e:
                logger.error(f"Error searching PMIDs for year {year}: {e}")
                continue
        
        logger.info(f"Total PMIDs collected from all years: {len(all_pmids)}")
        return all_pmids

    def search_pmids(self) -> Tuple[int, List[str]]:
        
        ncbi_limiter.wait_if_needed()
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            **self.base_params,
            "db": "pubmed",
            "term": PUBMED_QUERY,
            "retmode": "json",
            "retmax": 0
        }
        
        resp = requests.get(search_url, params=params, timeout=30)
        resp.raise_for_status()
        data = self._safe_json_parse(resp)["esearchresult"]
        total_count = int(data["count"])
        
        logger.info(f"Total PMIDs matching query: {total_count}")
        
        if total_count <= 9999:
            logger.info("Total count <= 9999, fetching all at once...")
            params["retmax"] = total_count
            
            ncbi_limiter.wait_if_needed()
            resp = requests.get(search_url, params=params, timeout=30)
            resp.raise_for_status()
            data = self._safe_json_parse(resp)["esearchresult"]
            all_pmids = data.get("idlist", [])
            
        else:
            logger.info(f"Total count > 9999, splitting by years...")
            all_pmids = self.search_all_pmids_multiple_queries()
        
        return total_count, all_pmids
    
    def fetch_articles_batch(self, pmids: List[str]) -> List[ET.Element]:
        """Fetch detailed information for batch of PMIDs"""
        articles = []
        max_retries = 3
        
        for i in range(0, len(pmids), NCBI_MAX_IDS_PER_REQUEST):
            chunk = pmids[i:i + NCBI_MAX_IDS_PER_REQUEST]
            
            ncbi_limiter.wait_if_needed()
            
            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                **self.base_params,
                "db": "pubmed",
                "id": ",".join(chunk),
                "retmode": "xml"
            }
            
            for attempt in range(max_retries):
                try:
                    resp = requests.get(efetch_url, params=params, timeout=30)
                    resp.raise_for_status()
                    
                    if not resp.text.strip():
                        logger.warning(f"Empty XML response for chunk, attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            break
                    
                    root = ET.fromstring(resp.text)
                    chunk_articles = root.findall(".//PubmedArticle")
                    articles.extend(chunk_articles)
                    
                    logger.info(f"Fetched XML for {len(chunk)} PMIDs, got {len(chunk_articles)} articles")
                    break
                    
                except (requests.RequestException, ET.ParseError) as e:
                    logger.warning(f"Error fetching articles for chunk, attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        logger.error(f"Failed to fetch chunk after {max_retries} attempts")
        
        return articles
    
    def convert_pmids_to_pmcids(self, pmids: List[str]) -> Dict[str, Optional[str]]:
        """Convert PMIDs to PMCIDs"""
        pmcid_map = {}
        max_retries = 3
        
        for i in range(0, len(pmids), NCBI_MAX_IDS_PER_REQUEST):
            chunk = pmids[i:i + NCBI_MAX_IDS_PER_REQUEST]
            
            ncbi_limiter.wait_if_needed()
            
            url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            params = {
                **self.base_params,
                "ids": ",".join(chunk),
                "format": "json"
            }
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = self._safe_json_parse(response)
                    
                    for record in data.get("records", []):
                        pmid = record.get("pmid")
                        pmcid = record.get("pmcid")
                        if pmid and pmcid:
                            if pmcid.startswith("PMC"):
                                pmcid = pmcid[3:]
                            pmcid_map[pmid] = pmcid
                    
                    logger.info(f"Converted {len(chunk)} PMIDs to PMCIDs, got {len([k for k, v in pmcid_map.items() if k in chunk and v])} conversions")
                    break
                    
                except Exception as e:
                    logger.warning(f"Error converting PMIDs to PMCIDs, attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        logger.error(f"Failed to convert chunk after {max_retries} attempts")
        
        return pmcid_map

class ClinicalTrialProcessor:
    """Clinical trial data processor"""
    def __init__(self):
        self.ncbi_client = NCBIClient()
        self.ctg_client = CTGAPIClient()
        self.db_manager = DatabaseManager()
    
    def extract_nct_ids_from_article(self, article: ET.Element) -> List[str]:
        """Extract NCT IDs from XML article"""
        nct_ids = []
        
        for db in article.findall(".//DataBankList/DataBank"):
            db_name = db.find("DataBankName")
            if db_name is not None and re.search(r"clinical\s?trials\.?\s?gov", db_name.text, re.IGNORECASE):
                for acc_num in db.findall(".//AccessionNumber"):
                    if acc_num is not None and acc_num.text:
                        nct_id = acc_num.text.strip()
                        if nct_id.startswith("NCT"):
                            nct_ids.append(nct_id)
        
        return nct_ids
    
    def process_articles_batch(self, articles: List[ET.Element], 
                             existing_pmids: Set[str]) -> List[Dict[str, str]]:
        """Process article batch"""
        candidates = []
        
        for article in articles:
            pmid_node = article.find(".//PMID")
            if pmid_node is None or not pmid_node.text:
                continue
            
            pmid = pmid_node.text.strip()
            if pmid in existing_pmids:
                continue
            
            nct_ids = self.extract_nct_ids_from_article(article)
            if len(nct_ids) == 1:
                candidates.append({
                    "pmid": pmid,
                    "nct_id": nct_ids[0],
                    "article": article
                })
        
        if not candidates:
            return []
        
        pmids = [c["pmid"] for c in candidates]
        pmcid_map = self.ncbi_client.convert_pmids_to_pmcids(pmids)
        
        candidates_with_pmcid = []
        for c in candidates:
            pmcid = pmcid_map.get(c["pmid"])
            if pmcid:
                c["pmcid"] = pmcid
                candidates_with_pmcid.append(c)
        
        if not candidates_with_pmcid:
            return []
        
        nct_ids = [c["nct_id"] for c in candidates_with_pmcid]
        has_results_map = self.ctg_client.fetch_studies_with_results(nct_ids)
        
        nct_pmid_pairs = [(c["nct_id"], c["pmid"]) for c in candidates_with_pmcid]
        one_to_one_map = self.db_manager.check_ctg_references(nct_pmid_pairs)
        
        valid_records = []
        for c in candidates_with_pmcid:
            nct_id = c["nct_id"]
            if has_results_map.get(nct_id, False) and one_to_one_map.get(nct_id, False):
                valid_records.append({
                    "NCTID": nct_id,
                    "PMID": c["pmid"],
                    "PMCID": c["pmcid"]
                })
                existing_pmids.add(c["pmid"])
                logger.info(f"Valid record: PMID={c['pmid']}, NCTID={nct_id}, PMCID={c['pmcid']}")
        
        return valid_records
    
    def process_all_pmids(self, pmids: List[str], existing_pmids: Set[str], 
                         max_records: int) -> List[Dict[str, str]]:
        """Process all PMIDs"""
        all_valid_records = []
        
        batch_size = 1000
        
        for i in range(0, len(pmids), batch_size):
            if len(all_valid_records) >= max_records:
                break
            
            batch_pmids = pmids[i:i + batch_size]
            
            new_pmids = [p for p in batch_pmids if p not in existing_pmids]
            if not new_pmids:
                continue
            
            logger.info(f"Processing batch {i//batch_size + 1}: {len(new_pmids)} new PMIDs")
            
            articles = self.ncbi_client.fetch_articles_batch(new_pmids)
            
            valid_records = self.process_articles_batch(articles, existing_pmids)
            all_valid_records.extend(valid_records)
            
            logger.info(f"Batch complete. Valid records: {len(valid_records)}, Total: {len(all_valid_records)}")
            
            if len(all_valid_records) >= max_records:
                all_valid_records = all_valid_records[:max_records]
                break
        
        return all_valid_records

def load_existing_pmids() -> Set[str]:
    """Extract PMIDs from existing files"""
    pmid_set = set()
    for file_path in [sample_id_file, sample_id_history_file, pmid_log_file]:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    pmid_set.update(re.findall(r"\b\d+\b", line))
    return pmid_set

def get_max_case_index() -> int:
    """Check maximum case number so far"""
    max_index = 0
    if not os.path.exists(sample_id_file):
        return max_index
    
    pattern = re.compile(r"^\[(\d+)\]")
    with open(sample_id_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                idx = int(match.group(1))
                if idx > max_index:
                    max_index = idx
    return max_index

def save_records(records: List[Dict[str, str]]):
    """Save records"""
    if not records:
        logger.info("No new records to save")
        return
    
    if os.path.exists(sample_id_file):
        backup_file = sample_id_file.replace(
            ".txt",
            f"_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(sample_id_file, "r", encoding="utf-8") as src:
            with open(backup_file, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        logger.info(f"Created backup: {backup_file}")
    
    start_case = get_max_case_index() + 1 if case_start_index == "AUTO" else int(case_start_index)

    with open(sample_id_file, "a", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            f.write(f"[{start_case + idx}] NCTID: {rec['NCTID']}, PMID: {rec['PMID']}, PMCID: {rec['PMCID']}\n")
    
    with open(sample_id_history_file, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n=== Update on {timestamp} ===\n")
        f.write(f"Added {len(records)} new records\n")
        f.write(f"Case numbers: {start_case} - {start_case + len(records) - 1}\n")
    
    with open(pmid_log_file, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(f"{rec['PMID']}\n")
    
    logger.info(f"Saved {len(records)} new records")

def main():
    """Main process"""
    logger.info("=" * 80)
    logger.info("Starting Optimized Clinical Trial Sample Collection")
    logger.info(f"Config: API Key={'Yes' if ncbi_api_key else 'No'}, Max records={max_new_records}")
    logger.info("=" * 80)
    
    os.makedirs(os.path.dirname(sample_id_file), exist_ok=True)

    logger.info("Loading existing PMIDs...")
    existing_pmids = load_existing_pmids()
    logger.info(f"Found {len(existing_pmids)} existing PMIDs")
    
    processor = ClinicalTrialProcessor()
    
    logger.info("Searching PubMed...")
    total_count, all_pmids = processor.ncbi_client.search_pmids()
    
    logger.info(f"Processing {len(all_pmids)} PMIDs...")
    valid_records = processor.process_all_pmids(
        all_pmids, 
        existing_pmids, 
        max_new_records
    )
    
    save_records(valid_records)
    
    logger.info("=" * 80)
    logger.info(f"Process completed. Total valid records found: {len(valid_records)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()