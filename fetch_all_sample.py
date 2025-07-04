import os
import re
import requests
import json
import time
from dotenv import load_dotenv
from pathlib import Path

# ------------------- Load environment variables -------------------
load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
CTG_PATH = BASE_DIR / "DATA/CTG"
PM_PATH = BASE_DIR / "DATA/PM"
PMC_PATH = BASE_DIR / "DATA/PMC"
TARGET_SAMPLES = BASE_DIR / "DATA/sample_ids.txt"

# ------------------- Output/Input File Paths -------------------
SELECTED_CASE_NUMBERS = list(range(1, 3))  

FETCH_CTG = True  # Set to False to skip ClinicalTrials.gov data
FETCH_PM = False  # Set to False to skip PubMed data
FETCH_PMC = False  # Set to False to skip PubMed Central data
FIELD_GROUP_KEY = "group1"
FIELD_GROUP = {
    "group1": "NCTId,OrgStudyId,OrgStudyIdType,OrgStudyIdLink,SecondaryId,SecondaryIdType,SecondaryIdDomain,SecondaryIdLink,BriefTitle,OfficialTitle,OrgFullName,OrgClass,BriefSummary,DetailedDescription,Condition,Keyword,StudyType,PatientRegistry,TargetDuration,Phase,DesignAllocation,DesignInterventionModel,DesignInterventionModelDescription,DesignPrimaryPurpose,DesignObservationalModel,DesignTimePerspective,DesignMasking,DesignMaskingDescription,DesignWhoMasked,EnrollmentCount,EnrollmentType,ArmGroupLabel,ArmGroupType,ArmGroupDescription,ArmGroupInterventionName,InterventionType,InterventionName,InterventionDescription,InterventionArmGroupLabel,PrimaryOutcomeMeasure,PrimaryOutcomeDescription,PrimaryOutcomeTimeFrame,OtherOutcomeMeasure,OtherOutcomeDescription,OtherOutcomeTimeFrame,EligibilityCriteria,HealthyVolunteers,Sex,MinimumAge,MaximumAge,StdAge,StudyPopulation,SamplingMethod",
}

# ------------------- Ensure directories exist -------------------
os.makedirs(CTG_PATH, exist_ok=True)
os.makedirs(PM_PATH, exist_ok=True)
os.makedirs(PMC_PATH, exist_ok=True)


# Save data to a file
def save_to_file(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if str(file_path).endswith('.json'):
                json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                f.write(data)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to file: {e}")


# Fetch ClinicalTrials.gov data
def fetch_ctg(nctid, caseno):
    url = f"https://clinicaltrials.gov/api/v2/studies/{nctid}"
    params = {"fields": FIELD_GROUP.get(FIELD_GROUP_KEY)}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        file_path = CTG_PATH / f"case_{caseno}_{nctid}.json"
        save_to_file(response.json(), file_path)
    else:
        print(f"Error fetching CTG data for {nctid}: {response.status_code}")


# Fetch PubMed data
def fetch_pm(pmid, caseno):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        file_path = PM_PATH / f"pm_{caseno}_{pmid}.xml"
        save_to_file(response.text, file_path)
    else:
        print(f"Error fetching PubMed data for {pmid}: {response.status_code}")


# Fetch PubMed Central data
def fetch_pmc(pmcid, caseno):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        file_path = PMC_PATH / f"pmc_{caseno}_PMC{pmcid}.xml"
        save_to_file(response.text, file_path)
    else:
        print(f"Error fetching PMC data for {pmcid}: {response.status_code}")


# Extract metadata from the input file
def extract_metadata():
    if not TARGET_SAMPLES.exists():
        raise FileNotFoundError(f"Input file not found: {TARGET_SAMPLES}")

    with open(TARGET_SAMPLES, 'r', encoding='utf-8') as file:
        content = file.readlines()

    # [0] NCTID: NCT02229851, PMID: 39991735, PMCID: 11842232
    pattern = re.compile(r'\[(\d+)\]\s*NCTID:\s*(NCT\d+),\s*PMID:\s*(\d+),\s*PMCID:\s*(\d+)')
    extracted_data = []

    for line in content:
        match = pattern.search(line)
        if match:
            caseno = int(match.group(1))
            if SELECTED_CASE_NUMBERS and caseno not in SELECTED_CASE_NUMBERS:
                continue  # Process only selected case numbers if specified

            metadata = {
                "caseno": caseno,
                "nctid": match.group(2),
                "pmid": match.group(3),
                "pmcid": match.group(4),
            }
            extracted_data.append(metadata)

    return extracted_data


if __name__ == "__main__":
    cases = extract_metadata()

    for case in cases:
        time.sleep(3)  # Delay between API requests
        caseno = case["caseno"]
        if FETCH_CTG:
            fetch_ctg(case["nctid"], caseno)
        if FETCH_PM:
            fetch_pm(case["pmid"], caseno)
        if FETCH_PMC:
            fetch_pmc(case["pmcid"], caseno)