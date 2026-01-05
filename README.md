# Clinical Trials Hub Evaluation Framework

This repository contains a complete pipeline for extracting structured information from PubMed Central (PMC) full-text articles and evaluating the extraction against ClinicalTrials.gov ground truth data. The pipeline extracts clinical trial information following the ClinicalTrials.gov schema using large language models (LLMs).

## Overview

The pipeline consists of six sequential stages:

1. **fetch_sample_ids**: Collect sample case IDs linking PMC articles to ClinicalTrials.gov records
2. **fetch_raw_samples**: Download raw PMC and CTG data for the collected samples
3. **extract**: Extract structured information from PMC text using LLMs
4. **combine**: Restructure extracted data into the target schema format
5. **eval**: Evaluate extracted information against ClinicalTrials.gov ground truth
6. **eval_generate_stats**: Generate comprehensive evaluation statistics and metrics

## Prerequisites

- Python 3.8+
- PostgreSQL database (for sample ID collection)
- NCBI API credentials (email and API key)
- LLM API access (OpenAI, Anthropic, or Google)
- Conda environment (recommended)

## Setup Instructions

### 1. Clone and Navigate

```bash
git clone https://github.com/jiwoo-jus/clinical-trials-hub-evaluation.git

cd clinical-trials-hub-evaluation
```

### 2. Configure Your Paths

**IMPORTANT**: Before running any scripts, you must replace all placeholder paths and credentials throughout the configuration files:

- **{YOURPATH}**: Replace with your absolute path to this directory
  - This path is consistent across all configuration files
  
- **{YOURS}**: Replace with your specific credentials and settings
  - API keys
  - Email addresses
  - Database credentials
  - Model names
  - Conda environment names

**Search and replace these placeholders:**

```bash
# Search for all instances in your IDE or use:
grep -r "{YOURPATH}" .
grep -r "{YOURS}" .
```

Files that require configuration:
- `fetch_sample_ids_config.env`
- `fetch_raw_samples_config.yaml`
- `extract_config.yaml`
- `combine_config.yaml`
- `eval_config.yaml`
- `eval_generate_stats_config.yaml`

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Pipeline Execution

### Stage 1: Fetch Sample IDs

Collect case IDs that link PMC articles with ClinicalTrials.gov records.

**Configuration**: `fetch_sample_ids_config.env`

```bash
python fetch_sample_ids.py
```

**Key configurations to update:**
- `NCBI_API_KEY`: Your NCBI E-utilities API key
- `NCBI_EMAIL`: Your email address registered with NCBI
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`: Database connection details

**Output**: `samples/sample_ids.txt`

---

### Stage 2: Fetch Raw Samples

Download PMC full-text articles and ClinicalTrials.gov records for the collected sample IDs.

**Configuration**: `fetch_raw_samples_config.yaml`

```bash
python fetch_raw_samples.py --config fetch_raw_samples_config.yaml
```

**Key configurations to update:**
- `paths.sample_ids`: Path to sample_ids.txt from Stage 1
- `paths.out_root`: Output directory for downloaded samples
- `api.ncbi.email`: Your NCBI email
- `api.ncbi.api_key`: Your NCBI API key
- `download.workers`: Number of parallel download workers (adjust based on your system)

**Output**:
- `samples/PMC/`: PMC full-text articles organized in buckets
- `samples/CTG/`: ClinicalTrials.gov JSON records organized in buckets

---

### Stage 3: Extract Information

Extract structured information from PMC text using LLMs according to predefined prompts.

**Configuration**: `extract_config.yaml`

**Prerequisites**: Prepare `extract_target/{MODEL_NAME}.csv` files with columns:
- `case_id`: Case identifier
- Module columns (e.g., `identificationModule`, `descriptionModule`): Set to `True` for modules to extract

**Run separately for each model:**

```bash
# Update extract_config.yaml for each model
python extract.py
```

**Key configurations to update:**
- `litellm_api.base_url`: Your LLM API endpoint
- `litellm_api.api_key`: Your LLM API key
- `litellm_api.model`: Model identifier
- `paths.input_csv`: Path to target CSV
- `paths.output_csv`: Path to output CSV
- `paths.pmc_dir`: Path to PMC directory from Stage 2
- `paths.prompts_dir`: Path to prompts directory

**Modules extracted:**
- identificationModule
- descriptionModule
- conditionsModule
- designModule
- armsInterventionsModule
- outcomesModule
- eligibilityModule

**Output**: `extract_output/{MODEL_NAME}.csv`

---

### Stage 4: Combine Modules

Restructure extracted module data into complete JSON documents following the ClinicalTrials.gov schema.

**Configuration**: `combine_config.yaml`

**Prerequisites**: Ensure `combine_target/{MODEL_NAME}/` directories contain the initial JSON files to be combined (if any).

```bash
python combine.py
```

**Key configurations to update:**
- `input.model_outputs_dir`: Path to combine target directory
- `input.models`: List of model names to process
- `output.csv_dir`: Path to extraction output CSVs
- `output.restructured_dir`: Path for restructured JSON output
- `module_overrides`: CSV paths for each model and module

**Output**: `combine_output/{MODEL_NAME}/{case_id}_{MODEL_NAME}.json`

---

### Stage 5: Evaluate

Evaluate the extracted information against ClinicalTrials.gov ground truth using multiple metrics.

**Configuration**: `eval_config.yaml`

```bash
python eval.py --config eval_config.yaml
```

**Key configurations to update:**
- `environment.conda_env`: Your conda environment name
- `environment.activate_command`: Conda activation command
- `processing.index_file`: Path to evaluation index CSV
- `processing.case_id_start` / `case_id_end`: Range of cases to evaluate
- `output.root`: Base directory for evaluation outputs
- `evaluation.models`: List of models with their output paths

**Output**: `eval_output/{section}/` containing metrics for each model

---

### Stage 6: Generate Statistics

Aggregate evaluation metrics and generate comprehensive statistics across all models and fields.

**Configuration**: `eval_generate_stats_config.yaml`

```bash
python eval_generate_stats.py --config eval_generate_stats_config.yaml
```

**Key configurations to update:**
- `index_csv`: Path to evaluation index CSV
- `output_base`: Path to evaluation output directory
- `field_catalog_path`: Path to data definition CSV
- `models`: List of model names to aggregate
- `target_fields`: Fields to include in statistics

**Output**: 
- Aggregated statistics CSV files
- Per-field performance metrics
- Cross-model comparison reports

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{park2025clinicaltrialshubbridgingregistriesliterature,
      title={ClinicalTrialsHub: Bridging Registries and Literature for Comprehensive Clinical Trial Access}, 
      author={Jiwoo Park and Ruoqi Liu and Avani Jagdale and Andrew Srisuwananukorn and Jing Zhao and Lang Li and Ping Zhang and Sachin Kumar},
      year={2025},
      eprint={2512.08193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.08193}, 
}
```

## Contact

For questions or issues, please contact park.3620@osu.edu.
