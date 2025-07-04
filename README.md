# Clinical Trials Hub Evaluation Framework

This framework evaluates the information extraction (IE) performance of large language models (LLMs) on biomedical literature. Each model is given a PMC full-text article and tasked with extracting clinical trial information in the structured format of ClinicalTrials.gov (CTG). Evaluation is performed by comparing model outputs against the ground truth CTG records, which are manually aligned 1:1 with the corresponding publications.

## Quick Start

The repository includes **sample IDs** for clinical trials that can be used to test the evaluation framework. To get started, you'll need to download the data and generate model outputs before running evaluations.

### Complete Pipeline Execution Order

To run the complete evaluation pipeline:

```bash
# 1. Download clinical trial data using provided sample IDs
python fetch_all_sample.py

# 2. Generate model outputs from the downloaded data
python generate_model_output.py

# 3. Evaluate model outputs against ground truth
python eval.py

# 4. Generate statistical analysis and visualizations
python generate_eval_stats.py
```

### Primary Evaluation Scripts

#### 1. `eval.py` - Model Output Evaluation
**Purpose:** Evaluates model outputs against ground truth data.

**Run:**
```bash
python eval.py
```

**Outputs:**
- `DATA/EVAL_METRICS_OUTPUT/`: Case-specific and model-specific metric files

#### 2. `generate_eval_stats.py` - Statistical Analysis and Visualization
**Purpose:** Analyzes evaluation results and generates comprehensive visualizations and statistical summaries.

**Run:**
```bash
python generate_eval_stats.py
```

**Outputs:**
- `DATA/EVAL_STATS_OUTPUT/field_frequency.png`: Field frequency charts
- `DATA/EVAL_STATS_OUTPUT/field_performance.png`: Field-wise performance charts
- `DATA/EVAL_STATS_OUTPUT/model_comparison_*.png`: Model comparison charts
- `DATA/EVAL_STATS_OUTPUT/model_field_heatmap.png`: Model-field performance heatmap
- `DATA/EVAL_STATS_OUTPUT/*.csv`: Summary statistics tables

## Installation and Setup

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Environment Variables Configuration

Copy `.env.example` to `.env` and configure the required values:

```bash
cp .env.example .env
```

#### Environment Variables Guide:

**Basic Configuration:**
- `BASE_DIR`: Project base directory (default: `./`)

**NCBI/PubMed API Settings:**
- `NCBI_API_KEY`: NCBI API key ([Get one here](https://www.ncbi.nlm.nih.gov/account/settings/))
- `NCBI_EMAIL`: NCBI account email
- `NCBI_TOOL_NAME`: API tool name (default: `ClinicalTrialSampleCollector`)
- `NCBI_MAX_IDS_PER_REQUEST`: Maximum IDs per request (default: `200`)

**ClinicalTrials.gov API Settings:**
- `CTG_MAX_PAGE_SIZE`: Maximum results per page (default: `1000`)
- `CTG_MAX_IDS_PER_REQUEST`: Maximum IDs per request (default: `300`)

**AI Model API Settings**

**For LiteLLM:**
- `LITELLM_BASE_URL`: LiteLLM server URL
- `LITELLM_API_KEY`: LiteLLM API key

**For Azure OpenAI:**
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_API_VERSION`: API version (e.g., `2024-08-01-preview`)

**Database Settings (Optional):**
- `DB_HOST`: PostgreSQL host (e.g., `127.0.0.1`)
- `DB_PORT`: PostgreSQL port (e.g., `5432`)
- `DB_NAME`: Database name (default: `trials`)
- `DB_USER`: Database username

**Other Settings:**
- `MAX_NEW_RECORDS`: Maximum new records (default: `10000`)
- `CASE_START_INDEX`: Case start index (default: `AUTO`)

## Data Preparation Scripts

The following scripts are used for data collection and preparation.

### 1. `fetch_sample_ids.py` (Optional)
**Purpose:** Collects sample IDs from ClinicalTrials.gov and PubMed(Central). **Sample IDs are already provided in the repository.**

**Run:**
```bash
python fetch_sample_ids.py
```

**Outputs:**
- `DATA/sample_ids.txt`: Collected sample ID list (already included)

### 2. `fetch_all_sample.py`
**Purpose:** Downloads complete data from ClinicalTrials.gov, PubMed, and PMC based on collected sample IDs.

**Run:**
```bash
python fetch_all_sample.py
```

**Configuration Options:**
- `FETCH_CTG`: Whether to download ClinicalTrials.gov data (default: `True`)
- `FETCH_PM`: Whether to download PubMed data (default: `False`)
- `FETCH_PMC`: Whether to download PubMed Central data (default: `False`)

**Outputs:**
- `DATA/CTG/`: ClinicalTrials.gov data (JSON format)
- `DATA/PM/`: PubMed data
- `DATA/PMC/`: PubMed Central data

### 3. `generate_model_output.py`
**Purpose:** Extracts and structures clinical trial information using LLMs (Claude-4-Sonnet, GPT-4o, Llama-70B) from PMC data.

**Run:**
```bash
python generate_model_output.py
```

**Configuration Options:**
- `PROCESS_MODE`: Processing mode (`"range"`, `"specific"`, or `"all"`)
- `START_CASE`: Start case number
- `END_CASE`: End case number
- `SPECIFIC_CASES`: Specific case list
- `MAX_WORKERS`: Maximum worker count (default: `3`)

**Outputs:**
- `DATA/MODEL_OUTPUT/`: Model output results

## Sample Data Included

The repository includes **sample IDs** for clinical trials in `DATA/sample_ids.txt`. You can use these IDs to:

1. Download ground truth data from ClinicalTrials.gov
2. Download corresponding publications from PubMed Central
3. Generate model outputs for evaluation
4. Run the complete evaluation pipeline

To use the provided sample IDs, simply run the data preparation scripts in order.
