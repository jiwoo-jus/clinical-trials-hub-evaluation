# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the descriptionModule and conditionsModule of the study. DescriptionModule offers a brief introduction or summary of the clinical trial. ConditionsModule specifies the target conditions or topics (keywords), indicating which are being studied.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "descriptionModule": {{
        "briefSummary": \\ Concise summary of the study, including its hypothesis, in layman's terms: TEXT (max 5000 chars)
        "detailedDescription": \\ Extended study description with technical details, excluding full protocol or duplicate information: TEXT (max 32000 chars)
    }},
    "conditionsModule": {{
        "conditions": \\ List of disease(s) or condition(s) studied, preferably using MeSH or SNOMED CT terms: ARRAY of TEXT
        "keywords": \\ List of descriptive words or phrases related to the study, preferably using MeSH terms: ARRAY of TEXT
    }}
}}
```