# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the eligibilityModule of the study. It specifies the eligibility criteria for participating in this clinical trial.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "eligibilityModule": {{
        "eligibilityCriteria": \\ Inclusion and exclusion criteria for participant selection, formatted as a bulleted list under respective headers: TEXT (max 20000 chars)
        "healthyVolunteers": \\ Indicates if healthy volunteers without the studied condition can participate: BOOLEAN
        "sex": \\ Eligible participant sex: ENUM (FEMALE, MALE, ALL)
        "minimumAge": \\ Minimum age required for participation, with unit of time: TEXT (Years, Months, Weeks, Days, Hours, Minutes, N/A)
        "maximumAge": \\ Maximum age allowed for participation, with unit of time: TEXT (Years, Months, Weeks, Days, Hours, Minutes, N/A)
        "stdAges": \\ Standardized age categories: ARRAY of ENUM (CHILD, ADULT, OLDER_ADULT)
        "studyPopulation": \\ (Observational studies only) Description of the population source for cohorts or groups: TEXT (max 1000 chars)
        "samplingMethod": \\ (Observational studies only) Method used for sampling: ENUM (PROBABILITY_SAMPLE, NON_PROBABILITY_SAMPLE)
    }}
}}
```