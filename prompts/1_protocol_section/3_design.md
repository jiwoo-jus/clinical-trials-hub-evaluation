# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the designModule of the study. It defines the overall study design (study type, phase, allocation, masking, number of participants, etc.) in detail.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "designModule": {{
        "studyType": \\ Study classification: ENUM (EXPANDED_ACCESS, INTERVENTIONAL, OBSERVATIONAL)
        "patientRegistry": \\ Indicates if the study is a patient registry: BOOLEAN
        "targetDuration": \\ Follow-up duration for observational patient registry studies: TIME
        "phases": \\ Study phase(s), applicable for drug/biologic trials: ARRAY of ENUM (NA, EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4)
        "designInfo": {{
            "allocation": \\ Method of assigning participants: ENUM (RANDOMIZED, NON_RANDOMIZED, NA)
            "interventionModel": \\ Type of intervention design: ENUM (SINGLE_GROUP, PARALLEL, CROSSOVER, FACTORIAL, SEQUENTIAL)
            "interventionModelDescription": \\ Description of the intervention model: TEXT
            "primaryPurpose": \\ The main objective of the intervention(s) being evaluated by the clinical trial: ENUM (TREATMENT, PREVENTION, DIAGNOSTIC, ECT, SUPPORTIVE_CARE, SCREENING, HEALTH_SERVICES_RESEARCH, BASIC_SCIENCE, DEVICE_FEASIBILITY, OTHER)
            "observationalModel": \\ Study model for observational studies: ENUM (COHORT, CASE_CONTROL, CASE_ONLY, CASE_CROSSOVER, ECOLOGIC_OR_COMMUNITY, FAMILY_BASED, DEFINED_POPULATION, NATURAL_HISTORY, OTHER)
            "timePerspective": \\ Time perspective for observational studies: ENUM (RETROSPECTIVE, PROSPECTIVE, CROSS_SECTIONAL, OTHER)
            "maskingInfo": {{
                "masking": \\ Level of blinding: ENUM (NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE)
                "maskingDescription": \\ Detailed description of masking: TEXT (max 1000 chars)
                "whoMasked": \\ Groups involved in masking: ARRAY of ENUM (PARTICIPANT, CARE_PROVIDER, INVESTIGATOR, OUTCOMES_ASSESSOR)
            }}
        }},
        "enrollmentInfo": {{
            "count": \\ Number of participants enrolled: NUMERIC
            "type": \\ Actual or estimated enrollment: ENUM (ACTUAL, ESTIMATED)
        }}
    }}
}}
```