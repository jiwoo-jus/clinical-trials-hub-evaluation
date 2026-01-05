# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the armsInterventionsModule of the study. It details which drugs (or procedures) are administered, to which arms (groups), and how they are applied.
# RESPONSE #
A syntactically correct JSON string:
Format:
```json
{{
    "armsInterventionsModule": {{
        "armGroups": [ \\ ARRAY of OBJECT
            {{
                "label": \\ Name of the arm/group: TEXT
                "type": \\ Type of arm: ENUM (EXPERIMENTAL, ACTIVE_COMPARATOR, PLACEBO_COMPARATOR, SHAM_COMPARATOR, NO_INTERVENTION, OTHER)
                "description": \\ Description of the arm/group: TEXT
                "interventionNames": \\ List of interventions used in this arm/group: ARRAY of TEXT
            }}
        ],
        "interventions": [ \\ ARRAY of OBJECT
            {{
                "type": \\ Type of intervention: ENUM (BEHAVIORAL, BIOLOGICAL, COMBINATION_PRODUCT, DEVICE, DIAGNOSTIC_TEST, DIETARY_SUPPLEMENT, DRUG, GENETIC, PROCEDURE, RADIATION, OTHER)
                "name": \\ Name of the intervention: TEXT
                "description": \\ Description of the intervention: TEXT
                "armGroupLabels": \\ List of arm/group labels associated with this intervention: ARRAY of TEXT
            }}
        ]
    }}
}}
```