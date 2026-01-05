# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the outcomesModule of the study. It dsscribes the primary, secondary, and other outcome measures, showing which indicators are used to assess the trial’s effectiveness and safety.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
"outcomesModule": {{
    "primaryOutcomes": [ \\ Required. List of primary outcome measures used to assess the trial's main objectives: ARRAY of OBJECT
        {{
            "measure": \\ Name of the primary outcome measure: TEXT (max 255 chars)
            "description": \\ Description of the metric used to characterize the primary outcome measure: TEXT (max 999 chars)
            "timeFrame": \\ Time point(s) at which the outcome is measured: TEXT (max 255 chars)
        }}
    ],
    "secondaryOutcomes": [ \\ Conditional. List of secondary outcome measures for additional study assessments: ARRAY of OBJECT
        {{
            "measure": \\ Name of the secondary outcome measure: TEXT
            "description": \\ Description of the metric used to characterize the secondary outcome measure: TEXT
            "timeFrame": \\ Time point(s) at which the outcome is measured: TEXT
        }}
    ],
    "otherOutcomes": [ \\ Optional. List of other pre-specified outcome measures (excluding post-hoc measures): ARRAY of OBJECT
        {{
            "measure": \\ Name of the pre-specified outcome measure: TEXT
            "description": \\ Description of the metric used to characterize the outcome measure: TEXT
            "timeFrame": \\ Time point(s) at which the outcome is measured: TEXT
        }}
    ]
}}
```