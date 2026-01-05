# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the adverseEventsModule of the study. It contains information about adverse events including serious adverse events, other adverse events, and mortality data.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "adverseEventsModule": {{
        "frequencyThreshold": // Threshold for reporting other AEs (e.g., "5%"): TEXT
        "timeFrame": // Time period for AE collection: TEXT (max 500 chars)
        "description": // Additional AE collection details: TEXT
        "allCauseMortalityComment": // Explanation about mortality data: TEXT
        "eventGroups": [ // ARRAY of OBJECT - Study arms/groups for AE reporting
            {{
                "id": // Unique group identifier. EG000 is the first group, EG001 is the second, and so on: TEXT
                "title": // Group name: TEXT (max 1500 chars)
                "description": // Group description: TEXT
                "deathsNumAffected": // Number affected by all-cause mortality: TEXT
                "deathsNumAtRisk": // Number at risk for mortality: TEXT
                "seriousNumAffected": // Number with any serious AE: TEXT
                "seriousNumAtRisk": // Number at risk for serious AEs: TEXT
                "otherNumAffected": // Number with any other AE: TEXT
                "otherNumAtRisk": // Number at risk for other AEs: TEXT
            }}
        ],
        "seriousEvents": [ // ARRAY of OBJECT - Serious adverse events by organ system
            {{
                "term": // AE preferred term: TEXT
                "organSystem": // Organ system category: TEXT
                "sourceVocabulary": // Coding dictionary (e.g., "MedDRA 23.0"): TEXT
                "assessmentType": // Collection method: ENUM (NON_SYSTEMATIC_ASSESSMENT, SYSTEMATIC_ASSESSMENT)
                "notes": // Additional description: TEXT
                "stats": [ // ARRAY of OBJECT - Statistics for each group
                    {{
                        "groupId": // References an event group ID: TEXT
                        "numEvents": // Total number of events: TEXT
                        "numAffected": // Number of participants affected: TEXT
                        "numAtRisk": // Number at risk: TEXT
                    }}
                ]
            }}
        ],
        "otherEvents": [ // ARRAY of OBJECT - Other (non-serious) adverse events
            {{
                "term": // AE preferred term: TEXT
                "organSystem": // Organ system category: TEXT
                "sourceVocabulary": // Coding dictionary: TEXT
                "assessmentType": // Collection method: ENUM (NON_SYSTEMATIC_ASSESSMENT, SYSTEMATIC_ASSESSMENT)
                "notes": // Additional description: TEXT
                "stats": [ // ARRAY of OBJECT - Statistics for each group
                    {{
                        "groupId": // References an event group ID: TEXT
                        "numEvents": // Total number of events: TEXT
                        "numAffected": // Number of participants affected: TEXT
                        "numAtRisk": // Number at risk: TEXT
                    }}
                ]
            }}
        ]
    }}
}}
```