# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the participantFlowModule of the study. It describes the flow of participants through each stage of the study, including enrollment, allocation, follow-up, and analysis.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "participantFlowModule": {{
        "preAssignmentDetails": // Description of significant events after participant enrollment but before group assignment: TEXT (max 500 chars)
        "recruitmentDetails": // Key information about recruitment process: TEXT (max 500 chars)
        "typeUnitsAnalyzed": // Unit of analysis (e.g., "Participants", "Eyes", "Lesions"): TEXT
        "groups": [ // ARRAY of OBJECT - Arms/groups in the study flow
            {{
                "id": // Unique group identifier. FG000 is the first group, FG001 is the second, and so on: TEXT
                "title": // Short name of the arm/group: TEXT (max 40 chars)
                "description": // Brief description of the arm/group: TEXT (max 1500 chars)
            }}
        ],
        "periods": [ // ARRAY of OBJECT - Time periods in the study
            {{
                "title": // Period name (e.g., "Overall Study", "Treatment Phase"): TEXT (max 40 chars)
                "milestones": [ // ARRAY of OBJECT - Key milestones
                    {{
                        "type": // Milestone name (e.g., "STARTED", "COMPLETED"): TEXT (max 100 chars)
                        "comment": // Additional information about the milestone: TEXT (max 500 chars)
                        "achievements": [ // ARRAY of OBJECT - Numbers for each group
                            {{
                                "groupId": // References a group ID: TEXT (max 500 chars)
                                "comment": // Explanation if number differs from expected: TEXT (max 500 chars)
                                "numSubjects": // Number of participants: TEXT
                                "numUnits": // Number of units if different from participants: TEXT
                            }}
                        ]
                    }}
                ],
                "dropWithdraws": [ // ARRAY of OBJECT - Reasons for not completing
                    {{
                        "type": // Reason category (e.g., "Adverse Event", "Lost to Follow-up"): TEXT (max 100 chars)
                        "comment": // Additional details about the reason: TEXT
                        "reasons": [ // ARRAY of OBJECT - Numbers for each group
                            {{
                                "groupId": // References a group ID: TEXT
                                "comment": // Additional explanation if needed: TEXT
                                "numSubjects": // Number of participants: TEXT
                            }}
                        ]
                    }}
                ]
            }}
        ]
    }}
}}
```