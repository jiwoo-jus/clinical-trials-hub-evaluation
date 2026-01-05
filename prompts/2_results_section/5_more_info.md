# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the moreInfoModule of the study. It contains additional information including limitations and caveats, certain agreements, and point of contact.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "moreInfoModule": {{
        "limitationsAndCaveats": {{
            "description": // Study limitations and caveats discussed in the paper: TEXT (max 500 chars)
        }},
        "certainAgreement": {{
            "piSponsorEmployee": // Whether PIs are sponsor employees: BOOLEAN
            "restrictionType": // Type of disclosure restriction: ENUM ("LTE60", "GT60", "OTHER")
            "restrictiveAgreement": // Whether restrictive agreements exist: BOOLEAN
            "otherDetails": // Details if restriction type is OTHER: TEXT
        }},
        "pointOfContact": {{
            "title": // Contact person's name or title: TEXT (max 255 chars)
            "organization": // Contact organization: TEXT (max 255 chars)
            "email": // Contact email: TEXT (max 255 chars)
            "phone": // Contact phone: TEXT (max 30 chars)
            "phoneExt": // Phone extension: TEXT
        }}
    }}
}}
```