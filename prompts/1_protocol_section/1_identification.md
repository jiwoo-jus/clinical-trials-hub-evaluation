# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the identificationModule of the study. It includes identifying information such as the trial’s unique identifier or title, illustrating “who is conducting which trial and where it is registered."
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "identificationModule": {{
        "nctId": \\ ClinicalTrials.gov Identifier. The format is "NCT" followed by an 8-digit number: TEXT (max 11 chars)
        "orgStudyIdInfo": {{
            "id": \\ organization's unique protocol ID: TEXT (max 30 chars)
            "type": \\ Type of organization-issued ID: ENUM (NIH, FDA, VA, CDC, AHRQ, SAMHSA)
            "link": \\ URL link related to OrgStudyId and OrgStudyIdType: TEXT
        }},
        "secondaryIdInfos": [ \\ ARRAY of OBJECT
            {{
                "id": \\ Secondary identifier for funding or registry: TEXT (max 30 chars)
                "type": \\ Type of secondary ID: ENUM (NIH, FDA, VA, CDC, AHRQ, SAMHSA, OTHER_GRANT, EUDRACT_NUMBER, CTIS, REGISTRY, OTHER)
                "domain": \\ Name of funding organization, registry, or issuer: TEXT (max 119 chars)
                "link": \\ URL link related to SecondaryId and SecondaryIdType: TEXT
            }}
        ],
        "organization": {{
            "fullName": \\ Name of the sponsoring organization: TEXT
            "class": \\ Organization type: ENUM (NIH, FED, OTHER_GOV, INDIV, INDUSTRY, NETWORK, AMBIG, OTHER, UNKNOWN)
        }},
        "briefTitle": \\ Short title of the study: TEXT (max 300 chars)
        "officialTitle": \\ Full official title of the study: TEXT (max 600 chars)
        "acronym": \\ Study acronym: TEXT (max 14 chars)
    }}
}}
```