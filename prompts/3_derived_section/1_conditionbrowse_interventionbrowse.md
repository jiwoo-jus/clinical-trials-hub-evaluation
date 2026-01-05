# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.

# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the following modules from the study:
1. conditionBrowseModule: MeSH condition term mappings
2. interventionBrowseModule: MeSH intervention term mappings

# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{
    "conditionBrowseModule": {
        "meshes": [ \\ Condition MeSH Terms MeSH terms of Condition/Diseases field
            {
                        "id": \\ Condition MeSH ID MeSH ID: TEXT,
                        "term": \\ Condition MeSH Term MeSH Heading: TEXT
            }
        ],
        "ancestors": [ \\ Ancestors of Condition MeSH Terms Ancestor (higher level and more broad) terms of Condition MeSH terms in MeSH Tree hierarchy
            {
                        "id": \\ Condition Ancestor MeSH ID MeSH ID: TEXT,
                        "term": \\ Condition Ancestor MeSH Term MeSH Heading: TEXT
            }
        ],
        "browseLeaves": [ \\ Condition Leaf Topics Leaf browsing topics for Condition field
            {
                        "id": \\ Condition Leaf Topic ID: TEXT,
                        "name": \\ Condition Leaf Topic Name: TEXT,
                        "relevance": \\ Relevance to Condition Leaf Topic: ENUM (LOW, HIGH)
            }
        ],
        "browseBranches": [ \\ Condition Branch Topics Branch browsing topics for Condition field
            {
                        "abbrev": \\ Condition Branch Topic Short Name: TEXT,
                        "name": \\ Condition Branch Topic Name: TEXT
            }
        ]
    },
    "interventionBrowseModule": {
        "meshes": [ \\ Condition MeSH Terms MeSH terms of Condition/Diseases field
            {
                        "id": \\ Condition MeSH ID MeSH ID: TEXT,
                        "term": \\ Condition MeSH Term MeSH Heading: TEXT
            }
        ],
        "ancestors": [ \\ Ancestors of Condition MeSH Terms Ancestor (higher level and more broad) terms of Condition MeSH terms in MeSH Tree hierarchy
            {
                        "id": \\ Condition Ancestor MeSH ID MeSH ID: TEXT,
                        "term": \\ Condition Ancestor MeSH Term MeSH Heading: TEXT
            }
        ],
        "browseLeaves": [ \\ Condition Leaf Topics Leaf browsing topics for Condition field
            {
                        "id": \\ Condition Leaf Topic ID: TEXT,
                        "name": \\ Condition Leaf Topic Name: TEXT,
                        "relevance": \\ Relevance to Condition Leaf Topic: ENUM (LOW, HIGH)
            }
        ],
        "browseBranches": [ \\ Condition Branch Topics Branch browsing topics for Condition field
            {
                        "abbrev": \\ Condition Branch Topic Short Name: TEXT,
                        "name": \\ Condition Branch Topic Name: TEXT
            }
        ]
    }
}
```