# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data. Omit any fields if the corresponding data does not exist in the target articles. Do not invent or assume information that is not present in the given content.
# PAPER CONTENT #
{{pmc_text}}

# TARGET #
Extract the outcomeMeasuresModule of the study. It contains the results of primary, secondary, and other outcome measures.
# RESPONSE #
The JSON response must adhere to the following structure, where each field is annotated with its expected type.
Ensure that the output is a valid and correctly formatted JSON string.
Format:
```json
{{
    "outcomeMeasuresModule": {{
        "outcomeMeasures": [ // ARRAY of OBJECT - All outcome measures with results
            {{
                "type": // Outcome type: ENUM ("PRIMARY", "SECONDARY", "OTHER_PRE_SPECIFIED", "POST_HOC")
                "title": // Outcome measure title: TEXT (max 255 chars)
                "description": // Detailed description: TEXT (max 999 chars)
                "populationDescription": // Analysis population if different: TEXT
                "reportingStatus": // Whether data is reported: ENUM ("POSTED", "NOT_POSTED")
                "anticipatedPostingDate": // Expected date if not posted: DATE (format: YYYY or YYYY-MM)
                "paramType": // Type of measure: ENUM ("GEOMETRIC_MEAN", "GEOMETRIC_LEAST_SQUARES_MEAN", "LEAST_SQUARES_MEAN", "LOG_MEAN", "MEAN", "MEDIAN", "NUMBER", "COUNT_OF_PARTICIPANTS", "COUNT_OF_UNITS")
                "dispersionType": // Dispersion/precision type: ENUM ("Not Applicable", "Standard Deviation", "Standard Error", "Inter-Quartile Range", "Full Range", "99% Confidence Interval", "97.5% Confidence Interval", "95% Confidence Interval", "90% Confidence Interval", "80% Confidence Interval", "Other Confidence Interval Level", "Geometric Coefficient of Variation")
                "unitOfMeasure": // Unit of measurement: TEXT
                "calculatePct": // Whether to calculate percentage: BOOLEAN
                "timeFrame": // The description of the time point(s) of assessment must be specific to the outcome measure and is generally the specific duration of time over which each participant is assessed (not the overall duration of the study).: TEXT (max 255 chars)
                "typeUnitsAnalyzed": //  (Optional) If units are not participants (e.g., eyes, lesions).: TEXT (max 40 chars)
                "denomUnitsSelected": // Selected denominator units (e.g., Participants, eyes, lesions): TEXT
                "groups": [ // ARRAY of OBJECT - Study arms/groups
                    {{
                        "id": //Unique group identifier. OG000 is the first group, OG001 is the second, and so on: TEXT
                        "title": // Group name: TEXT
                        "description": // Group description: TEXT
                    }}
                ],
                "denoms": [ // ARRAY of OBJECT - Denominators
                    {{
                        "units": // Unit type: TEXT
                        "counts": [ // ARRAY of OBJECT
                            {{
                                "groupId": // References a group ID: TEXT
                                "value": // Count value: TEXT
                            }}
                        ]
                    }}
                ],
                "classes": [ // ARRAY of OBJECT - Outcome categories/timepoints
                    {{
                        "title": // Category/timepoint name: TEXT
                        "denoms": [ // ARRAY of OBJECT. Similar structure as above.
                            {{
                                "units": // TEXT
                                "counts": [
                                    {{
                                        "groupId": // References a group ID: TEXT
                                        "value": // TEXT
                                    }}
                                ]
                            }}
                        ],
                        "categories": [ // ARRAY of OBJECT - Subcategories
                            {{
                                "title": // Subcategory name: TEXT
                                "measurements": [ // ARRAY of OBJECT - Results for each group
                                    {{
                                        "groupId": // References a group ID: TEXT
                                        "value": // Result value: TEXT
                                        "spread": // Spread (e.g., SD, SE): TEXT
                                        "lowerLimit": // CI lower limit: TEXT
                                        "upperLimit": // CI upper limit: TEXT
                                        "comment": // Explanation for NA values: TEXT
                                    }}
                                ]
                            }}
                        ]
                    }}
                ],
                "analyses": [ // ARRAY of OBJECT - Result(s) of scientifically appropriate tests of statistical significance of the primary and secondary outcome measures, if any. Such analyses include: pre-specified in the protocol and/or statistical analysis plan; made public by the sponsor or responsible party; conducted on a primary outcome measure in response to a request made by FDA. If a statistical analysis is reported "Comparison Group Selection" and "Type of Statistical Test" are required. In addition, one of the following data elements are required with the associated information: "P-Value," "Estimation Parameter," or "Other Statistical Analysis."
                    {{
                        "paramType": // Parameter type analyzed: ENUM("Cox Proportional Hazard","Hazard Ratio (HR)","Hazard Ratio, Log","Mean Difference (Final Values)","Mean Difference (Net)","Median Difference (Final Values)","Median Difference (Net)","Odds Ratio (OR)","Odds Ratio, Log","Risk Difference (RD)","Risk Ratio (RR)","Risk Ratio, Log","Slope","Other")
                        "paramValue": // The name of the estimation parameter, if "Other" Estimation Parameter is selected.: TEXT
                        "dispersionType": // Dispersion type: ENUM ("STANDARD_DEVIATION", "STANDARD_ERROR_OF_MEAN")
                        "dispersionValue": // The calculated value for the dispersion of the estimated parameter.: TEXT
                        "statisticalMethod": // Method used: TEXT (max 150 chars)
                        "statisticalComment": // Additional comments: TEXT
                        "pValue": // Calculated p-value given the null-hypothesis: TEXT (max 250 chars)
                        "pValueComment": //  Additional information about p-value.: TEXT
                        "ciNumSides": // CI sides: ENUM ("ONE_SIDED", "TWO_SIDED")
                        "ciPctValue": // CI percentage (e.g., "95"): TEXT
                        "ciLowerLimit": // CI lower bound: TEXT
                        "ciUpperLimit": // CI upper bound: TEXT
                        "ciLowerLimitComment": // Lower limit explanation: TEXT (max 250 chars)
                        "ciUpperLimitComment": // Upper limit explanation: TEXT (max 250 chars)
                        "estimateComment": // Estimation comment: TEXT
                        "testedNonInferiority": // Non-inferiority tested: BOOLEAN
                        "nonInferiorityType": // Test type: ENUM ("SUPERIORITY", "NON_INFERIORITY", "EQUIVALENCE", "OTHER", "NON_INFERIORITY_OR_EQUIVALENCE", "SUPERIORITY_OR_OTHER", "NON_INFERIORITY_OR_EQUIVALENCE_LEGACY", "SUPERIORITY_OR_OTHER_LEGACY")
                        "nonInferiorityComment": // Test explanation: TEXT (max 999 chars)
                        "otherAnalysisDescription": // Other analysis details: TEXT (max 500 chars)
                        "groupDescription": // Comparison group details: TEXT
                        "groupIds": // Groups compared: ARRAY of TEXT
                    }}
                ]
            }}
        ]
    }}
}}
```