"""Phase 2 tool definitions: Cockcroft-Gault calculator and OpenFDA drug search."""

from collections.abc import Callable
from typing import Any

import requests
import numexpr

# --- Tool 1: Cockcroft-Gault Creatinine Clearance Calculator ---

def calculate_creatinine_cockcroft(
    age: int,
    weight_kg: float,
    scr: float,
    sex: str,
) -> dict[str, Any]:
    """Calculate creatinine clearance (CrCl) using the Cockcroft-Gault equation.

    Args:
        age: Patient age in years.
        weight_kg: Patient weight in kilograms.
        scr: Serum creatinine in mg/dL.
        sex: Patient biological sex, 'male' or 'female'.

    Returns:
        Dict with 'creatinine_clearance_ml_min' and 'unit', or 'error' on failure.
    """
    sex_lower = str(sex).strip().lower()
    if sex_lower not in ("male", "female"):
        return {"error": f"Invalid sex '{sex}'. Must be 'male' or 'female'."}
    if float(scr) <= 0:
        return {"error": f"Serum creatinine must be positive, got {scr}."}
    if int(age) <= 0 or float(weight_kg) <= 0:
        return {"error": "Age and weight must be positive values."}

    age_i, weight_f, scr_f = int(age), float(weight_kg), float(scr)
    crcl = ((140 - age_i) * weight_f) / (72 * scr_f)
    if sex_lower == "female":
        crcl *= 0.85

    return {"creatinine_clearance_ml_min": round(crcl, 2), "unit": "mL/min"}


# --- Tool 2: OpenFDA Drug Search ---

_FDA_BASE_URL = "https://api.fda.gov/drug/label.json"
_FDA_TIMEOUT = 10


def fda_drug_search(drug_name: str) -> dict[str, Any]:
    """Search the OpenFDA API for drug labeling information.

    Args:
        drug_name: Brand or generic drug name to search.

    Returns:
        Dict with brand_name, generic_name, indications_and_usage, warnings,
        and dosage_and_administration fields, or 'error' on failure.
    """
    drug_name = str(drug_name).strip()
    if not drug_name:
        return {"error": "Drug name must not be empty."}

    try:
        resp = requests.get(
            _FDA_BASE_URL,
            params={"search": f'openfda.brand_name:"{drug_name}"', "limit": 1},
            timeout=_FDA_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": f"FDA API request timed out for '{drug_name}'."}
    except requests.exceptions.RequestException as exc:
        return {"error": f"FDA API request failed: {exc}"}

    data = resp.json()
    results = data.get("results")
    if not results:
        return {"error": f"No FDA results found for '{drug_name}'."}

    record = results[0]
    openfda = record.get("openfda", {})

    def _first(field: str, max_len: int = 500) -> str:
        val = record.get(field, [])
        text = val[0] if val else "Not available"
        return text[:max_len]

    return {
        "brand_name": ", ".join(openfda.get("brand_name", ["Not available"])),
        "generic_name": ", ".join(openfda.get("generic_name", ["Not available"])),
        "indications_and_usage": _first("indications_and_usage"),
        "warnings": _first("warnings"),
        "dosage_and_administration": _first("dosage_and_administration"),
    }


# --- Tool 3: Math operations ---

def math_operation(operation: str) -> dict[str, Any]:
    """Tool to perform math operations and calculations

    Args:
        operation: A string containing a valid mathematical expression.

    Returns:
        dict[str, Any]: Result of the calculation or error message.
    """
    try:
        result = numexpr.evaluate(operation).item()
        return {"result": result}
    except Exception as exc:
        return {"error": f"Math operation failed: {exc}"}

# --- Registry ---

AVAILABLE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate_creatinine_cockcroft": calculate_creatinine_cockcroft,
    "fda_drug_search": fda_drug_search,
    "math_operation": math_operation,
}

# --- JSON Schemas for Qwen2.5 apply_chat_template(tools=...) ---

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_creatinine_cockcroft",
            "description": (
                "Calculate estimated creatinine clearance (kidney function) "
                "using the Cockcroft-Gault equation. Use this for questions about "
                "GFR, kidney function, or creatinine clearance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "Patient age in years",
                    },
                    "weight_kg": {
                        "type": "number",
                        "description": "Patient weight in kilograms",
                    },
                    "scr": {
                        "type": "number",
                        "description": "Serum creatinine level in mg/dL",
                    },
                    "sex": {
                        "type": "string",
                        "enum": ["male", "female"],
                        "description": "Patient biological sex",
                    },
                },
                "required": ["age", "weight_kg", "scr", "sex"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fda_drug_search",
            "description": (
                "Search the OpenFDA database for drug labeling information "
                "including indications, warnings, and dosage. Use this for "
                "questions about medications, drug side effects, or drug interactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The brand or generic name of the drug",
                    },
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math_operation",
            "description": (
                "Perform a mathematical operation given as a string expression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "A string containing a valid mathematical expression.",
                    },
                },
                "required": ["operation"],
            },
        },
    },
]
