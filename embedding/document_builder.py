"""
document_builder.py

Converts structured JSON documents into readable text suitable for
semantic embedding models (Sentence Transformers, BGE, E5, etc.).

Author: Your Team
"""

from typing import Any


class DocumentBuilder:
    """
    Converts arbitrary JSON/dictionary objects into readable text.

    Example
    -------
    Input

    {
        "patient": {
            "age": 45,
            "gender": "Male"
        },
        "claim": {
            "status": "Approved",
            "amount": 50000
        }
    }

    Output

    Patient:
      Age: 45
      Gender: Male

    Claim:
      Status: Approved
      Amount: 50000
    """

    DEFAULT_IGNORE_FIELDS = {
        "id",
        "_id",
        "uuid",
        "row_number",
        "created_at",
        "updated_at",
        "timestamp",
    }

    def __init__(self, ignore_fields=None):
        self.ignore_fields = set(ignore_fields or [])
        self.ignore_fields.update(self.DEFAULT_IGNORE_FIELDS)

    def build(self, document: dict) -> str:
        """Convert a dictionary into readable text."""

        lines = []
        self._walk(document, lines, level=0)

        return "\n".join(lines).strip()

    # -------------------------------------------------------

    def _walk(
        self,
        obj: Any,
        lines: list[str],
        level: int,
        key_name: str | None = None,
    ):
        indent = "  " * level

        # --------------------------
        # Dictionary
        # --------------------------

        if isinstance(obj, dict):

            if key_name:
                lines.append(f"{indent}{self._pretty(key_name)}:")

            for key, value in obj.items():

                if key in self.ignore_fields:
                    continue

                self._walk(
                    value,
                    lines,
                    level + (1 if key_name else 0),
                    key,
                )

        # --------------------------
        # List
        # --------------------------

        elif isinstance(obj, list):

            if key_name:
                lines.append(f"{indent}{self._pretty(key_name)}:")

            for item in obj:

                if isinstance(item, (dict, list)):
                    self._walk(item, lines, level + 1)

                else:
                    lines.append(
                        f"{indent}  - {item}"
                    )

        # --------------------------
        # Primitive value
        # --------------------------

        else:

            if obj is None:
                value = "Unknown"
            else:
                value = obj

            lines.append(
                f"{indent}{self._pretty(key_name)}: {value}"
            )

    # -------------------------------------------------------

    @staticmethod
    def _pretty(text: str) -> str:
        """
        Convert snake_case or camelCase into readable labels.

        claim_amount
            ->
        Claim Amount

        policyType
            ->
        Policy Type
        """

        if text is None:
            return ""

        text = text.replace("_", " ")

        pretty = ""

        for i, ch in enumerate(text):

            if i > 0 and ch.isupper() and text[i - 1].islower():
                pretty += " "

            pretty += ch

        return pretty.title()


# -----------------------------------------------------------
# Example usage
# -----------------------------------------------------------

if __name__ == "__main__":

    sample = {

        "patient": {
            "age": 45,
            "gender": "Male",
            "conditions": [
                "Diabetes",
                "Hypertension"
            ]
        },

        "policy": {
            "type": "Premium",
            "sum_insured": 500000
        },

        "claim": {
            "status": "Approved",
            "amount": 120000,
            "hospital": "Apollo Hospital"
        },

        "claim_id": "CLM00123"
    }

    builder = DocumentBuilder(
        ignore_fields={"claim_id"}
    )

    print(builder.build(sample))