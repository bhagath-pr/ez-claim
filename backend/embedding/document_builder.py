from typing import Any

class DocumentBuilder:
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
        lines = []
        self._walk(document, lines, level=0)
        return "\n".join(lines).strip()

    def _walk(
        self,
        obj: Any,
        lines: list[str],
        level: int,
        key_name: str | None = None,
    ):
        indent = "  " * level

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

        elif isinstance(obj, list):
            if key_name:
                lines.append(f"{indent}{self._pretty(key_name)}:")

            for item in obj:
                if isinstance(item, (dict, list)):
                    self._walk(item, lines, level + 1)
                else:
                    lines.append(f"{indent}  - {item}")

        else:
            value = "Unknown" if obj is None else obj
            lines.append(f"{indent}{self._pretty(key_name)}: {value}")

    @staticmethod
    def _pretty(text: str) -> str:
        if text is None:
            return ""

        text = text.replace("_", " ")
        pretty = ""

        for i, ch in enumerate(text):
            if i > 0 and ch.isupper() and text[i - 1].islower():
                pretty += " "
            pretty += ch

        return pretty.title()
