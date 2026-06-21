"""Offline guard: the /narrative response schema must satisfy OpenAI strict
structured-outputs rules — WITHOUT calling the API. Catches the class of bug
the mocked eval/unit tests miss (open dicts, tuple prefixItems, etc.)."""
from backend.rag_narrative import _response_format_json_schema

_UNSUPPORTED_ARRAY_KEYS = {"minItems", "maxItems", "prefixItems", "minLength", "maxLength"}


def _walk(node, path="root", errs=None):
    errs = errs if errs is not None else []
    if isinstance(node, dict):
        if node.get("type") == "object":
            if node.get("additionalProperties") is not False:
                errs.append(f"{path}: additionalProperties must be False (got {node.get('additionalProperties')!r})")
        for k in _UNSUPPORTED_ARRAY_KEYS & node.keys():
            errs.append(f"{path}: unsupported strict keyword {k!r}")
        for k, v in node.items():
            _walk(v, f"{path}.{k}", errs)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk(v, f"{path}[{i}]", errs)
    return errs


def test_narrative_response_schema_is_openai_strict_valid():
    rf = _response_format_json_schema()
    assert rf["json_schema"]["strict"] is True
    errs = _walk(rf["json_schema"]["schema"])
    assert not errs, "strict-schema violations:\n  " + "\n  ".join(errs)
