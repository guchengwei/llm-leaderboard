import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


if "mistralai" not in sys.modules:
    mistralai_stub = types.ModuleType("mistralai")
    mistralai_stub.Mistral = object
    sys.modules["mistralai"] = mistralai_stub

from llm_inference_adapter import (
    _parse_tool_call_arguments,
)


def test_parse_tool_call_arguments_accepts_valid_json():
    parsed = _parse_tool_call_arguments('{"city": "Tokyo", "days": 3}')
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_repairs_missing_closing_brace():
    parsed = _parse_tool_call_arguments('{"city": "Tokyo", "days": 3')
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_repairs_code_fence_and_trailing_comma():
    parsed = _parse_tool_call_arguments(
        '```json\n{"city": "Tokyo", "days": 3,}\n```'
    )
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_repairs_missing_comma_between_fields():
    parsed = _parse_tool_call_arguments(
        '{"city": "Tokyo" "days": 3}'
    )
    assert parsed == {"city": "Tokyo", "days": 3}


def test_parse_tool_call_arguments_accepts_python_literal_style_dict():
    parsed = _parse_tool_call_arguments(
        "{'city': 'Tokyo', 'weekend': True, 'note': None}"
    )
    assert parsed == {"city": "Tokyo", "weekend": True, "note": None}


def test_parse_tool_call_arguments_raises_on_irreparable_input():
    try:
        _parse_tool_call_arguments('{"city": Tokyo ???')
    except json.JSONDecodeError:
        return
    assert False, "Expected JSONDecodeError for irreparable tool call arguments"
