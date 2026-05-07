import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = (
    ROOT
    / "scripts"
    / "evaluator"
    / "evaluate_utils"
    / "bfcl_pkg"
    / "bfcl"
    / "constants"
    / "default_prompts.py"
)


def _load_default_prompts_namespace():
    namespace = {}
    exec(DEFAULT_PROMPTS.read_text(), namespace)
    return namespace


def test_jsonschema_system_prompt_renders_valid_json_example():
    namespace = _load_default_prompts_namespace()

    rendered = namespace["JSONSCHEMA_SYSTEM_PROMPT"].format(functions="[]")

    assert "{{" not in rendered
    assert "}}" not in rendered

    match = re.search(r"```json\n(.*?)\n```", rendered, re.S)
    assert match is not None
    json.loads(match.group(1))
