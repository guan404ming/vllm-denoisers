import pytest

pytest.importorskip("vllm")

from tokenizers import Tokenizer, models, pre_tokenizers

from denoisers.models.llada2 import tokenizer as llada2_tokenizer
from denoisers.models.llada2.tokenizer import CustomFastTokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {"enum": ["c", "f"], "default": "c"},
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]
DECLARATION = """# Tools

## functions
namespace functions {
// Get the weather.
type get_weather = (_: {
  // City name
  city: string,
  // Default: "c"
  unit?: "c" | "f"
}) => any;
}
"""


def make_tokenizer():
    core = Tokenizer(models.WordLevel({"[UNK]": 0, "aa": 1, "b": 2}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    return CustomFastTokenizer(tokenizer_object=core, chat_template="{{ tools_ts_str }}")


def test_encode_splits_long_runs(monkeypatch):
    monkeypatch.setattr(llada2_tokenizer, "MAX_RUN_CHARS", 2)
    tokenizer = make_tokenizer()
    assert tokenizer.encode("aaaa b") == [1, 1, 2]
    assert tokenizer.encode("aaaa b", add_special_tokens=False) == [0, 2]


def test_tools_render_as_typescript_declarations():
    messages = [{"role": "user", "content": "b"}]
    text = make_tokenizer().apply_chat_template(messages, tools=TOOLS, tokenize=False)
    assert text == DECLARATION
