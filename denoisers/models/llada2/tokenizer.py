from collections.abc import Iterator

from transformers import PreTrainedTokenizerFast
from vllm.logger import init_logger

from denoisers.models.llada2.tool_declaration_ts import encode_tools_to_typescript_style

logger = init_logger("vllm.denoisers")

MAX_ENCODE_CHARS = 400_000
MAX_RUN_CHARS = 25_000


def sorted_keys(value):
    if isinstance(value, dict):
        return {key: sorted_keys(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [sorted_keys(item) for item in value]
    return value


def split_runs(text: str, limit: int) -> Iterator[str]:
    length = 0
    space = text[:1].isspace()
    start = 0
    for index, char in enumerate(text):
        if char.isspace() != space:
            length = 1
            space = not space
            continue
        length += 1
        if length > limit:
            yield text[start:index]
            start = index
            length = 1
    yield text[start:]


class CustomFastTokenizer(PreTrainedTokenizerFast):
    def encode(self, text, *args, **kwargs) -> list[int]:
        if not isinstance(text, str) or args or kwargs:
            return super().encode(text, *args, **kwargs)
        ids = []
        for start in range(0, len(text), MAX_ENCODE_CHARS):
            for piece in split_runs(text[start : start + MAX_ENCODE_CHARS], MAX_RUN_CHARS):
                ids.extend(super().encode(piece, add_special_tokens=False))
        return ids

    def apply_chat_template(self, conversation, tools=None, **kwargs):
        tools = sorted_keys(tools)
        if tools:
            try:
                kwargs["tools_ts_str"] = encode_tools_to_typescript_style(tools)
            except Exception as error:
                logger.error("Failed to convert tools to TypeScript style: %s", error)
        return super().apply_chat_template(conversation=conversation, tools=tools, **kwargs)
