from denoisers.hooks import cudagraphs, draft_tokens


def apply() -> None:
    cudagraphs.apply()
    draft_tokens.apply()
