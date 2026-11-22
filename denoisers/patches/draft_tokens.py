from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

provider = None

_set_draft_tokens = DraftTokensHandler.set_draft_tokens


def set_draft_tokens(self, input_batch, draft_tokens) -> None:
    _set_draft_tokens(self, input_batch, draft_tokens)
    if provider is None:
        return
    for req_id, width in zip(input_batch.req_ids, provider(input_batch).tolist()):
        drafts = self._draft_tokens_by_req[req_id]
        self._draft_tokens_by_req[req_id] = (
            min(drafts, width) if isinstance(drafts, int) else drafts[:width]
        )


def apply() -> None:
    DraftTokensHandler.set_draft_tokens = set_draft_tokens
