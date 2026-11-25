from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from denoisers import hooks
from denoisers.models import families


def register() -> None:
    hooks.apply()
    register_backend(
        AttentionBackendEnum.FLASH_ATTN, "denoisers.attention.RangedFlashAttentionBackend"
    )
    families.register_all()
