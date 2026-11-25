from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from denoisers import patches
from denoisers.models import families


def register() -> None:
    patches.apply()
    register_backend(
        AttentionBackendEnum.FLASH_ATTN, "denoisers.attention.RangedFlashAttentionBackend"
    )
    families.register_all()
