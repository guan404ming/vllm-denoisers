from denoisers.policies.dream import Dream
from denoisers.policies.entropy_bound import EntropyBound
from denoisers.policies.fast_dllm import FastDLLM
from denoisers.policies.indel import Indel
from denoisers.policies.interface import DecodePolicy, ScoreMovePolicy
from denoisers.policies.joint import JointThreshold
from denoisers.policies.llada import LLaDA
from denoisers.policies.threshold import LowConfidence

POLICIES: dict[str, type[DecodePolicy]] = {
    "low_confidence": LowConfidence,
    "joint": JointThreshold,
    "indel": Indel,
    "fast_dllm": FastDLLM,
    "entropy_bound": EntropyBound,
    "dream": Dream,
    "llada": LLaDA,
}


def register_policy(name: str):
    def wrapper(cls: type[DecodePolicy]) -> type[DecodePolicy]:
        if not issubclass(cls, DecodePolicy):
            raise ValueError("A decode policy must subclass DecodePolicy.")
        POLICIES[name] = cls
        return cls

    return wrapper


__all__ = [
    "POLICIES",
    "DecodePolicy",
    "Dream",
    "EntropyBound",
    "FastDLLM",
    "Indel",
    "JointThreshold",
    "LLaDA",
    "LowConfidence",
    "ScoreMovePolicy",
    "register_policy",
]
