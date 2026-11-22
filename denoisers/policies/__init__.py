from denoisers.policies.dream import Dream
from denoisers.policies.entropy_bound import EntropyBound
from denoisers.policies.fast_dllm import FastDLLM
from denoisers.policies.indel import Indel
from denoisers.policies.joint import JointThreshold
from denoisers.policies.llada import LLaDA
from denoisers.policies.threshold import LowConfidence

POLICIES = {
    "low_confidence": LowConfidence,
    "joint": JointThreshold,
    "indel": Indel,
    "fast_dllm": FastDLLM,
    "entropy_bound": EntropyBound,
    "dream": Dream,
    "llada": LLaDA,
}


def build(name: str, config):
    return POLICIES[name].from_config(config)


__all__ = [
    "POLICIES",
    "Dream",
    "EntropyBound",
    "FastDLLM",
    "Indel",
    "JointThreshold",
    "LLaDA",
    "LowConfidence",
    "build",
]
