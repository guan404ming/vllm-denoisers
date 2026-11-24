from denoisers.policies.hold.aging import Aging
from denoisers.policies.hold.fair import FairShare
from denoisers.policies.hold.interface import PRIOR, HoldPolicy, registry
from denoisers.policies.hold.random_order import RandomOrder
from denoisers.policies.hold.sjf import ShortestFirst

register = registry.register

__all__ = [
    "PRIOR",
    "Aging",
    "FairShare",
    "HoldPolicy",
    "RandomOrder",
    "ShortestFirst",
    "register",
    "registry",
]
