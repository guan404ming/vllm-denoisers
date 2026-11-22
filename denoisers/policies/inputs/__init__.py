from denoisers.policies.inputs.interface import InputPolicy, registry
from denoisers.policies.inputs.self_conditioning import SelfConditioning
from denoisers.policies.inputs.smooth import Smooth

register = registry.register

__all__ = ["InputPolicy", "SelfConditioning", "Smooth", "register", "registry"]
