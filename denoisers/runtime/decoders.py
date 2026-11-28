import logging

import numpy as np

from denoisers.policies.decode import DecodePolicy, registry

logger = logging.getLogger("vllm.denoisers")

STRUCTURE = ("aligned", "causal_prefill", "causal_commit", "shifted")


class Decoders:
    def __init__(self, default: DecodePolicy, build, state, max_num_reqs: int):
        self.build = build
        self.state = state
        self.policies = [default]
        self.index: dict[str | None, int] = {None: 0}
        self.ids = np.zeros(max_num_reqs, dtype=np.int64)
        default.bind(state)

    @property
    def default(self) -> DecodePolicy:
        return self.policies[0]

    def _create(self, name: str) -> int | None:
        if self.build is None:
            logger.warning("Per-request decode_policy is off; ignoring %r.", name)
            return None
        if name not in registry.entries:
            logger.warning("Unknown decode_policy %r; using the default.", name)
            return None
        try:
            policy = self.build(registry.entries[name])
        except Exception as error:
            logger.warning("decode_policy %r failed to build (%s); using the default.", name, error)
            return None
        mismatch = [a for a in STRUCTURE if getattr(policy, a) != getattr(self.default, a)]
        if mismatch:
            logger.warning("decode_policy %r differs in %s; using the default.", name, mismatch)
            return None
        policy.dense_scores = self.default.dense_scores
        policy.bind(self.state)
        self.policies.append(policy)
        return len(self.policies) - 1

    def assign(self, slot: int, sampling_params) -> DecodePolicy:
        name = (getattr(sampling_params, "extra_args", None) or {}).get("decode_policy")
        if name not in self.index:
            created = self._create(name)
            self.index[name] = 0 if created is None else created
        self.ids[slot] = self.index[name]
        return self.policies[self.ids[slot]]

    def of(self, slot: int) -> DecodePolicy:
        return self.policies[self.ids[slot]]

    def groups(self, slots_np: np.ndarray) -> list[tuple[DecodePolicy, np.ndarray | None]]:
        ids = self.ids[slots_np]
        if (ids == ids[0]).all():
            return [(self.policies[ids[0]], None)]
        return [(self.policies[i], np.flatnonzero(ids == i)) for i in np.unique(ids)]

    def all(self, slots_np: np.ndarray, attr: str) -> bool:
        return all(getattr(self.policies[i], attr) for i in np.unique(self.ids[slots_np]))

    def flags(self, slots_np: np.ndarray, attr: str) -> np.ndarray:
        values = np.array([bool(getattr(p, attr)) for p in self.policies])
        return values[self.ids[slots_np]]
