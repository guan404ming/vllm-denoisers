import importlib


class Registry:
    def __init__(self, kind: str, base: type):
        self.kind = kind
        self.base = base
        self.entries: dict[str, type] = {}

    def register(self, name: str):
        def wrapper(cls: type) -> type:
            self.entries[name] = self._check(cls)
            return cls

        return wrapper

    def resolve(self, name: str) -> type:
        if name in self.entries:
            return self.entries[name]
        module, _, attr = name.rpartition(".")
        if not module:
            raise ValueError(
                f"Unknown {self.kind} {name!r}: use one of {sorted(self.entries)} "
                "or a qualified class name."
            )
        return self._check(getattr(importlib.import_module(module), attr))

    def _check(self, cls) -> type:
        if not (isinstance(cls, type) and issubclass(cls, self.base)):
            raise ValueError(f"A {self.kind} must subclass {self.base.__name__}.")
        return cls
