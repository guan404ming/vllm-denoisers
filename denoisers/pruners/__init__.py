from denoisers.pruners.interface import RowPruner
from denoisers.pruners.masked import MaskedRows

PRUNERS: dict[str, type[RowPruner]] = {"masked": MaskedRows}


def register_pruner(name: str):
    def wrapper(cls: type[RowPruner]) -> type[RowPruner]:
        if not issubclass(cls, RowPruner):
            raise ValueError("A row pruner must subclass RowPruner.")
        PRUNERS[name] = cls
        return cls

    return wrapper


__all__ = ["PRUNERS", "MaskedRows", "RowPruner", "register_pruner"]
