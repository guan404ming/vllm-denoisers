from vllm.utils.import_utils import resolve_obj_by_qualname


def resolve(table: dict, name: str):
    if name in table:
        return table[name]
    if "." not in name:
        raise ValueError(f"{name} must be one of {sorted(table)} or a qualified class name.")
    return resolve_obj_by_qualname(name)
