import hashlib
import json
import random
from collections import defaultdict


def prompt_key(prompt_token_ids) -> str:
    return hashlib.blake2b(bytes(str(list(prompt_token_ids)), "utf8"), digest_size=8).hexdigest()


class Trace:
    def __init__(self, path: str):
        self.file = open(path, "a")

    def commit(self, key: str, block: int, steps: int, out: int, prompt: int) -> None:
        row = {"key": key, "block": block, "steps": steps, "out": out, "prompt": prompt}
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()


def load_trace(path: str) -> dict[str, dict]:
    blocks: dict[str, dict[int, int]] = defaultdict(dict)
    out: dict[str, int] = {}
    prompt: dict[str, int] = {}
    for line in open(path):
        row = json.loads(line)
        blocks[row["key"]][row["block"]] = row["steps"]
        out[row["key"]] = max(out.get(row["key"], 0), row["out"])
        prompt[row["key"]] = row["prompt"]
    return {
        k: {"steps": [blocks[k][i] for i in sorted(blocks[k])], "out": out[k], "prompt": prompt[k]}
        for k in blocks
    }


class Oracle:
    def __init__(self, path: str):
        self.trace = load_trace(path)
        steps = [s for v in self.trace.values() for s in v["steps"]]
        self.mean_steps = sum(steps) / max(len(steps), 1)

    def remaining(self, key: str, block: int, elapsed: int) -> float:
        steps = self.trace.get(key, {}).get("steps", [])
        current = steps[block] if block < len(steps) else self.mean_steps
        later = sum(steps[block + 1 :]) if block < len(steps) else 0.0
        return max(current - elapsed, 1.0) + later


class Regressor:
    def __init__(self, path: str):
        trace = load_trace(path)
        xs = [v["prompt"] for v in trace.values()]
        ys = [v["out"] for v in trace.values()]
        n = max(len(xs), 1)
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        self.slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
        self.bias = my - self.slope * mx
        steps = [s for v in trace.values() for s in v["steps"]]
        self.mean_steps = sum(steps) / max(len(steps), 1)

    def remaining(self, prompt_len: int, output_tokens: int, canvas: int, elapsed: int) -> float:
        predicted = self.bias + self.slope * prompt_len
        blocks_left = max((predicted - output_tokens) / canvas, 1.0)
        return max(blocks_left * self.mean_steps - elapsed, 1.0)


def random_scores(keys: list[str], low: float, high: float) -> dict[str, float]:
    return {k: random.uniform(low, high) for k in keys}
