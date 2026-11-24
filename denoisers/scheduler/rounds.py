from collections import Counter


class DecodeTimings:
    def __init__(self):
        self.counts = Counter()
        self.cost = {}
        self.pending = {}
        self.last_end = 0
        self.last_size = 0

    def submitted(self, output, size, start):
        self.pending[id(output)] = (size, start)

    def completed(self, output, end):
        size, start = self.pending.pop(id(output), (0, end))
        if size and (start >= self.last_end or size == self.last_size):
            elapsed = end - max(start, self.last_end)
            if elapsed > 0:
                self.counts[size] += 1
                mean = self.cost.get(size, 0)
                self.cost[size] = mean + (elapsed - mean) / self.counts[size]
        self.last_end = end
        self.last_size = size

    def partition(self, size):
        full = self.cost.get(size)
        widths = sorted((k for k in self.cost if k < size), reverse=True)
        if full is None or all(self.cost[k] * size >= full * k for k in widths):
            return (size,)

        best = [(float("inf"), 0)] * (size + 1)
        first = [0] * (size + 1)
        best[0] = (0, 0)
        best[size] = (full, 1)
        first[size] = size
        for total in range(1, size + 1):
            for k in widths:
                if k <= total:
                    cost, groups = best[total - k]
                    candidate = (cost + self.cost[k], groups + 1)
                    if candidate < best[total]:
                        best[total] = candidate
                        first[total] = k
        parts = []
        while size:
            k = first[size]
            parts.append(k)
            size -= k
        return tuple(sorted(parts, reverse=True))
