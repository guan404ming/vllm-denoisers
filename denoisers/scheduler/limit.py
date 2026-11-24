import math
import statistics
from collections import deque

BLOCK = 8
SAMPLES = 16


class ActiveLimit:
    def __init__(self, seqs, loss):
        self.seqs = seqs
        self.loss = loss
        self.limit = max(1, seqs // 2)
        self.costs = {1: [], 2: []}
        self.batches = deque([(0, 0)] * 2, maxlen=2)
        self.last_end = 0

    def probe(self, step, size):
        if size * 2 <= self.seqs or not self.costs:
            return 0
        return 1 + step // BLOCK % 2

    def submitted(self, size, probe):
        self.batches.append((size, probe))

    def completed(self, end):
        elapsed = end - self.last_end
        self.last_end = end
        (before, kind), (size, probe) = self.batches
        if not (self.costs and before and size and probe and probe == kind):
            return
        self.costs[probe].append((1 / size, elapsed / size))
        if min(map(len, self.costs.values())) < SAMPLES:
            return
        (wide, full), (narrow, half) = (
            map(statistics.median, zip(*self.costs[key])) for key in (1, 2)
        )
        self.costs = {}
        if narrow <= wide:
            return
        fixed = (half - full) / (narrow - wide)
        each = full - fixed * wide
        if fixed > 0 and each > 0:
            seqs, loss = self.seqs, self.loss
            limit = math.ceil((1 - loss) * seqs * fixed / (fixed + loss * each * seqs))
            self.limit = max(seqs // 4, limit)
