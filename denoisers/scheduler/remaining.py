import math
from collections import Counter


class RemainingWork:
    def __init__(self):
        self.future = {}
        self.counts = Counter()
        self.blocks = {}
        self.probed = {}
        self.score = 0.0
        self.scored = 0

    def estimate(self, rid, block, steps, probe):
        marks = self.blocks.setdefault(rid, [])
        while len(marks) <= block:
            marks.append([steps, False])
        if probe is None:
            if self.probed.get(rid) != block:
                return -math.inf
            score = self.score
        else:
            score, stop = probe
            marks[block][1] |= stop
            self.scored += 1
            self.score += (score - self.score) / self.scored
        return score + self.future.get((block, marks[block][1]), 0.0)

    def probing(self, rid, block):
        self.probed[rid] = block

    def finished(self, rid, steps, learn):
        marks = self.blocks.pop(rid, [])
        self.probed.pop(rid, None)
        if not learn:
            return
        ends = [start for start, _ in marks[1:]] + [steps]
        for block, ((_, stop), end) in enumerate(zip(marks, ends)):
            cell = (block, stop)
            self.counts[cell] += 1
            mean = self.future.get(cell, 0.0)
            self.future[cell] = mean + (steps - end - mean) / self.counts[cell]
