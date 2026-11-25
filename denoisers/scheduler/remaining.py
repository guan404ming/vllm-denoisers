import math
from collections import Counter

WIDTH = 1.0


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
            marks.append([steps, False, None])
        if probe is None:
            if self.probed.get(rid) != block:
                return -math.inf
            score = self.score
        else:
            score, stop = probe
            marks[block][1] |= stop
            if marks[block][2] is None:
                marks[block][2] = int(score / WIDTH)
            self.scored += 1
            self.score += (score - self.score) / self.scored
        _, stop, bucket = marks[block]
        coarse = self.future.get((block, stop, None), 0.0)
        return score + self.future.get((block, stop, bucket), coarse)

    def probing(self, rid, block):
        self.probed[rid] = block

    def finished(self, rid, steps, learn):
        marks = self.blocks.pop(rid, [])
        self.probed.pop(rid, None)
        if not learn:
            return
        ends = [mark[0] for mark in marks[1:]] + [steps]
        for block, ((_, stop, bucket), end) in enumerate(zip(marks, ends)):
            for cell in {(block, stop, bucket), (block, stop, None)}:
                self.counts[cell] += 1
                mean = self.future.get(cell, 0.0)
                self.future[cell] = mean + (steps - end - mean) / self.counts[cell]
