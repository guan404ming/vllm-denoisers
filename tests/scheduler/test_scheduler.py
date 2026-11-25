import math

import pytest
import torch

from denoisers import probe
from denoisers.scheduler import aligned_prefill_chunk
from denoisers.scheduler.hold import Aging, ShortestFirst
from denoisers.scheduler.remaining import RemainingWork


def test_aligned_prefill_chunk():
    assert aligned_prefill_chunk(0, 48, 100, 32) == 32
    assert aligned_prefill_chunk(32, 48, 100, 32) == 32
    assert aligned_prefill_chunk(64, 48, 100, 32) == 48
    assert aligned_prefill_chunk(0, 20, 100, 32) == 0
    assert aligned_prefill_chunk(96, 4, 100, 32) == 4
    assert aligned_prefill_chunk(100, 32, 100, 32) == 32


def test_score_table_crosses_processes_and_drops_committing_rows(tmp_path):
    path = str(tmp_path / "scores")
    reader = probe.ScoreTable(path, 4)
    assert reader.read(["a"]) == {}
    writer = probe.Probe(probe.ScoreTable(path, 4, create=True))
    idle = torch.tensor([False])
    writer.record(
        [2, 0],
        ["a", "b"],
        torch.tensor([1.0, 2.0]),
        torch.tensor([False, True]),
        torch.tensor([True, False]),
    )
    assert reader.read(["a", "b", "z"]) == {}
    writer.record([2], ["c"], torch.tensor([3.0]), idle, idle)
    assert reader.read(["a", "b", "z"]) == {"a": (1.0, True)}
    empty = torch.empty(0, dtype=torch.bool)
    writer.record([], [], torch.empty(0), empty, empty)
    assert reader.read(["a", "c"]) == {"c": (3.0, False)}
    writer.table.close()
    assert not (tmp_path / "scores").exists()


def test_sjf_holds_highest_scores_and_prefers_unscored():
    scores = {"a": 3.0, "b": 1.0, "c": 2.0}
    candidates = [("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0)]
    assert ShortestFirst().held(candidates, 10.0, 2, scores) == {"a", "c"}


def test_aging_lets_old_requests_overtake():
    scores = {"old": 5.0, "new": 1.0}
    candidates = [("old", 0.0), ("new", 9.0)]
    assert Aging(1.0).held(candidates, 10.0, 1, scores) == {"new"}
    assert ShortestFirst().held(candidates, 10.0, 1, scores) == {"old"}
    with pytest.raises(ValueError):
        Aging(0.0)


def test_reserve_runs_the_oldest_held_requests():
    scores = {"a": 1.0, "b": 2.0, "old": 9.0, "new": 8.0}
    candidates = [("a", 5.0), ("b", 6.0), ("old", 0.0), ("new", 7.0)]
    assert ShortestFirst().held(candidates, 10.0, 3, scores) == {"old"}
    assert ShortestFirst().held(candidates, 10.0, 3, scores, 1) == {"new"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_probe_waits_for_scores_on_the_copy_stream(tmp_path):
    table = probe.ScoreTable(str(tmp_path / "scores"), 1, create=True)
    writer = probe.Probe(table)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scores = torch.tensor([2.0], device="cuda")
        committing = torch.tensor([False], device="cuda")
        torch.cuda._sleep(20_000_000)
        writer.record([0], ["first"], scores, committing, committing)
        writer.record([0], ["second"], scores, committing, committing)
    assert table.read(["first"]) == {"first": (2.0, False)}


def test_new_block_gets_one_probe_step_then_the_mean_score():
    work = RemainingWork()
    assert work.estimate("scored", 0, 0, (6.0, False)) == 6.0
    assert work.estimate("new", 0, 0, None) == -math.inf
    work.probing("new", 0)
    assert work.estimate("new", 0, 1, None) == 6.0
    assert work.estimate("new", 1, 9, None) == -math.inf


def test_finished_requests_teach_the_steps_left_after_each_block():
    work = RemainingWork()
    work.estimate("long", 0, 0, (5.0, False))
    work.estimate("long", 1, 10, (5.0, False))
    work.estimate("long", 2, 22, (5.0, True))
    work.finished("long", 30, True)
    assert work.future[0, False, None] == 20.0 and work.future[1, False, None] == 8.0
    assert work.future[2, True, None] == 0.0
    assert work.estimate("early", 0, 0, (9.0, False)) == 29.0
    assert work.estimate("late", 1, 12, (9.0, False)) == 17.0
    assert work.estimate("ending", 2, 20, (9.0, True)) == 9.0
    work.estimate("aborted", 0, 0, (5.0, False))
    work.finished("aborted", 3, False)
    assert work.future[0, False, None] == 20.0 and "aborted" not in work.blocks


def test_the_first_score_of_a_block_selects_its_learned_future():
    work = RemainingWork()
    for rid, score, steps in (("short", 1.5, 4), ("long", 3.5, 40)):
        work.estimate(rid, 0, 0, (score, False))
        work.estimate(rid, 1, 2, (score, False))
        work.finished(rid, steps, True)
    assert work.estimate("a", 0, 0, (1.2, False)) == 1.2 + 2.0
    assert work.estimate("b", 0, 0, (3.9, False)) == 3.9 + 38.0
    assert work.estimate("b", 0, 1, (1.2, False)) == 1.2 + 38.0
    assert work.estimate("unseen", 0, 0, (7.0, False)) == 7.0 + 20.0
