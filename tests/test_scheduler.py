import math

import pytest
import torch

from denoisers.policies.hold import Aging, FairShare, ShortestFirst
from denoisers.scheduler import probe
from denoisers.scheduler.limit import ActiveLimit
from denoisers.scheduler.remaining import RemainingWork


def test_mean_entropy_counts_only_selected_positions():
    logits = torch.zeros(1, 4, 16)
    logits[0, 2, 0] = 50.0
    counted = torch.tensor([[True, True, False, True]])
    entropy = probe.entropy(logits)
    assert math.isclose(probe.mean_entropy(entropy, counted).item(), math.log(16), rel_tol=1e-5)
    assert probe.mean_entropy(entropy, torch.ones(1, 4, dtype=torch.bool)).item() < math.log(16)


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


def test_fair_share_ranks_by_remaining_steps_plus_the_join_clock():
    policy = FairShare()
    scores = {"long": 40.0, "short": 5.0}
    waiting = [("long", 0.0), ("short", 0.0)]
    assert policy.held(waiting, 0.0, 1, scores) == {"long"}
    for _ in range(79):
        policy.held(waiting, 0.0, 1, scores)
    assert policy.clock == 40.0
    scores["new"] = 5.0
    arrived = [*waiting, ("new", 0.0)]
    assert policy.rank(arrived, 0.0, scores) == ["short", "long", "new"]
    assert policy.held(arrived, 0.0, 1, scores) == {"long", "new"}
    assert policy.held([("new", 0.0)], 0.0, 1, scores) == set() and list(policy.joined) == ["new"]


def run_steps(limit, steps, cost):
    for size, probe_kind in steps:
        limit.submitted(size, probe_kind)
        limit.completed(limit.last_end + cost(size))


def test_active_limit_calibrates_once_from_full_and_half_steps():
    limit = ActiveLimit(64, 0.05)
    assert limit.limit == 32
    assert [limit.probe(step, 64) for step in (0, 7, 8, 16)] == [1, 1, 2, 1]
    assert limit.probe(0, 32) == 0
    blocks = [(64, 1)] * 9 + [(32, 2)] * 9
    run_steps(limit, blocks, lambda size: 20 + 5 * size)
    assert limit.limit == 32 and limit.probe(0, 64) == 1
    run_steps(limit, blocks, lambda size: 20 + 5 * size)
    assert limit.limit == 34 and limit.probe(0, 64) == 0
    assert 34 / (20 + 5 * 34) >= 0.95 * 64 / (20 + 5 * 64) > 33 / (20 + 5 * 33)
    run_steps(limit, blocks * 2, lambda size: 1 + 5 * size)
    assert limit.limit == 34


def test_active_limit_times_the_latest_batch_and_ignores_stalls():
    limit = ActiveLimit(64, 0.05)
    run_steps(limit, [(32, 0), (32, 0), (64, 1), (0, 1), (64, 1)], lambda size: 300)
    assert limit.costs == {1: [], 2: []}
    run_steps(limit, [(64, 1)], lambda size: 320)
    assert limit.costs[1] == [(1 / 64, 5.0)]
    cost = iter([10**9] + [20 + 5 * 64] * 20 + [20 + 5 * 32] * 17)
    run_steps(limit, [(64, 1)] * 21 + [(32, 2)] * 17, lambda size: next(cost))
    assert limit.limit == 34
    floor = ActiveLimit(64, 0.05)
    run_steps(floor, [(64, 1)] * 17 + [(32, 2)] * 17, lambda size: 1 + 5 * size)
    assert floor.limit == 16


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
    assert work.future == {(0, False): 20.0, (1, False): 8.0, (2, True): 0.0}
    assert work.estimate("early", 0, 0, (9.0, False)) == 29.0
    assert work.estimate("late", 1, 12, (9.0, False)) == 17.0
    assert work.estimate("ending", 2, 20, (9.0, True)) == 9.0
    work.estimate("aborted", 0, 0, (5.0, False))
    work.finished("aborted", 3, False)
    assert work.future[0, False] == 20.0 and "aborted" not in work.blocks
