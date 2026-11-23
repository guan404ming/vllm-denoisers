import math
from types import SimpleNamespace

import pytest
import torch

from denoisers.policies.hold import Aging, ShortestFirst
from denoisers.scheduler import probe
from denoisers.scheduler.active import WINDOW, DecodeTimings, DynamicActive


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
    writer.record([2, 0], ["a", "b"], torch.tensor([1.0, 2.0]), torch.tensor([False, True]))
    assert reader.read(["a", "b", "z"]) == {}
    writer.record([2], ["c"], torch.tensor([3.0]), torch.tensor([False]))
    assert reader.read(["a", "b", "z"]) == {"a": 1.0}
    writer.record([], [], torch.empty(0), torch.empty(0, dtype=torch.bool))
    assert reader.read(["a", "c"]) == {"c": 3.0}
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_probe_waits_for_scores_on_the_copy_stream(tmp_path):
    table = probe.ScoreTable(str(tmp_path / "scores"), 1, create=True)
    writer = probe.Probe(table)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scores = torch.tensor([2.0], device="cuda")
        committing = torch.tensor([False], device="cuda")
        torch.cuda._sleep(20_000_000)
        writer.record([0], ["first"], scores, committing)
        writer.record([0], ["second"], scores, committing)
    assert table.read(["first"]) == {"first": 2.0}


def observe(timings, size, duration, steps=WINDOW):
    for _ in range(steps):
        step = object()
        start = timings.last_completion
        timings.submitted(step, size, start)
        timings.completed(step, start + duration)


def test_limit_learns_local_costs_and_rechecks_the_full_batch():
    timings = DecodeTimings()
    observe(timings, 6, 6)
    observe(timings, 11, 12, WINDOW - 1)
    assert timings.target(11) == 11
    observe(timings, 11, 12, 1)
    assert [timings.target(11) for _ in range(WINDOW)] == [6] * (WINDOW - 1) + [11]
    observe(timings, 11, 8)
    assert timings.target(11) == 11
    observe(timings, 6, 4)
    assert timings.target(11) == 6


def test_overlap_uses_completion_spacing_only_for_matching_batches():
    timings = DecodeTimings()
    a, b, c, prefill = (object() for _ in range(4))
    timings.submitted(a, 6, 1)
    timings.submitted(b, 6, 2)
    timings.submitted(c, 11, 3)
    timings.completed(a, 10)
    timings.completed(b, 14)
    timings.completed(c, 20)
    timings.completed(prefill, 25)
    assert list(timings.samples[6]) == [9, 4]
    assert 11 not in timings.samples
    assert not timings.pending


def test_dynamic_active_protects_service_and_block_progress():
    requests = [
        SimpleNamespace(request_id=str(i), arrival_time=10, num_output_tokens=0) for i in range(10)
    ]
    limit = DynamicActive()
    policy = ShortestFirst()
    observe(limit.timings, 8, 8)
    observe(limit.timings, 10, 10)
    requests[9].arrival_time = 6
    assert not limit.held(requests, False, policy, {}, 0, 10)
    requests[8].num_output_tokens = 32
    assert limit.held(requests, True, policy, {}, 0, 10) == {"7", "8"}
    limit.record_scheduled([str(i) for i in range(7)] + ["9"], 0.11)
    assert limit.held(requests, True, policy, {}, 0.12, 10.12) == {"5", "6"}
    limit.record_scheduled([str(i) for i in range(10)], 1.51)
    assert limit.held(requests, True, policy, {}, 1.52, 11.52) == {"6", "7"}
    requests[8].num_output_tokens += 32
    assert limit.held(requests, True, policy, {}, 1.53, 11.53) == {"7", "8"}
    limit.held([], True, policy, {}, 2, 12)
    assert not limit.progress
