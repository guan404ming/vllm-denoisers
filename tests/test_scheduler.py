import math

import pytest
import torch

from denoisers.policies.hold import Aging, RandomOrder, ShortestFirst, registry
from denoisers.scheduler import probe


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


def test_random_holds_exactly_the_overflow():
    candidates = [(str(i), 0.0) for i in range(6)]
    assert len(RandomOrder().held(candidates, 0.0, 4, {})) == 2


def test_hold_registry_resolves_built_ins_and_rejects_bad_aging():
    assert registry.resolve("sjf") is ShortestFirst
    with pytest.raises(ValueError):
        Aging(0.0)
