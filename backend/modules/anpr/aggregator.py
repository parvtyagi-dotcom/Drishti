"""Per-track OCR observation aggregation.

A single frame's OCR read is never trusted on its own. Each track
accumulates a bounded window of (text, confidence) observations; the
aggregate is the exact normalized-text group with the highest total
confidence weight, distinguishing three states so the pipeline (and caller)
never has to guess:

  - no observations at all                    -> nothing to report
  - a leading candidate exists but hasn't      -> "read" (provisional,
    reached the observation-count/confidence      don't trust it yet)
    bar
  - the leading candidate cleared both bars    -> "confirmed"

Observations are grouped by exact normalized text, then near-identical
groups are merged. Exact grouping alone was the original design, on the
assumption that a real plate produces one consistent majority reading and
only garbage frames differ. Real footage does not behave that way: a plate
photographed at 100px across a moving vehicle reads as "20OX944", "200X944"
and "200X994" on consecutive frames - the same plate every time, never the
same string. Measured on road footage, one vehicle produced nine readings
and no two matched, so a three-observation bar could never be met and
nothing was ever confirmed.

Merging is safe here in a way it would not be globally, because
aggregation is per track: every observation in one group is a reading of one
vehicle's one plate by construction, so tolerating character-level OCR noise
within a track cannot conflate two different vehicles. The reported text is
still a real reading - the highest-confidence variant in the group - never a
synthesised blend of several.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from backend.modules.anpr.ocr import plates_match


@dataclass
class PlateObservation:
    text: str
    confidence: float
    frame_index: int
    timestamp_sec: float


@dataclass
class AggregatedPlate:
    text: str
    mean_confidence: float
    observation_count: int
    total_observation_count: int  # across all text variants seen for this track
    confirmed: bool


class PlateAggregator:
    def __init__(self, max_observations_per_track: int, vote_min_observations: int,
                 confirm_min_mean_confidence: float, agreement_threshold: float = 0.8,
                 agreement_max_len_diff: int = 1):
        self.max_observations_per_track = max_observations_per_track
        self.vote_min_observations = vote_min_observations
        self.confirm_min_mean_confidence = confirm_min_mean_confidence
        self.agreement_threshold = agreement_threshold
        self.agreement_max_len_diff = agreement_max_len_diff
        self._observations: dict[int, list[PlateObservation]] = defaultdict(list)

    def add_observation(self, track_id: int, text: str, confidence: float, frame_index: int, timestamp_sec: float) -> None:
        if not text:
            return
        obs_list = self._observations[track_id]
        obs_list.append(PlateObservation(text, confidence, frame_index, timestamp_sec))
        if len(obs_list) > self.max_observations_per_track:
            del obs_list[0]

    def observation_count(self, track_id: int) -> int:
        return len(self._observations.get(track_id, []))

    def aggregate(self, track_id: int) -> AggregatedPlate | None:
        obs_list = self._observations.get(track_id)
        if not obs_list:
            return None

        groups: dict[str, list[float]] = defaultdict(list)
        for obs in obs_list:
            groups[obs.text].append(obs.confidence)

        cluster = self._best_cluster(groups)
        confidences = [c for text in cluster for c in groups[text]]
        # The reported plate is the single best-supported spelling in the
        # cluster, not a merge of them: an operator has to be able to compare
        # what the system says against what is on the vehicle.
        best_text = max(cluster, key=lambda t: (sum(groups[t]), len(groups[t]), t))
        mean_conf = sum(confidences) / len(confidences)
        confirmed = (
            len(confidences) >= self.vote_min_observations
            and mean_conf >= self.confirm_min_mean_confidence
        )

        return AggregatedPlate(
            text=best_text,
            mean_confidence=round(mean_conf, 1),
            observation_count=len(confidences),
            total_observation_count=len(obs_list),
            confirmed=confirmed,
        )

    def _best_cluster(self, groups: dict[str, list[float]]) -> list[str]:
        """The set of spellings carrying the most total confidence.

        Each distinct spelling seeds a cluster of the spellings close enough
        to be OCR noise on the same plate; the heaviest cluster wins. Seeding
        from every spelling rather than growing one greedily keeps the result
        independent of the order observations happened to arrive in.
        """
        best: list[str] = []
        best_weight = -1.0
        for seed in sorted(groups, key=lambda t: (-sum(groups[t]), t)):
            cluster = [
                text for text in groups
                if text == seed or plates_match(
                    text, seed, self.agreement_threshold, self.agreement_max_len_diff)[0]
            ]
            weight = sum(c for text in cluster for c in groups[text])
            if weight > best_weight:
                best, best_weight = cluster, weight
        return best

    def all_track_ids(self) -> list[int]:
        return list(self._observations.keys())
