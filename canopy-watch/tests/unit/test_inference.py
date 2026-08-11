"""Tier1 / Tier2 / motion gate / crop / risk scorer / event schema tests."""

from __future__ import annotations

import numpy as np
import pytest

from edge.events import DetectionEvent, SchemaError, event_bytes, validate_event
from edge.inference import (
    BBox,
    Candidate,
    DropNewestQueue,
    MotionGate,
    create_tier1,
    create_tier2,
    crop_for_tier2,
    letterbox,
    tier2_mode,
)
from edge.inference.tier1 import available_tier1_backends, load_tier1_class
from edge.inference.tier2 import available_tier2_backends, load_tier2_class
from edge.scoring import PersistenceWindow, RiskScorer, ScoringContext


def frame_with_blob(w=320, h=240, box=(120, 90, 180, 150), bright=240, base=30):
    img = np.full((h, w, 3), base, dtype=np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = bright
    return img


def candidate(category="human", confidence=0.9, bbox=None):
    return Candidate(category=category, confidence=confidence,
                     bbox=bbox or BBox(0.3, 0.3, 0.5, 0.6), backend="test")


# --------------------------------------------------------------------------- #
# BBox
# --------------------------------------------------------------------------- #

def test_bbox_rejects_degenerate():
    with pytest.raises(ValueError):
        BBox(0.5, 0.5, 0.5, 0.9)


def test_bbox_padding_clamps_at_frame_edge():
    padded = BBox(0.01, 0.01, 0.1, 0.1).padded(0.5)
    assert padded.x1 == 0.0 and padded.y1 == 0.0
    assert padded.x2 > 0.1


def test_bbox_pixel_roundtrip():
    box = BBox(0.25, 0.5, 0.75, 1.0)
    assert box.to_pixels(400, 200) == (100, 100, 300, 200)


def test_bbox_area_is_frame_fraction():
    assert BBox(0.0, 0.0, 0.5, 0.5).area == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Motion gate
# --------------------------------------------------------------------------- #

def test_disabled_gate_passes_everything():
    gate = MotionGate({"enabled": False})
    for _ in range(5):
        assert gate.evaluate(frame_with_blob()).motion is True


def test_static_scene_is_suppressed_after_warmup():
    gate = MotionGate({"enabled": True, "warmup_frames": 3})
    static = np.full((240, 320, 3), 40, dtype=np.uint8)
    for _ in range(6):
        gate.evaluate(static)
    assert gate.evaluate(static).motion is False


def test_moving_object_triggers_motion():
    gate = MotionGate({"enabled": True, "warmup_frames": 3})
    for _ in range(6):
        gate.evaluate(frame_with_blob(box=(10, 10, 60, 60)))
    result = gate.evaluate(frame_with_blob(box=(200, 150, 260, 210)))
    assert result.motion is True
    assert result.bbox is not None


def test_whole_frame_change_is_rejected_as_exposure_step():
    gate = MotionGate({"enabled": True, "warmup_frames": 2, "max_area_fraction": 0.5})
    dark = np.full((240, 320, 3), 20, dtype=np.uint8)
    for _ in range(5):
        gate.evaluate(dark)
    result = gate.evaluate(np.full((240, 320, 3), 220, dtype=np.uint8))
    assert result.motion is False
    assert gate.rejected_too_large == 1


def test_gate_reset_clears_background():
    gate = MotionGate({"enabled": True, "warmup_frames": 1})
    gate.evaluate(frame_with_blob())
    gate.reset()
    assert gate._background is None


# --------------------------------------------------------------------------- #
# Tier1
# --------------------------------------------------------------------------- #

def test_heuristic_tier1_finds_the_blob():
    tier1 = create_tier1(backend="heuristic", params={"score_threshold": 0.0})
    results = tier1.detect(frame_with_blob())
    assert len(results) == 1
    box = results[0].bbox
    assert 0.25 < box.x1 < 0.45
    assert 0.3 < box.y1 < 0.5


def test_heuristic_tier1_tracks_a_moving_blob():
    """mock camera -> heuristic Tier1 must be a real detection, not a stub."""
    tier1 = create_tier1(backend="heuristic", params={"score_threshold": 0.0})
    left = tier1.detect(frame_with_blob(box=(20, 90, 80, 150)))[0]
    right = tier1.detect(frame_with_blob(box=(220, 90, 280, 150)))[0]
    assert right.bbox.x1 > left.bbox.x1


def test_heuristic_tier1_finds_nothing_in_a_flat_frame():
    tier1 = create_tier1(backend="heuristic")
    assert tier1.detect(np.full((240, 320, 3), 80, dtype=np.uint8)) == []


def test_tier1_class_map_projects_coco_onto_four_categories():
    tier1 = create_tier1(backend="heuristic", params={"score_threshold": 0.0})
    assert tier1.map_category("person", None) == "human"
    assert tier1.map_category("truck", None) == "vehicle"
    assert tier1.map_category("elephant", None) == "animal"
    assert tier1.map_category("toaster", None) == "unknown"


def test_tier1_threshold_filters():
    tier1 = create_tier1(backend="heuristic", params={"score_threshold": 0.99})
    assert tier1.detect(frame_with_blob()) == []


def test_tier1_ignore_categories():
    tier1 = create_tier1(backend="heuristic",
                         params={"score_threshold": 0.0,
                                 "assumed_category": "animal",
                                 "ignore_categories": ["animal"]})
    assert tier1.detect(frame_with_blob()) == []


@pytest.mark.parametrize("backend", ["heuristic", "tflite_ssd"])
def test_tier1_backends_import_without_the_ml_stack(backend):
    cls = load_tier1_class(backend)
    assert cls.name == backend


def test_tflite_backend_fails_with_an_actionable_message():
    from edge.inference.tier1 import Tier1Error
    tier1 = create_tier1(backend="tflite_ssd", params={"model_path": "/nope/x.tflite"})
    with pytest.raises(Tier1Error) as err:
        tier1.load()
    assert "fetch_models" in str(err.value)


def test_tier1_registry_contents():
    assert available_tier1_backends() == ("heuristic", "tflite_ssd")


# --------------------------------------------------------------------------- #
# Crop
# --------------------------------------------------------------------------- #

def test_crop_letterboxes_instead_of_squashing():
    frame = frame_with_blob(w=400, h=400, box=(100, 50, 140, 350))  # tall subject
    crop, original = crop_for_tier2(frame, BBox(0.25, 0.125, 0.35, 0.875), target_size=260)
    assert crop.shape == (260, 260, 3)
    assert original[1] > original[0]           # source really was taller than wide


def test_crop_includes_context_padding():
    frame = frame_with_blob(w=400, h=400, box=(150, 150, 250, 250))
    _, tight = crop_for_tier2(frame, BBox(0.375, 0.375, 0.625, 0.625),
                              padding_fraction=0.0)
    _, padded = crop_for_tier2(frame, BBox(0.375, 0.375, 0.625, 0.625),
                               padding_fraction=0.25)
    assert padded[0] > tight[0] and padded[1] > tight[1]


def test_crop_copies_so_readonly_frames_are_safe():
    frame = frame_with_blob()
    frame.flags.writeable = False
    crop, _ = crop_for_tier2(frame, BBox(0.3, 0.3, 0.6, 0.6))
    crop[0, 0] = 1                              # must not raise


def test_letterbox_pads_rather_than_stretches():
    wide = np.full((10, 100, 3), 200, dtype=np.uint8)
    out = letterbox(wide, 64, pad_value=114)
    assert out.shape == (64, 64, 3)
    assert out[0, 0].tolist() == [114, 114, 114]      # padding at the top
    assert out[32, 32].tolist() == [200, 200, 200]    # content in the middle


# --------------------------------------------------------------------------- #
# Tier2
# --------------------------------------------------------------------------- #

def test_heuristic_tier2_is_deterministic():
    tier2 = create_tier2(backend="heuristic")
    crop, _ = crop_for_tier2(frame_with_blob(), BBox(0.3, 0.3, 0.6, 0.7))
    a = tier2.verify(crop, candidate())
    b = tier2.verify(crop, candidate())
    assert a.confidence == b.confidence


def test_heuristic_tier2_passes_category_through():
    tier2 = create_tier2(backend="heuristic")
    crop, _ = crop_for_tier2(frame_with_blob(), BBox(0.3, 0.3, 0.6, 0.7))
    assert tier2.verify(crop, candidate("vehicle")).category == "vehicle"


def test_heuristic_tier2_penalises_a_mostly_padded_crop():
    tier2 = create_tier2(backend="heuristic")
    flat = np.full((260, 260, 3), 114, dtype=np.uint8)
    textured, _ = crop_for_tier2(frame_with_blob(), BBox(0.3, 0.25, 0.65, 0.7))
    assert (tier2.verify(flat, candidate()).confidence
            < tier2.verify(textured, candidate()).confidence)


def test_tier2_mode_from_config():
    assert tier2_mode({"ai": {"tier2": {"mode": "defer_to_cloud"}}}) == "defer_to_cloud"
    assert tier2_mode({"ai": {"tier2": {}}}) == "local"


def test_invalid_tier2_mode_fails_fast():
    from edge.inference.tier2 import Tier2Error
    with pytest.raises(Tier2Error):
        tier2_mode({"ai": {"tier2": {"mode": "maybe"}}})


@pytest.mark.parametrize("backend", ["heuristic", "onnx"])
def test_tier2_backends_import_without_onnxruntime(backend):
    assert load_tier2_class(backend).name == backend


def test_tier2_registry_contents():
    assert available_tier2_backends() == ("heuristic", "onnx")


# --------------------------------------------------------------------------- #
# Tier2 queue — drop-newest
# --------------------------------------------------------------------------- #

def test_tier2_queue_drops_newest_not_oldest():
    """Opposite of the Tier1 queue on purpose: a Tier2 task is work already
    paid for, so under overload finish what you started."""
    q: DropNewestQueue[int] = DropNewestQueue(maxsize=2)
    assert q.put(1) is True
    assert q.put(2) is True
    assert q.put(3) is False
    assert q.get(timeout=0) == 1
    assert q.get(timeout=0) == 2
    assert q.stats()["tasks_dropped_overload"] == 1


def test_tier2_queue_high_water_is_tracked():
    q: DropNewestQueue[int] = DropNewestQueue(maxsize=4)
    for i in range(3):
        q.put(i)
    q.get(timeout=0)
    assert q.stats()["high_water"] == 3


# --------------------------------------------------------------------------- #
# Risk scorer — deterministic
# --------------------------------------------------------------------------- #

NIGHT = 1_700_000_000.0     # 2023-11-14T22:13:20Z
DAY = NIGHT - 12 * 3600


def test_score_is_deterministic():
    scorer = RiskScorer()
    ctx = ScoringContext(zone="core", wall_clock=NIGHT, bbox_area=0.1, recent_similar=1)
    a = scorer.score(candidate(), None, ctx)
    b = scorer.score(candidate(), None, ctx)
    assert a.score == b.score


def test_human_at_night_in_core_outscores_animal_by_day_in_periphery():
    scorer = RiskScorer()
    high = scorer.score(candidate("human", 0.95), None,
                        ScoringContext(zone="core", wall_clock=NIGHT, bbox_area=0.2))
    low = scorer.score(candidate("animal", 0.95), None,
                       ScoringContext(zone="periphery", wall_clock=DAY, bbox_area=0.01))
    assert high.score > low.score
    assert high.band in ("high", "critical")
    assert low.band == "low"


def test_every_factor_is_reported_with_a_reason():
    scorer = RiskScorer()
    result = scorer.score(candidate(), None,
                          ScoringContext(zone="core", wall_clock=NIGHT))
    names = {f.name for f in result.factors}
    assert names == {"class_base", "confidence", "nocturnal",
                     "zone_sensitivity", "persistence", "proximity"}
    assert all(f.reason for f in result.factors)


def test_contributions_sum_to_the_score():
    scorer = RiskScorer()
    result = scorer.score(candidate(), None,
                          ScoringContext(zone="buffer", wall_clock=NIGHT,
                                         bbox_area=0.05, recent_similar=2))
    assert sum(f.contribution for f in result.factors) == pytest.approx(result.score, abs=1e-3)


def test_score_is_bounded_0_to_100():
    scorer = RiskScorer()
    maxed = scorer.score(candidate("human", 1.0), None,
                         ScoringContext(zone="core", wall_clock=NIGHT,
                                        bbox_area=1.0, recent_similar=99))
    assert maxed.score == pytest.approx(100.0, abs=0.01)


def test_weights_are_tunable_from_config():
    quiet = RiskScorer({"weights": {"nocturnal": 0.0}})
    loud = RiskScorer({"weights": {"nocturnal": 5.0}})
    ctx = ScoringContext(zone="core", wall_clock=NIGHT)
    assert loud.score(candidate("animal", 0.3), None, ctx).score > \
           quiet.score(candidate("animal", 0.3), None, ctx).score


def test_tier2_confidence_wins_when_tier2_ran():
    from edge.inference.types import Verification
    scorer = RiskScorer()
    ctx = ScoringContext(zone="core", wall_clock=NIGHT)
    with_t2 = scorer.score(candidate("human", 0.5),
                           Verification(category="human", confidence=0.99), ctx)
    conf = next(f for f in with_t2.factors if f.name == "confidence")
    assert conf.value == pytest.approx(0.99)
    assert "tier2" in conf.reason


def test_missing_timestamp_is_neutral_not_zero():
    scorer = RiskScorer()
    result = scorer.score(candidate(), None, ScoringContext(zone="core"))
    nocturnal = next(f for f in result.factors if f.name == "nocturnal")
    assert nocturnal.value == 0.5


def test_bands():
    scorer = RiskScorer()
    assert scorer.band_for(95) == "critical"
    assert scorer.band_for(65) == "high"
    assert scorer.band_for(40) == "medium"
    assert scorer.band_for(10) == "low"


# --------------------------------------------------------------------------- #
# Persistence window
# --------------------------------------------------------------------------- #

def test_persistence_counts_within_the_window_only():
    window = PersistenceWindow({"window_seconds": 60})
    window.record("human", "core", now=1000)
    window.record("human", "core", now=1030)
    assert window.count("human", "core", now=1040) == 2
    assert window.count("human", "core", now=1200) == 0


def test_persistence_separates_zones_and_categories():
    window = PersistenceWindow({"window_seconds": 600})
    window.record("human", "core", now=1000)
    assert window.count("human", "buffer", now=1000) == 0
    assert window.count("animal", "core", now=1000) == 0


def test_persistence_is_bounded():
    window = PersistenceWindow({"window_seconds": 10 ** 9, "max_entries_per_key": 10})
    for i in range(100):
        window.record("human", "core", now=1000 + i)
    assert window.count("human", "core", now=1100) == 10


# --------------------------------------------------------------------------- #
# DetectionEvent schema
# --------------------------------------------------------------------------- #

def build_event(tier2_status="skipped", verification=None):
    scorer = RiskScorer()
    cand = candidate()
    risk = scorer.score(cand, verification,
                        ScoringContext(zone="core", wall_clock=NIGHT))
    return DetectionEvent.build(
        node_id="rpi-northgate-01", frame_id="rpi-northgate-01-000000042",
        captured_at=NIGHT, candidate=cand, verification=verification,
        tier2_status=tier2_status, risk=risk,
        frame_width=1920, frame_height=1080, zone="core")


def test_event_validates():
    build_event().validate()


def test_event_timestamps_are_iso_utc_not_raw_floats():
    payload = build_event().to_dict()
    assert payload["captured_at"].endswith("+00:00")
    assert "T" in payload["detected_at"]


def test_event_bbox_is_normalized_with_frame_size():
    payload = build_event().to_dict()
    assert all(0.0 <= payload["bbox"][k] <= 1.0 for k in ("x1", "y1", "x2", "y2"))
    assert payload["frame_size"] == {"width": 1920, "height": 1080}


def test_tier2_payload_and_status_must_agree():
    from edge.inference.types import Verification
    with pytest.raises(SchemaError):
        build_event(tier2_status="deferred",
                    verification=Verification(category="human", confidence=0.9)).validate()
    with pytest.raises(SchemaError):
        build_event(tier2_status="local").validate()


def test_deferred_event_is_valid_with_null_tier2():
    payload = build_event(tier2_status="deferred").to_dict()
    validate_event(payload)
    assert payload["tier2"] is None
    assert payload["tier2_status"] == "deferred"


def test_tier2_result_wins_when_it_ran():
    from edge.inference.types import Verification
    event = build_event(tier2_status="local",
                        verification=Verification(category="animal", confidence=0.77,
                                                  label="elephant"))
    payload = event.to_dict()
    assert payload["category"] == "animal"
    assert payload["label"] == "elephant"
    assert payload["confidence"] == pytest.approx(0.77)


def test_event_carries_no_image_bytes_and_stays_small():
    payload = build_event().to_dict()
    assert "image" not in payload and "crop" not in payload
    assert event_bytes(payload) < 2048          # must survive LoRa/4G


def test_risk_factors_must_not_be_empty():
    payload = build_event().to_dict()
    payload["risk"]["factors"] = []
    with pytest.raises(SchemaError):
        validate_event(payload)


def test_schema_version_present():
    assert build_event().to_dict()["schema_version"] == "1.0.0"