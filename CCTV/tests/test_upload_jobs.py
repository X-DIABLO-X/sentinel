import json

import pytest

from netra.jobs import VideoJobManager, safe_video_name


def test_upload_filename_is_flat_and_safe():
    assert safe_video_name(r"..\..\My crash (1).MP4") == "My_crash_1.mp4"


def test_upload_rejects_non_video_extension():
    with pytest.raises(ValueError, match="unsupported video type"):
        safe_video_name("payload.exe")


def test_completed_upload_is_recovered_after_restart(tmp_path):
    job_id = "abc123def456"
    result_dir = tmp_path / "results" / job_id
    result_dir.mkdir(parents=True)
    video = result_dir / "sample_annotated.webm"
    video.write_bytes(b"webm")
    report = {
        "camera_id": "UPLOAD_abc123",
        "events_total": 1,
        "events_by_type": {"collision": 1},
        "events_by_severity": {"medium": 1},
        "events": [{"event_type": "collision"}],
        "stats": {"wall_seconds": 3.5},
    }
    (result_dir / "sample.json").write_text(json.dumps(report), encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / f"{job_id}_sample.mp4").write_bytes(b"video")

    manager = VideoJobManager(tmp_path, {})
    recovered = manager.get(job_id)

    assert recovered is not None
    assert recovered["filename"] == "sample.mp4"
    assert recovered["status"] == "complete"
    assert recovered["result"]["events_total"] == 1
    assert recovered["annotated_video"] == str(video)
