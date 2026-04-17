import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stageii_benchmark import (
    normalize_stageii_sample,
    run_public_stageii_benchmark,
    write_benchmark_report,
)


def test_normalize_stageii_sample_reads_legacy_support_sample():
    result = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")

    assert result.sample_format == "legacy_stageii_pkl"
    assert result.trans.shape[0] == result.poses.shape[0] == result.markers_obs.shape[0]
    assert result.poses.shape[1] > 100
    assert result.markers_obs.shape[2] == 3
    assert len(result.latent_labels) == result.markers_latent.shape[0]
    assert len(result.marker_labels) == result.markers_obs.shape[1]
    assert result.mocap_frame_rate > 0


def test_run_public_stageii_benchmark_reports_repeatable_summary(tmp_path):
    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=1,
        measured_runs=2,
    )

    assert report["sample"]["format"] == "legacy_stageii_pkl"
    assert report["workload"]["frames"] > 0
    assert report["speed"]["latency_ms"]["count"] == 2
    assert report["speed"]["throughput_ops_s"] > 0
    assert report["error"]["repeatability"]["max_abs_diff"] == 0.0
    assert report["error"]["all_finite"] is True

    output_path = tmp_path / "report.json"
    write_benchmark_report(report, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded["sample"]["sha256"] == report["sample"]["sha256"]
    assert loaded["artifact"]["report_path"].endswith("report.json")
