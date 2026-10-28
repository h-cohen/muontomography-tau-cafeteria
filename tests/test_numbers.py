import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from cafetomo.numbers import (
    archive_macros,
    config_macros,
    flatten,
    format_value,
    macro_name,
    paper_constants,
    write_config_numbers,
    write_numbers,
)


def test_macro_name_spells_digits():
    assert macro_name("data", "pos0_tracks") == "DataPosZeroTracks"
    assert macro_name("beamdepth", "h") == "BeamdepthH"


def test_value_rounded_to_sigma():
    assert format_value(1.23456, 0.0871) == "1.235"
    assert format_value(0.0871) == "0.0871"
    assert format_value(6.95512) == "6.96"
    assert format_value(28500000) == "28500000"


def test_nan_renders_as_dash():
    assert format_value(float("nan")) == "--"
    assert format_value(float("nan"), 0.1) == "--"
    assert format_value(1.5, float("nan")) == "1.5"


def test_large_values_stay_fixed_notation():
    assert format_value(1134.7) == "1135"
    assert format_value(12345.6) == "12346"


def test_tiny_values_use_num():
    assert format_value(0.000888) == "\\num{0.000888}"
    assert format_value(8.88e-16) == "\\num{8.88e-16}"


def test_midrange_and_negative():
    assert format_value(7.04) == "7.04"
    assert format_value(-0.07) == "\\ensuremath{-0.07}"
    assert format_value(-0.0376, 0.01) == "\\ensuremath{-0.038}"


def test_infinity_raises_with_origin(tmp_path):
    (tmp_path / "z.json").write_text('{"k": Infinity}')
    with pytest.raises(ValueError, match=r"z\.json:k"):
        flatten(tmp_path)


def test_flatten_pairs_sigma_and_skips_lists(tmp_path):
    (tmp_path / "beamdepth.json").write_text(
        json.dumps({"h": 1.234, "h_sigma": 0.087, "xs": [1, 2], "at_bound": False})
    )
    m = flatten(tmp_path)
    assert m == {"BeamdepthH": "1.234", "BeamdepthHSigma": "0.087"}


def test_flatten_skips_nested_dicts_and_strings(tmp_path):
    (tmp_path / "pose.json").write_text(json.dumps({"x": 1.5, "pose": {"x": 1.5}, "free": "p"}))
    assert flatten(tmp_path) == {"PoseX": "1.5"}


def test_collision_raises(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"b_c": 1.0, "bC": 2.0}))
    with pytest.raises(ValueError, match="b_c.*bC|bC.*b_c"):
        flatten(tmp_path)


def test_write_numbers(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"y": 2.0}))
    out = write_numbers(tmp_path, tmp_path / "n.tex")
    assert "\\newcommand{\\XY}{2}" in out.read_text()


def test_macro_names_are_letters_only(tmp_path):
    (tmp_path / "d2.json").write_text(json.dumps({"pos0_n": 3, "a.b-c": 1.0}))
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in flatten(tmp_path))


def test_config_macros_cover_scalars_and_lists():
    raw = {
        "reconstruction": {"algorithm": "tv", "n_iter": 150, "tv_alpha": 0.03},
        "beams": {"z_scan_m": [4.0, 9.0], "n_z": 81},
        "volume": {"viewer_clip_m": [[-2.29, 4.19], [5.32, 8.2]]},
        "site": "x",
    }
    m = config_macros(raw)
    assert m["ConfigReconstructionNIter"] == "150"
    assert m["ConfigReconstructionTvAlpha"] == "0.03"
    assert m["ConfigBeamsZScanMZero"] == "4" and m["ConfigBeamsZScanMOne"] == "9"
    assert m["ConfigVolumeViewerClipMThree"] == "8.2"
    assert not any("Algorithm" in k or "Site" in k for k in m)


def test_paper_constants_match_module_values():
    from cafetomo import arraydepth, beams, inversion, raycast, selfcal

    c = paper_constants()
    assert c["ConstSamplesPerVoxel"] == str(raycast.SAMPLES_PER_VOXEL)
    assert c["ConstTvScalePercentile"] == str(inversion.TV_SCALE_PERCENTILE)
    assert c["ConstProxInnerIterations"] == str(inversion.PROX_INNER_ITERATIONS)
    assert c["ConstPowellMaxIter"] == str(selfcal.POWELL_MAX_ITER)
    assert c["ConstFftWindowTan"] == "0.6" and c["ConstFftMinBins"] == str(beams.FFT_MIN_BINS)
    assert c["ConstArrayHStartsMZero"] == "0.3" and c["ConstColumnMax"] == "2"
    assert c["ConstContinuedPerEnd"] == str(arraydepth.CONTINUED_PER_END)


def test_archive_macros_read_the_selected_iterate(tmp_path):
    import json

    import numpy as np

    info = {"best_iter": 56, "n_iter_used": 150, "chi2_history": [42.7, 3.0]}
    np.savez(tmp_path / "v.npz", meta=np.array(json.dumps({"info": info})))
    m = archive_macros(tmp_path / "v.npz")
    assert m == {"ArchiveBestIter": "56", "ArchiveNIterUsed": "150", "ArchiveChisqStart": "42.7"}


def test_config_macros_do_not_collide_with_results(tmp_path):
    out = (
        write_config_numbers(
            Path("configs/cafeteria.yaml"),
            Path("runs/response-aware-2026-10-07/voxels/volume_full.npz"),
            tmp_path / "config-numbers.tex",
        )
        if Path("runs/response-aware-2026-10-07/voxels/volume_full.npz").exists()
        else None
    )
    names = set(config_macros(yaml.safe_load(Path("configs/cafeteria.yaml").read_text())))
    names |= set(paper_constants())
    assert not names & set(flatten(Path("results")))
    if out is not None:
        assert out.read_text().count("\\newcommand") >= len(names)


BASELINE_CONFIG_COMMIT = "c36afe3"
ANALYSIS_SECTIONS = (
    "opacity",
    "reconstruction",
    "selfcal",
    "autofocus",
    "beams",
    "beamdepth",
    "uncertainty",
    "detector",
    "binning",
)


def test_baseline_config_sections_unchanged():
    """The macros are read from today's config; they must equal what the baseline ran with."""
    old = yaml.safe_load(
        subprocess.run(
            ["git", "show", f"{BASELINE_CONFIG_COMMIT}:configs/cafeteria.yaml"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    new = yaml.safe_load(Path("configs/cafeteria.yaml").read_text())
    for s in ANALYSIS_SECTIONS:
        assert old[s] == new[s], s
    keep = ("z_min_m", "z_max_m", "spacing_m", "n_aperture_sub")
    assert {k: old["volume"][k] for k in keep} == {k: new["volume"][k] for k in keep}


def test_raw_macros_cover_rawdata_and_settings(tmp_path, cfg):
    from cafetomo.numbers import raw_macros

    path = tmp_path / "rawdata.json"
    path.write_text(json.dumps({"first_root_chi": 0.3334, "sky_cal_ok": 75, "x": float("nan")}))
    m = raw_macros(path, cfg)
    assert m["RawFirstRootChi"] == "0.333"
    assert m["RawSkyCalOk"] == "75"
    assert m["RawX"] == "--"
    assert m["RawCalibrationOutlierFrac"] == "0.15"
    assert any(k.startswith("RawView") for k in m)
    assert m["RawNeighbourSigmas"] == "3"  # numeric raw settings: the hit selection
    assert m["RawBarPitchCm"] == "1.6"
    assert not set(m) & set(flatten(Path("results")))
