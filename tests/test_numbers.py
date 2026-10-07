import json
import re

import pytest

from cafetomo.numbers import flatten, format_value, macro_name, write_numbers


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


def test_exponent_uses_num():
    assert format_value(8.88e-16) == "\\num{8.88e-16}"


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
