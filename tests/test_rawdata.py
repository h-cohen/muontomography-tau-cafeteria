from dataclasses import replace

import numpy as np
import pytest

from cafetomo.config import RawSettings, RunSubset
from cafetomo.rawdata import (
    CHARGE_EDGES,
    cluster_positions,
    count_file,
    ingest_raw,
    read_blocks,
    tracks_from_positions,
)

PED = 2200.0
DZ = 38.9
SETTINGS = RawSettings(
    exposures={"pos0": "."},
    channel_bars=tuple(tuple(range(23)) for _ in range(4)),
    tx_asics=(3, 2),
    ty_asics=(0, 1),
    gap_cm=(DZ, DZ),
    bar_pitch_cm=1.6,
    pedestal_hg=(PED,) * 4,
    clock_divisor=50.0,
)
HEADER = ";".join(["ID_CLUSTER", *(f"c{i}" for i in range(1, 424))])


def _row(cluster_id, t_ns, asic_order, hits):
    """One 424-column cluster row. `hits[asic]` is {channel: HG charge}; slot s reports
    ASIC asic_order[s], so the hit columns follow the slot, not the ASIC number."""
    cells = [cluster_id, t_ns, 0, 4]
    for asic in asic_order:
        flag = np.zeros(32)
        hg = np.full(32, PED)
        for ch, q in hits[asic].items():
            flag[ch] = 1
            hg[ch] = q
        cells += [asic, 0, 0, 0, 0, 0, 0, 0, 0, *flag, *np.zeros(32), *hg]
    assert len(cells) == 424
    return ";".join(f"{c:g}" if isinstance(c, float) else str(int(c)) for c in cells)


def _write(path, rows, extra_header_at=None):
    lines = [HEADER, *rows]
    if extra_header_at is not None:
        lines.insert(extra_header_at, HEADER)
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())


def _single(bar_x_lo, bar_x_hi, bar_y_lo, bar_y_hi):
    q = PED + 4000.0
    return {3: {bar_x_lo: q}, 2: {bar_x_hi: q}, 0: {bar_y_lo: q}, 1: {bar_y_hi: q}}


def test_blocks_skip_repeated_headers_and_follow_the_asic_column(tmp_path):
    rows = [
        _row(1, 1e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),
        _row(2, 2e9, (1, 2, 3, 0), _single(5, 5, 5, 5)),
    ]
    _write(tmp_path / "a.data", rows, extra_header_at=2)
    (blk,) = list(read_blocks(tmp_path / "a.data", rows=10))
    assert blk.timecode_ns.tolist() == [1e9, 2e9]
    # whichever slot carried an ASIC, its hit lands under that ASIC's own index
    assert blk.hit[:, :, 5].all() and blk.hit.sum() == 8
    assert blk.charge[1, 2, 5] == PED + 4000.0


def test_blocks_reject_malformed_rows_by_name(tmp_path):
    (tmp_path / "bad.data").write_bytes((HEADER + "\r\n1;2;3\r\n").encode())
    with pytest.raises(ValueError, match="bad.data"):
        list(read_blocks(tmp_path / "bad.data", rows=10))


def test_cluster_position_is_charge_centroid_of_peak_and_neighbours():
    hit = np.zeros((3, 23), bool)
    q = np.full((3, 23), PED)
    hit[0, 4], q[0, 4] = True, PED + 3000  # one bar: its centre
    hit[1, 4:6], q[1, 4], q[1, 5] = True, PED + 1000, PED + 3000  # shared 1:3
    hit[2, [4, 5, 9]], q[2, [4, 5, 9]] = True, PED + 1000  # distant hit is not the cluster
    pos, ok = cluster_positions(hit, q, SETTINGS.channel_bars[0], PED, 1.6)
    centre = lambda b: 1.6 * (b + 1)  # noqa: E731
    assert ok.all()
    assert pos[0] == pytest.approx(centre(4))
    assert pos[1] == pytest.approx(0.25 * centre(4) + 0.75 * centre(5))
    assert pos[2] == pytest.approx(0.5 * (centre(4) + centre(5)))


def test_hit_without_charge_above_pedestal_is_no_cluster():
    hit = np.zeros((1, 23), bool)
    hit[0, 7] = True
    q = np.full((1, 23), PED - 5)
    _, ok = cluster_positions(hit, q, SETTINGS.channel_bars[0], PED, 1.6)
    assert not ok.any()


def test_channel_map_sends_channels_to_bars():
    channel_bars = tuple(reversed(range(23)))  # bar b reads channel 22 - b
    hit = np.zeros((1, 32), bool)
    q = np.full((1, 32), PED)
    hit[0, 20], q[0, 20] = True, PED + 500  # channel 20 is bar 2
    pos, ok = cluster_positions(hit, q, channel_bars, PED, 1.6)
    assert ok.all() and pos[0] == pytest.approx(1.6 * 3)


def test_tracks_use_upper_minus_lower_over_layer_gap():
    pos = np.zeros((1, 4))
    pos[0, 3], pos[0, 2] = 10.0, 10.0 + 0.5 * DZ  # x: lower ASIC 3, upper ASIC 2
    pos[0, 0], pos[0, 1] = 20.0, 20.0 - 0.25 * DZ  # y: lower ASIC 0, upper ASIC 1
    tx, ty = tracks_from_positions(pos, SETTINGS)
    assert tx[0] == pytest.approx(0.5) and ty[0] == pytest.approx(-0.25)


def test_count_file_histogram_live_time_and_rejects(tmp_path):
    rows = [
        _row(1, 1e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),  # vertical track
        _row(2, 2.0e9, (3, 2, 1, 0), _single(5, 5, 5, 5)),
        _row(3, 51e9, (0, 1, 2, 3), _single(2, 12, 5, 5)),  # tx = 10 bars / DZ
        _row(
            4, 101e9, (0, 1, 2, 3), {3: {}, 2: {5: PED + 900}, 0: {5: PED + 900}, 1: {5: PED + 900}}
        ),
    ]
    _write(tmp_path / "a.data", rows)
    edges = np.linspace(-2, 2, 801)
    c = count_file(tmp_path / "a.data", SETTINGS, edges, block_rows=2)
    assert c.n_rows == 4 and c.n_tracks == 3 and c.n_no_cluster == 1
    i0 = np.searchsorted(edges, 0.0, side="right") - 1
    assert c.values[i0, i0] == 2 and c.values.sum() == 3
    tx = 10 * 1.6 / DZ
    assert c.values[np.searchsorted(edges, tx, side="right") - 1, i0] == 1
    # run clock counts 50x real time: (101 - 1) s of clock is 2 s of data
    assert c.live_time_s == pytest.approx(100.0 / 50.0)
    assert c.n_clock_resets == 0


def test_clock_reset_starts_a_new_run_without_adding_its_jump(tmp_path):
    rows = [
        _row(1, 10e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),
        _row(2, 60e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),
        _row(3, 5e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),
        _row(4, 25e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),
    ]
    _write(tmp_path / "a.data", rows)
    c = count_file(tmp_path / "a.data", SETTINGS, np.linspace(-2, 2, 801), block_rows=3)
    assert c.n_clock_resets == 1
    assert c.live_time_s == pytest.approx((50.0 + 20.0) / 50.0)


def test_ingest_raw_combines_files_and_reuses_the_cache(tmp_path, cfg):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(2):
        rows = [_row(k, (k + 1) * 1e9, (0, 1, 2, 3), _single(5, 5, 5, 5)) for k in range(3)]
        _write(raw / f"run{i}.data", rows)
    settings = replace(SETTINGS, exposures={"pos0": str(raw)})
    cfg = replace(cfg, raw=settings)
    out = ingest_raw(cfg, "pos0", tmp_path / "cache", jobs=1)
    assert out.hist.total == 6 and out.total_in_files == 6
    assert out.live_time_s == pytest.approx(2 * 2.0 / 50.0)
    assert [f.name for f in out.files] == ["run0.data", "run1.data"]
    again = ingest_raw(cfg, "pos0", tmp_path / "cache", jobs=1)
    assert again.hist.values.sum() == 6 and again.cached == 2
    changed = ingest_raw(
        replace(cfg, raw=replace(settings, bar_pitch_cm=1.7)), "pos0", tmp_path / "cache", jobs=1
    )
    assert changed.cached == 0


def test_ingest_raw_without_files_is_an_error(tmp_path, cfg):
    empty = tmp_path / "raw"
    empty.mkdir()
    cfg = replace(cfg, raw=replace(SETTINGS, exposures={"pos0": str(empty)}))
    with pytest.raises(ValueError, match="no .data files"):
        ingest_raw(cfg, "pos0", tmp_path / "cache", jobs=1)


def test_count_file_histograms_solo_hit_charge_and_pedestal_per_channel(tmp_path):
    rows = [
        _row(1, 1e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),  # every ASIC: channel 5 alone
        _row(
            2,
            2e9,
            (0, 1, 2, 3),
            {3: {5: PED + 1010, 6: PED + 500}, 2: {5: 3000}, 0: {5: 3000}, 1: {5: 3000}},
        ),
    ]
    _write(tmp_path / "a.data", rows)
    c = count_file(tmp_path / "a.data", SETTINGS, np.linspace(-2, 2, 801))
    assert c.hit_hist.shape == (4, 32, len(CHARGE_EDGES) - 1)
    # ASIC 3 channel 5 fired alone once (PED + 4000); in row 2 channels 5 and 6 shared
    assert c.hit_hist[3, 5].sum() == 1 and c.hit_hist[3, 6].sum() == 0
    assert c.hit_hist[3, 5][np.searchsorted(CHARGE_EDGES, PED + 4000, side="right") - 1] == 1
    assert c.n_flagged[3, 5] == 2 and c.n_flagged[3, 6] == 1
    # an unflagged channel contributes its charge to the pedestal histogram
    assert c.ped_hist[3, 0].sum() == 2 and c.ped_hist[3, 5].sum() == 0
    assert c.n_valid_rows == 2


def _record(cluster_id, t_ns, asic_order, hits, t_fmt="{:.0f}"):
    """The same cluster in the multi-line record export: a marker line, a cluster line,
    then per slot a bookkeeping line, a hit-flag line and one (high-gain) charge line."""
    lines = ["1234BBBB1234, ", f"{cluster_id},{t_fmt.format(t_ns)},0,4,"]
    for asic in asic_order:
        flag = np.zeros(32, int)
        hg = np.full(32, int(PED))
        for ch, q in hits[asic].items():
            flag[ch] = 1
            hg[ch] = int(q)
        lines += [
            f"{asic},0,0,0,0,0,0,0,0,",
            ",".join(map(str, flag)) + ",",
            ",".join(map(str, hg)),
        ]
    lines[-1] += "\r,"
    return "\n".join(lines) + "\n"


def test_record_export_reads_like_the_table(tmp_path):
    clusters = [
        (1, 1e9, (0, 1, 2, 3), _single(5, 5, 5, 5)),
        (2, 3.3736645326475e15, (1, 2, 3, 0), _single(2, 12, 7, 9)),
    ]
    _write(tmp_path / "a.data", [_row(*c) for c in clusters])
    (tmp_path / "b.data").write_text(
        "".join(_record(*clusters[0]) + _record(*clusters[1], t_fmt="{:.13E}") for _ in [0])
    )
    (a,) = list(read_blocks(tmp_path / "a.data"))
    (b,) = list(read_blocks(tmp_path / "b.data"))
    assert (a.hit == b.hit).all() and (a.charge == b.charge).all()
    assert b.timecode_ns.tolist() == pytest.approx([1e9, 3.3736645326475e15])


def test_record_export_streams_across_chunks_and_blocks(tmp_path):
    text = "".join(_record(k, (k + 1) * 1e9, (0, 1, 2, 3), _single(5, 5, 5, 5)) for k in range(7))
    (tmp_path / "b.data").write_text(text)
    blocks = list(read_blocks(tmp_path / "b.data", rows=3, chunk_bytes=500))
    assert [b.timecode_ns.size for b in blocks] == [3, 3, 1]
    assert np.concatenate([b.timecode_ns for b in blocks]).tolist() == [
        (k + 1) * 1e9 for k in range(7)
    ]


def test_truncated_record_is_an_error_naming_the_file(tmp_path):
    text = _record(1, 1e9, (0, 1, 2, 3), _single(5, 5, 5, 5))
    (tmp_path / "cut.data").write_text(text + text[: len(text) // 2])
    with pytest.raises(ValueError, match="cut.data"):
        list(read_blocks(tmp_path / "cut.data"))


def _runs_file(path):
    """Three runs back to back: the clock restarts at each, with 1, 2 and 3 clusters."""
    rows, k = [], 0
    for n, start in ((1, 100e9), (2, 10e9), (3, 10e9)):
        for j in range(n):
            k += 1
            rows.append(_row(k, start + j * 10e9, (0, 1, 2, 3), _single(5, 5, 5, 5)))
    _write(path, rows)


def test_count_file_keeps_only_the_chosen_runs(tmp_path):
    _runs_file(tmp_path / "a.data")
    edges = np.linspace(-2, 2, 801)
    c = count_file(tmp_path / "a.data", SETTINGS, edges, block_rows=2, segments=(1, 3))
    assert c.n_tracks == 5 and c.values.sum() == 5 and c.n_rows == 5
    assert c.live_time_s == pytest.approx((10.0 + 20.0) / 50.0)
    assert c.n_clock_resets == 2 and c.n_segments == 3
    whole = count_file(tmp_path / "a.data", SETTINGS, edges, block_rows=2)
    assert whole.n_tracks == 6 and whole.n_segments == 3


def test_ingest_raw_reads_a_run_subset_of_its_exposure(tmp_path, cfg):
    raw = tmp_path / "raw"
    raw.mkdir()
    _runs_file(raw / "roof.data")
    settings = replace(
        SETTINGS,
        exposures={"SKY": str(raw)},
        first_run={"SKY": 37},
        subsets={"SKY_B": RunSubset(exposure="SKY", runs=(38, 39), turn_deg=-90.0)},
    )
    out = ingest_raw(replace(cfg, raw=settings), "SKY_B", tmp_path / "cache", jobs=1)
    assert out.hist.total == 5 and out.live_time_s == pytest.approx(30.0 / 50.0)
    late = replace(settings, subsets={"SKY_B": RunSubset("SKY", (38, 40), -90.0)})
    with pytest.raises(ValueError, match="runs 37-39"):
        ingest_raw(replace(cfg, raw=late), "SKY_B", tmp_path / "cache", jobs=1)


def test_a_run_subset_needs_a_single_file(tmp_path, cfg):
    raw = tmp_path / "raw"
    raw.mkdir()
    _runs_file(raw / "a.data")
    _runs_file(raw / "b.data")
    settings = replace(
        SETTINGS,
        exposures={"SKY": str(raw)},
        first_run={"SKY": 37},
        subsets={"SKY_B": RunSubset("SKY", (38, 39), -90.0)},
    )
    with pytest.raises(ValueError, match="one .data file"):
        ingest_raw(replace(cfg, raw=settings), "SKY_B", tmp_path / "cache", jobs=1)


def test_run_subsets_are_validated():
    with pytest.raises(ValueError, match="unknown exposure"):
        replace(SETTINGS, first_run={"pos0": 1}, subsets={"A": RunSubset("SKY", (1, 2), 0.0)})
    with pytest.raises(ValueError, match="first run"):
        replace(SETTINGS, subsets={"A": RunSubset("pos0", (1, 2), 0.0)})
    with pytest.raises(ValueError, match="runs"):
        replace(SETTINGS, first_run={"pos0": 5}, subsets={"A": RunSubset("pos0", (4, 6), 0.0)})
    with pytest.raises(ValueError, match="names an exposure"):
        replace(SETTINGS, first_run={"pos0": 1}, subsets={"pos0": RunSubset("pos0", (1, 2), 0.0)})
