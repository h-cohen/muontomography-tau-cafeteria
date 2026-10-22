"""Campaign configuration.

Everything site-specific -- measured detector geometry, poses, data files,
binning, volume, solver and analysis settings -- lives in configs/*.yaml,
never in code. Lengths are metres, except the detector block (centimetres, as
measured). A pose fitted by `cafetomo selfcal` overrides the config's prior
through `pose_file`, so the fitted pose is data, not an edit to the config.
"""

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import yaml

from cafetomo.detector import Detector
from cafetomo.muonphysics import MODELS


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    az_deg: float = 0.0

    def rotation(self) -> np.ndarray:
        a = np.radians(self.az_deg)
        return np.array(
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
        )


@dataclass(frozen=True)
class Exposure:
    id: str
    root_file: str
    pose: Pose
    root_hist: str = "txty"


@dataclass(frozen=True)
class SkyReference:
    id: str
    root_file: str
    root_hist: str = "txty"


@dataclass(frozen=True)
class Binning:
    t_max: float = 1.25
    n_bins: int = 500

    def edges(self) -> np.ndarray:
        return np.linspace(-self.t_max, self.t_max, self.n_bins + 1)


@dataclass(frozen=True)
class OpacitySettings:
    rebin_factor: int = 10
    max_tan: float = 1.25
    min_sky: float = 25.0
    t_floor: float = 0.05
    sky_t_max: float = 2.5
    sky_n_bins: int = 100
    n_sigma_replicas: int = 12


@dataclass(frozen=True)
class Volume:
    z_min_m: float = 1.0
    z_max_m: float = 9.0
    spacing_m: float = 0.2
    xy_m: tuple | None = None
    viewer_crop_xy_m: tuple | None = None
    # Display-only initial viewer clip box, ((x0, x1), (y0, y1), (z0, z1)) in
    # world metres. Unlike viewer_crop_xy_m, no analysis reads it.
    viewer_clip_m: tuple | None = None
    n_aperture_sub: int = 4

    def __post_init__(self):
        box = self.viewer_clip_m
        if box is not None and (len(box) != 3 or any(len(a) != 2 or not a[0] < a[1] for a in box)):
            raise ValueError("volume.viewer_clip_m needs three increasing (lo, hi) pairs")


@dataclass(frozen=True)
class Reconstruction:
    algorithm: str = "tv"
    n_iter: int = 150
    nonneg: bool = True
    chi2_target: float = 1.0
    tv_alpha: float = 0.03
    tv_z_weight: float = 0.0
    coverage_damping: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class SelfcalSettings:
    free_pose: str = "pos1"
    baseline_m: float | None = None
    bounds_m: float = 1.0
    bounds_deg: float = 10.0
    spacing_m: float = 0.4
    n_iter: int = 60
    n_bootstrap: int = 8
    scan_deg: float = 5.0


@dataclass(frozen=True)
class AutofocusSettings:
    z_min_m: float = 6.0
    z_max_m: float = 8.0
    coarse_m: float = 0.2
    fine_m: float = 0.1
    layer_thickness_m: float = 0.3
    spacing_m: float = 0.1
    trim_pct: float = 90.0


@dataclass(frozen=True)
class BeamSettings:
    band_x: float = 0.32
    band_y: float = 0.45
    match_tol_m: float = 0.6
    prominence_sigmas: float = 0.5
    gate_max_offset_m: float = 0.15
    z_scan_m: tuple = (4.0, 9.0)
    n_z: int = 81
    min_y_corr: float = 0.5


@dataclass(frozen=True)
class BeamDepthSettings:
    model: str = "geometry"
    y_extent_m: tuple = (-5.0, 5.0)
    band_sy: float = 0.32
    w_init_m: float = 0.3
    h_init_m: float = 0.6
    h_starts_m: tuple = (0.6, 1.2, 1.8)
    h_max_m: float = 3.0
    bg_degree: int = 2
    n_sub: int = 4
    y_band_m: tuple = (-2.0, 2.0)

    def __post_init__(self):
        if self.model not in ("geometry", "concrete"):
            raise ValueError(f"beamdepth: unknown model {self.model!r}")


@dataclass(frozen=True)
class UncertaintySettings:
    n_replicas: int = 50
    seed: int = 0
    flux_scale_frac: float = 0.03
    mcs_p_min_gev: float = 0.5


@dataclass(frozen=True)
class ValidationSettings:
    focus_heights_m: tuple = (6.6, 7.0, 7.4)
    depth_h_true_m: tuple = (0.6, 1.25, 2.0)
    n_realizations: int = 8


@dataclass(frozen=True)
class PhysicsSettings:
    """Muon physics behind the pinned beam opacity (see muonphysics)."""

    concrete_density_gcm3: float = 2.4
    concrete_density_sigma: float = 0.1
    detector_threshold_gev: float = 0.03
    flux_model: str = "guan"
    flux_model_alt: str = "shukla"

    def __post_init__(self):
        for m in (self.flux_model, self.flux_model_alt):
            if m not in MODELS:
                raise ValueError(f"physics: unknown muon flux model {m!r}; have {MODELS}")
        if self.concrete_density_gcm3 <= 0 or self.detector_threshold_gev <= 0:
            raise ValueError("physics: density and detector threshold must be positive")


@dataclass(frozen=True)
class RunSubset:
    """Consecutive runs of one raw exposure taken as a source of their own. A run is the
    span between two resets of the run clock; runs are numbered from the exposure's
    first run."""

    exposure: str
    runs: tuple  # (first, last) run number, both included
    turn_deg: (
        float  # detector turn about the vertical from the exposure's first run, ccw from above
    )


@dataclass(frozen=True)
class RawSettings:
    """The DAQ's event-level cluster dumps (`.data`), an alternative source of one
    exposure's angular counts. Lengths are centimetres, as measured."""

    exposures: dict  # exposure id -> directory of its .data files
    channel_bars: tuple  # per ASIC: the DAQ channel read by bar 0, 1, ...
    tx_asics: tuple  # (lower, upper) ASIC pair giving tan(theta_x) = (upper - lower) / gap
    ty_asics: tuple  # (lower, upper) ASIC pair giving tan(theta_y)
    gap_cm: tuple  # layer gap of the tx pair and of the ty pair
    bar_pitch_cm: float  # bar centre b sits at pitch * (b + 1)
    pedestal_hg: tuple  # per ASIC: high-gain charge of a channel with no signal
    clock_divisor: float  # run-clock "seconds" per second of data
    first_run: dict = field(default_factory=dict)  # exposure id -> run number of its first run
    subsets: dict = field(default_factory=dict)  # subset id -> RunSubset of one exposure

    def __post_init__(self):
        for sid, s in self.subsets.items():
            if sid in self.exposures:
                raise ValueError(f"raw: subset {sid!r} names an exposure")
            if s.exposure not in self.exposures:
                raise ValueError(f"raw: subset {sid!r} of unknown exposure {s.exposure!r}")
            if s.exposure not in self.first_run:
                raise ValueError(f"raw: subset {sid!r} needs the first run of {s.exposure!r}")
            first, last = s.runs
            if not self.first_run[s.exposure] <= first <= last:
                raise ValueError(f"raw: subset {sid!r} runs {s.runs} out of order or too early")
        if len(self.channel_bars) != 4 or any(len(c) != len(set(c)) for c in self.channel_bars):
            raise ValueError("raw: channel_bars needs four lists of distinct channels")
        if len({len(c) for c in self.channel_bars}) != 1:
            raise ValueError("raw: every ASIC must read the same number of bars")
        if sorted((*self.tx_asics, *self.ty_asics)) != [0, 1, 2, 3]:
            raise ValueError("raw: tx_asics and ty_asics must name the four ASICs once each")
        if len(self.pedestal_hg) != 4 or len(self.gap_cm) != 2 or min(self.gap_cm) <= 0:
            raise ValueError("raw: need four pedestals and two positive layer gaps")
        if not self.bar_pitch_cm > 0 or not self.clock_divisor > 0:
            raise ValueError("raw: pitch and clock divisor must be positive")


@dataclass(frozen=True)
class CalibrationSettings:
    """Per-channel charge calibration (`cafetomo channelcal`): a Landau convolved with a
    Gaussian fitted to each channel's single-hit spectrum, net of its pedestal."""

    fit_min_frac: float = 0.5  # fit window starts at this fraction of the spectrum peak
    fit_max_adc: float = 9000.0  # window ends here: above it the high-gain range saturates
    min_solo_counts: int = 2000  # single-hit counts below which a channel is not fitted
    outlier_frac: float = 0.15  # |MPV / ASIC median - 1| above which a channel is flagged
    peak_tol_frac: float = 0.10  # |fitted MPV / empirical peak - 1| above which the fit is poor
    mpv_axis_adc: tuple = (3500.0, 8500.0)  # display only: one MPV axis for every source


@dataclass(frozen=True)
class ViewSettings:
    """The 2D angular beam view (`cafetomo beams2d`). Presentation only: nothing here
    feeds an inference, and the count floors only decide which bins are drawn.

    The floors are deliberately strict: the acceptance is a product of two falling spans, so
    the corners of the square fall below them and the drawn region is rounded. The noisy
    corners (sigma >~ 0.1 per bin) are left out on purpose."""

    rebin_factor: int = 5  # 0.025 tan bins from the 0.005 ingest bins
    min_counts: float = 150.0  # position counts below which a bin is not drawn
    min_sky_counts: float = 100.0
    smooth_bins: float = 1.0  # Gaussian sigma of the smoothed panels, in view bins
    background_bins: float = 12.0  # sigma of the broad background removed in the contrast panel
    t_display: float = 0.9  # drawn out to |tan| = t_display
    lam_range: tuple = (0.03, 0.30)
    contrast_range: float = 0.03


@dataclass(frozen=True)
class Config:
    site: str
    data_dir: Path
    detector: Detector
    sky_reference: SkyReference
    exposures: tuple[Exposure, ...]
    binning: Binning = field(default_factory=Binning)
    opacity: OpacitySettings = field(default_factory=OpacitySettings)
    volume: Volume = field(default_factory=Volume)
    reconstruction: Reconstruction = field(default_factory=Reconstruction)
    selfcal: SelfcalSettings = field(default_factory=SelfcalSettings)
    autofocus: AutofocusSettings = field(default_factory=AutofocusSettings)
    beams: BeamSettings = field(default_factory=BeamSettings)
    beamdepth: BeamDepthSettings = field(default_factory=BeamDepthSettings)
    uncertainty: UncertaintySettings = field(default_factory=UncertaintySettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    physics: PhysicsSettings = field(default_factory=PhysicsSettings)
    raw: RawSettings | None = None
    view: ViewSettings = field(default_factory=ViewSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)

    def exposure(self, eid: str) -> Exposure:
        for e in self.exposures:
            if e.id == eid:
                return e
        raise KeyError(f"no exposure {eid!r}; have {list(self.position_ids)}")

    @property
    def position_ids(self) -> tuple[str, ...]:
        """One exposure per position: the position id IS the exposure id."""
        return tuple(e.id for e in self.exposures)

    def origins(self) -> dict[str, tuple[float, float, float]]:
        return {e.id: (float(e.pose.x), float(e.pose.y), float(e.pose.z)) for e in self.exposures}

    def with_pose(self, eid: str, pose: Pose) -> "Config":
        self.exposure(eid)
        exps = tuple(replace(e, pose=pose) if e.id == eid else e for e in self.exposures)
        return replace(self, exposures=exps)

    def fast(self) -> "Config":
        """A cheap variant for CI and smoke runs; never used for paper numbers."""
        return replace(
            self,
            opacity=replace(self.opacity, n_sigma_replicas=3),
            volume=replace(self.volume, spacing_m=0.4),
            reconstruction=replace(self.reconstruction, n_iter=30),
            selfcal=replace(self.selfcal, n_iter=20, n_bootstrap=2),
            autofocus=replace(self.autofocus, coarse_m=0.4, fine_m=0.2),
            uncertainty=replace(self.uncertainty, n_replicas=2),
            validation=ValidationSettings(
                focus_heights_m=(7.0,), depth_h_true_m=(1.25,), n_realizations=2
            ),
        )


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, list) else v


def _build(cls, raw: dict | None, section: str):
    raw = dict(raw or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"config section {section!r}: unknown keys {unknown}")
    return cls(**{k: _tuples(v) for k, v in raw.items()})


def _load_raw(raw: dict | None, base: Path) -> RawSettings | None:
    if raw is None:
        return None
    raw = dict(raw)
    subsets = {
        k: _build(RunSubset, v, f"raw.subsets.{k}")
        for k, v in (raw.pop("subsets", None) or {}).items()
    }
    settings = _build(RawSettings, raw | {"subsets": {}}, "raw")
    settings = replace(settings, subsets=subsets)
    dirs = {k: str((base / v).resolve()) for k, v in settings.exposures.items()}
    return replace(settings, exposures=dirs)


def load_config(path: str | Path, pose_file: str | Path | None = None) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    sections = {
        "binning": Binning,
        "opacity": OpacitySettings,
        "volume": Volume,
        "reconstruction": Reconstruction,
        "selfcal": SelfcalSettings,
        "autofocus": AutofocusSettings,
        "beams": BeamSettings,
        "beamdepth": BeamDepthSettings,
        "uncertainty": UncertaintySettings,
        "validation": ValidationSettings,
        "physics": PhysicsSettings,
        "view": ViewSettings,
        "calibration": CalibrationSettings,
    }
    top = {"site", "data_dir", "detector", "sky_reference", "exposures", "raw", *sections}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"config {path.name}: unknown top-level keys {unknown}")
    exposures = tuple(
        Exposure(
            id=b["id"],
            root_file=b["root_file"],
            pose=_build(Pose, b["pose"], "pose"),
            root_hist=b.get("root_hist", "txty"),
        )
        for b in raw["exposures"]
    )
    cfg = Config(
        site=raw["site"],
        data_dir=(path.parent / raw["data_dir"]).resolve(),
        detector=_build(Detector, raw["detector"], "detector"),
        sky_reference=_build(SkyReference, raw["sky_reference"], "sky_reference"),
        exposures=exposures,
        raw=_load_raw(raw.get("raw"), path.parent),
        **{k: _build(cls, raw.get(k), k) for k, cls in sections.items()},
    )
    if pose_file is not None:
        doc = json.loads(Path(pose_file).read_text())
        cfg = cfg.with_pose(doc["free_pose"], _build(Pose, doc["pose"], "pose"))
    return cfg
