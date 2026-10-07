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
    n_aperture_sub: int = 4


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
    y_extent_m: tuple = (-5.0, 5.0)
    band_sy: float = 0.32
    w_init_m: float = 0.3
    h_init_m: float = 0.6
    h_max_m: float = 3.0
    bg_degree: int = 2
    n_sub: int = 4
    y_band_m: tuple = (-2.0, 2.0)


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
    }
    top = {"site", "data_dir", "detector", "sky_reference", "exposures", *sections}
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
        **{k: _build(cls, raw.get(k), k) for k, cls in sections.items()},
    )
    if pose_file is not None:
        doc = json.loads(Path(pose_file).read_text())
        cfg = cfg.with_pose(doc["free_pose"], _build(Pose, doc["pose"], "pose"))
    return cfg
