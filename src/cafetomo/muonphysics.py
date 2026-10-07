"""Muon transmission through concrete: what an opacity lambda means in grammage.

lambda = -ln T is measured against the open sky, so a ray that crossed slant
grammage X (g/cm^2) keeps the muons energetic enough to cross X and still
fire the tracker:

    T(X, cos) = I(>e_min(X), cos) / I(>e_min(0), cos)

with I(>E) the sea-level integral intensity and e_min(X) the kinetic energy
whose CSDA range is X plus the range of the detector threshold energy. All
straggling, scattering out of the ray and geomagnetic effects are neglected:
this is the CSDA "range-out" picture used for muon radiography.

Sea-level spectra (energies are total muon energy E in GeV; intensity
per cm^2 s sr GeV, converted to per m^2 here):

- ``gaisser`` -- PDG Review of Particle Physics, "Cosmic Rays" (2022 edition),
  eq. (30.4): 0.14 E^-2.7 [1/(1 + 1.1 E cos/115) + 0.054/(1 + 1.1 E cos/850)].
  https://pdg.lbl.gov/2022/reviews/rpp2022-rev-cosmic-rays.pdf
  The review states it is valid only where muon decay is negligible,
  E > 100/cos GeV, and theta < 70 deg; at the sub-GeV energies that decide a
  building's transmission it overshoots the measured flux by an order of
  magnitude (vertical I(>1 GeV) ~ 820 vs ~70 per m^2 s sr), so it is kept
  only to show that the low-energy models reduce to it at high energy.
- ``guan`` -- Guan, Chu, Cao, Luk, Yang, "A parametrization of the cosmic-ray
  muon flux at sea-level", arXiv:1509.06176, eq. (3): Gaisser's form with
  E^-2.7 -> [E (1 + 3.64/(E cos*^1.29))]^-2.7 and cos -> cos*, with cos* the
  Earth-curvature-corrected zenith cosine of their eq. (2) / Table 1 (taken
  from Chirkin / Volkova). Fitted to world sea-level data down to low energy.
- ``shukla`` -- Shukla & Sankrith, "Energy and angular distributions of
  atmospheric muons at the Earth", Int. J. Mod. Phys. A 33 (2018) 1850175,
  arXiv:1606.06907, eq. (10): I0 N (E0 + E)^-n (1 + E/eps)^-1 D(theta)^-(n-1),
  N = (n - 1)(E0 + Ec)^(n-1), with D(theta) the curved-atmosphere path ratio
  of eq. (7). Parameters: Table 1, vertical sea-level muons (Tsukuba,
  E > 0.5 GeV): I0 = 70.7 m^-2 s^-1 sr^-1, n = 3.01, E0 = 4.29 GeV,
  eps = 854 GeV, Ec = 0.5 GeV; Table 2: R/d = 174. An independent
  low-energy parametrisation, used for the alternative-spectrum systematic.

Tang et al. 2006 (arXiv:hep-ph/0604078, eqs. 3-10) was also considered: as
printed, its decay factor A = 1.1 (90 sqrt(cos + 0.001)/1030)^(4.5/(E cos*))
gives a vertical I(>1 GeV) of ~22 per m^2 s sr, a third of the PDG value, so
it could not be reproduced and is not used.

Range: CSDA range of muons in "shielding concrete" (PDG Atomic and Nuclear
Properties, index 144, Groom, Mokhov & Striganov tables, file
muE_shielding_concrete.txt, density 2.300 g/cm^3).
https://pdg.lbl.gov/2024/AtomicNuclearProperties/MUE/muE_shielding_concrete.txt
Columns T [MeV] and "CSDA Range [g/cm^2]" are embedded below from 10 MeV (the
table flags lower entries as not dependable) to 100 GeV. The range in
grammage barely depends on density (only through the density-effect
correction), so the beam density enters only via X = 100 * rho * L. Steel
reinforcement changes the composition slightly (higher Z/A-weighted
stopping power for the steel fraction); that is within the density systematic.
"""

import numpy as np

M_MU_GEV = 0.1056584

# PDG muE_shielding_concrete.txt: kinetic energy [MeV] and CSDA range [g/cm^2].
_TABLE_T_MEV = np.array(
    [
        1.0e1, 1.2e1, 1.4e1, 1.7e1, 2.0e1, 2.5e1, 3.0e1, 3.5e1, 4.0e1, 4.5e1,
        5.0e1, 5.5e1, 6.0e1, 7.0e1, 8.0e1, 9.0e1, 1.0e2, 1.2e2, 1.4e2, 1.7e2,
        2.0e2, 2.5e2, 3.0e2, 3.5e2, 4.0e2, 4.5e2, 5.0e2, 5.5e2, 6.0e2, 7.0e2,
        8.0e2, 9.0e2, 1.0e3, 1.2e3, 1.4e3, 1.7e3, 2.0e3, 2.5e3, 3.0e3, 3.5e3,
        4.0e3, 4.5e3, 5.0e3, 5.5e3, 6.0e3, 7.0e3, 8.0e3, 9.0e3, 1.0e4, 1.2e4,
        1.4e4, 1.7e4, 2.0e4, 2.5e4, 3.0e4, 3.5e4, 4.0e4, 4.5e4, 5.0e4, 5.5e4,
        6.0e4, 7.0e4, 8.0e4, 9.0e4, 1.0e5,
    ]
)  # fmt: skip
_TABLE_RANGE_GCM2 = np.array(
    [
        8.343e-1, 1.156e0, 1.520e0, 2.138e0, 2.835e0, 4.155e0, 5.649e0, 7.295e0,
        9.073e0, 1.097e1, 1.296e1, 1.505e1, 1.722e1, 2.177e1, 2.655e1, 3.151e1,
        3.664e1, 4.725e1, 5.822e1, 7.511e1, 9.232e1, 1.214e2, 1.506e2, 1.798e2,
        2.089e2, 2.378e2, 2.666e2, 2.952e2, 3.237e2, 3.801e2, 4.360e2, 4.912e2,
        5.460e2, 6.541e2, 7.607e2, 9.181e2, 1.073e3, 1.327e3, 1.577e3, 1.823e3,
        2.066e3, 2.307e3, 2.546e3, 2.783e3, 3.018e3, 3.484e3, 3.945e3, 4.402e3,
        4.855e3, 5.752e3, 6.637e3, 7.948e3, 9.242e3, 1.137e4, 1.346e4, 1.551e4,
        1.755e4, 1.956e4, 2.154e4, 2.351e4, 2.545e4, 2.929e4, 3.305e4, 3.675e4,
        4.038e4,
    ]
)  # fmt: skip
_LOG_T = np.log(_TABLE_T_MEV / 1000.0)
_LOG_R = np.log(_TABLE_RANGE_GCM2)

# Kinetic-energy grid of the integral intensity: from below the table's first
# entry to 1e6 GeV, where E^-3.7 leaves a negligible tail.
_GRID_T = np.geomspace(5e-3, 1e6, 1600)
_LOG_GRID_T = np.log(_GRID_T)

# Guan et al. 2015, Table 1 (Earth-curvature zenith correction, eq. 2).
_P = (0.102573, -0.068287, 0.958633, 0.0407253, 0.817285)

MODELS = ("gaisser", "guan", "shukla")

# Step of the numerical derivative in kappa_concrete, g/cm^2.
_DX = 0.5


def cos_star(c: np.ndarray) -> np.ndarray:
    """Zenith cosine at the muon production height, Guan et al. eq. (2)."""
    p1, p2, p3, p4, p5 = _P
    c = np.asarray(c, dtype=np.float64)
    return np.sqrt((c**2 + p1**2 + p2 * c**p3 + p4 * c**p5) / (1 + p1**2 + p2 + p4))


def _gaisser_bracket(e: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 1.1 * e * c / 115.0) + 0.054 / (1.0 + 1.1 * e * c / 850.0)


# Shukla & Sankrith 2018, Tables 1 and 2 (vertical sea level, Tsukuba).
_SHUKLA = {"i0": 70.7, "n": 3.01, "e0": 4.29, "eps": 854.0, "ec": 0.5, "r_over_d": 174.0}


def _shukla_path_ratio(c: np.ndarray) -> np.ndarray:
    """D(theta), Shukla & Sankrith eq. (7): inclined over vertical path."""
    r = _SHUKLA["r_over_d"]
    return np.sqrt(r**2 * c**2 + 2 * r + 1) - r * c


def _shukla(e: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Shukla & Sankrith eq. (10), already per m^2 s sr GeV."""
    p = _SHUKLA
    norm = (p["n"] - 1) * (p["e0"] + p["ec"]) ** (p["n"] - 1)
    return (
        p["i0"]
        * norm
        * (p["e0"] + e) ** -p["n"]
        / (1 + e / p["eps"])
        * _shukla_path_ratio(c) ** -(p["n"] - 1)
    )


def differential_flux(e_gev, cos_theta, model: str) -> np.ndarray:
    """Sea-level dN/(dE dOmega) in m^-2 s^-1 sr^-1 GeV^-1 at total energy E."""
    e = np.asarray(e_gev, dtype=np.float64)
    c = np.asarray(cos_theta, dtype=np.float64)
    if model == "gaisser":
        f = 0.14 * e**-2.7 * _gaisser_bracket(e, c)
    elif model == "guan":
        cs = cos_star(c)
        f = 0.14 * (e * (1.0 + 3.64 / (e * cs**1.29))) ** -2.7 * _gaisser_bracket(e, cs)
    elif model == "shukla":
        return _shukla(e, c)
    else:
        raise ValueError(f"unknown muon flux model {model!r}; have {MODELS}")
    return 1e4 * f


def _log_integral_table(cos_theta: np.ndarray, model: str) -> np.ndarray:
    """ln I(>T) on _GRID_T for each cosine, [n, n_grid]: trapezoid in ln T,
    accumulated from the top of the grid down (g = T dN/dE is the integrand
    in ln T)."""
    c = np.asarray(cos_theta, dtype=np.float64).reshape(-1, 1)
    if np.any(~np.isfinite(c)) or np.any((c <= 0) | (c > 1)):
        raise ValueError("cos_theta must lie in (0, 1]")
    g = differential_flux(_GRID_T + M_MU_GEV, c, model) * _GRID_T
    seg = 0.5 * (g[:, 1:] + g[:, :-1]) * np.diff(_LOG_GRID_T)
    # Beyond the grid every model falls as E^-3.7, whose tail integral is
    # f(T) T / 2.7: negligible, but it keeps ln I finite at the last node.
    tail = g[:, -1:] / 2.7
    cum = np.concatenate([np.cumsum(seg[:, ::-1], axis=1)[:, ::-1], np.zeros_like(tail)], 1)
    return np.log(cum + tail)


def _interp_rows(table: np.ndarray, log_t: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation of `table` [n, n_grid] at ln T [n]."""
    if np.any(log_t < _LOG_GRID_T[0]) or np.any(log_t > _LOG_GRID_T[-1]):
        raise ValueError("muon energy outside the integration grid")
    i = np.clip(np.searchsorted(_LOG_GRID_T, log_t) - 1, 0, _LOG_GRID_T.size - 2)
    x0, x1 = _LOG_GRID_T[i], _LOG_GRID_T[i + 1]
    rows = np.arange(log_t.size)
    y0, y1 = table[rows, i], table[rows, i + 1]
    return y0 + (y1 - y0) * (log_t - x0) / (x1 - x0)


def integral_flux(e_kin_gev, cos_theta, model: str) -> np.ndarray:
    """I(>T, cos) in m^-2 s^-1 sr^-1: muons of kinetic energy above T."""
    t, c = np.broadcast_arrays(
        np.asarray(e_kin_gev, dtype=np.float64), np.asarray(cos_theta, dtype=np.float64)
    )
    table = _log_integral_table(c.ravel(), model)
    return np.exp(_interp_rows(table, np.log(t.ravel()))).reshape(t.shape)


def csda_range(e_kin_gev) -> np.ndarray:
    """CSDA range (g/cm^2) of a muon of kinetic energy T in concrete,
    log-log interpolated in the PDG table; outside the table it raises."""
    log_t = np.log(np.asarray(e_kin_gev, dtype=np.float64))
    if np.any(log_t < _LOG_T[0]) or np.any(log_t > _LOG_T[-1]):
        raise ValueError(
            f"muon kinetic energy outside the PDG range table "
            f"({_TABLE_T_MEV[0] / 1000} - {_TABLE_T_MEV[-1] / 1000} GeV)"
        )
    return np.exp(np.interp(log_t, _LOG_T, _LOG_R))


def e_min(x_gcm2, threshold_gev: float) -> np.ndarray:
    """Kinetic energy (GeV) a muon needs to cross grammage X and still carry
    the detector threshold: range(e_min) = X + range(threshold)."""
    r = np.asarray(x_gcm2, dtype=np.float64) + csda_range(threshold_gev)
    if np.any(r < _TABLE_RANGE_GCM2[0]) or np.any(r > _TABLE_RANGE_GCM2[-1]):
        raise ValueError("grammage outside the PDG range table")
    return np.exp(np.interp(np.log(r), _LOG_R, _LOG_T))


class Transmission:
    """lambda(X) = -ln T(X) for a fixed set of ray cosines (one per element),
    tabulating each ray's integral intensity once."""

    def __init__(self, cos_theta, *, threshold_gev: float, model: str):
        self.cos = np.atleast_1d(np.asarray(cos_theta, dtype=np.float64))
        self.threshold = float(threshold_gev)
        self._table = _log_integral_table(self.cos, model)
        self._log_i0 = _interp_rows(self._table, np.full(self.cos.size, np.log(self.threshold)))

    def lam(self, x_gcm2) -> np.ndarray:
        """Opacity of slant grammage X, per element."""
        x = np.broadcast_to(np.asarray(x_gcm2, dtype=np.float64), self.cos.shape)
        return self._log_i0 - _interp_rows(self._table, np.log(e_min(x, self.threshold)))

    def overburden(self, lam) -> np.ndarray:
        """Slant grammage X with lambda(X) = lam: the exact inverse of `lam`.
        ln I(>T) falls monotonically with T, so the target ln I is located on
        each element's tabulated curve and mapped back through the range
        table. NaN in -> NaN out; lam <= 0 -> X = 0."""
        lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), self.cos.shape)
        out = np.where(lam <= 0, 0.0, np.nan)
        target = self._log_i0 - lam
        for k in np.flatnonzero(lam > 0):
            # np.interp needs increasing abscissae: reverse the falling curve.
            log_t = np.interp(target[k], self._table[k, ::-1], _LOG_GRID_T[::-1])
            out[k] = max(float(csda_range(np.exp(log_t)) - csda_range(self.threshold)), 0.0)
        return out

    def kappa(self, x_gcm2, rho_gcm3: float) -> np.ndarray:
        """Opacity density (1/m) of extra concrete of density rho behind
        grammage X: 100 rho S(X), S = d lambda/dX by central difference."""
        x = np.broadcast_to(np.asarray(x_gcm2, dtype=np.float64), self.cos.shape)
        s = (self.lam(x + _DX) - self.lam(np.clip(x - _DX, 0.0, None))) / (
            x + _DX - np.clip(x - _DX, 0.0, None)
        )
        return 100.0 * rho_gcm3 * s


def transmission(x_gcm2, cos_theta, *, threshold_gev: float, model: str) -> np.ndarray:
    """T(X, cos) = I(>e_min(X)) / I(>e_min(0))."""
    x, c = np.broadcast_arrays(
        np.asarray(x_gcm2, dtype=np.float64), np.asarray(cos_theta, dtype=np.float64)
    )
    tr = Transmission(c.ravel(), threshold_gev=threshold_gev, model=model)
    return np.exp(-tr.lam(x.ravel())).reshape(x.shape)


def overburden_from_lambda(lam, cos_theta, *, threshold_gev: float, model: str) -> np.ndarray:
    """Invert lambda = -ln T(X) for the slant grammage X (g/cm^2)."""
    lam_a, c = np.broadcast_arrays(
        np.asarray(lam, dtype=np.float64), np.asarray(cos_theta, dtype=np.float64)
    )
    tr = Transmission(c.ravel(), threshold_gev=threshold_gev, model=model)
    return tr.overburden(lam_a.ravel()).reshape(lam_a.shape)


def kappa_concrete(
    x_gcm2, cos_theta, rho_gcm3: float, *, threshold_gev: float, model: str
) -> np.ndarray:
    """Opacity density (1/m) of an extra metre of concrete of density rho for
    a muon that already crossed slant grammage X: 100 rho S(X), with
    S(X) = -d ln I(>e_min(X))/dX."""
    x, c = np.broadcast_arrays(
        np.asarray(x_gcm2, dtype=np.float64), np.asarray(cos_theta, dtype=np.float64)
    )
    tr = Transmission(c.ravel(), threshold_gev=threshold_gev, model=model)
    return tr.kappa(x.ravel(), rho_gcm3).reshape(x.shape)
