"""The tracker's measured geometry and its exact geometric acceptance.

A track of slope t entering the lower layer at x0 reaches x0 + t*dz in the
upper one; both must lie inside the active width W, so the accepted entry span
is max(W - |t|*dz, 0) per axis. Converting from per-(dtx dty) to per-solid-angle
adds (1 + tx^2 + ty^2)^(-3/2). No free parameters.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detector:
    active_width_cm: float
    layer_dz_cm: float
    source: str = ""

    @property
    def aperture_m(self) -> float:
        return self.active_width_cm / 100.0

    @property
    def max_tan(self) -> float:
        """Acceptance edge: the slope at which the accepted span reaches zero."""
        return self.active_width_cm / self.layer_dz_cm

    def acceptance(self, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
        """Relative acceptance per angular bin; the absolute scale cancels in
        every ratio this analysis forms."""
        tx = np.asarray(tx, dtype=np.float64)
        ty = np.asarray(ty, dtype=np.float64)
        w, dz = self.active_width_cm, self.layer_dz_cm
        span_x = np.clip(w - np.abs(tx) * dz, 0.0, None)
        span_y = np.clip(w - np.abs(ty) * dz, 0.0, None)
        return span_x * span_y * (1.0 + tx**2 + ty**2) ** -1.5
