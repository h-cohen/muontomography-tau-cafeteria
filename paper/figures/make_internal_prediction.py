"""Detector-bin held-out counts, raw residuals and conditional uncertainty."""

import numpy as np

from internal_figure_utils import Archive, begin_output, cli, read_npz, save
from style import SEQUENTIAL, plt


def generate(root, out):
    archive = Archive(root)
    archive.result("prediction")
    records = []
    for fold in archive.folds:
        path, record, _ = archive.fold(fold["name"])
        edges = read_npz(path / "test-counts.npz", ["edges"])["edges"]
        for pid in record["positions"]:
            arrays = read_npz(
                path / f"{pid}-prediction.npz", ["observed", "predicted", "residual", "coverage"]
            )
            shape = (len(edges) - 1,) * 2
            if any(v.shape != shape for v in arrays.values()):
                raise ValueError("detector-bin array shape mismatch")
            mask = arrays["coverage"]
            if mask.dtype != np.bool_:
                raise ValueError("prediction coverage must be boolean")
            if not np.allclose(
                arrays["residual"][mask], (arrays["observed"] - arrays["predicted"])[mask]
            ):
                raise ValueError("residual differs from archived count difference")
            uncertainty = None
            if "uncertainty_replicas" in record:
                uncertainty = read_npz(
                    path / f"{pid}-conditional-uncertainty.npz", ["std", "n_supported"]
                )
            records.append((record, pid, arrays, edges, uncertainty))
    out = begin_output(
        archive,
        out,
        dict(
            residual_normalization="observed - predicted counts; no independent pull",
            uncertainty="first fold per protocol; shared reference variation included",
        ),
    )
    for record, pid, data, edges, uncertainty in records:
        fields = [
            ("observed", "Held-out observed counts"),
            ("predicted", "Predicted counts"),
            ("residual", "Held-out residual (counts)"),
        ]
        if uncertainty is not None:
            fields.append(("std", "Conditional prediction SD\n(counts; shared roof variation)"))
        fig, axes = plt.subplots(
            1, len(fields), figsize=(3 * len(fields), 3.1), layout="constrained"
        )
        count_max = max(
            np.max(data["observed"][data["coverage"]], initial=0),
            np.max(data["predicted"][data["coverage"]], initial=0),
            1,
        )
        for ax, (key, title) in zip(axes, fields, strict=True):
            mask = (
                data["coverage"]
                if key != "std"
                else data["coverage"] & (uncertainty["n_supported"] > 1)
            )
            values = data[key] if key != "std" else uncertainty[key]
            residual_max = max(np.max(np.abs(values[mask]), initial=0), 1)
            image = ax.imshow(
                np.where(mask, values, np.nan).T,
                origin="lower",
                extent=[edges[0], edges[-1]] * 2,
                interpolation="nearest",
                cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#efeeea")
                if key == "residual"
                else SEQUENTIAL,
                vmin=-residual_max if key == "residual" else 0,
                vmax=residual_max if key in {"residual", "std"} else count_max,
            )
            ax.set(xlabel="detector tan θx", ylabel="detector tan θy", title=title)
            plt.colorbar(image, ax=ax, shrink=0.7)
        fig.suptitle(
            f"{record['name']} / {pid}: {record['point_score_reference_scope']}\n"
            "Grey: masked or missing prediction coverage; retrospective check",
            fontsize=8,
        )
        save(fig, out, f"{record['name']}-{pid}")


if __name__ == "__main__":
    cli(generate)
