"""Synthetic truth-associated localization errors, retaining failed and censored outcomes."""

import json

import numpy as np

from internal_figure_utils import Archive, begin_output, cli, save
from style import plt


def generate(root, out):
    archive = Archive(root)
    cases = archive.result("recovery")["cases"]
    if not cases:
        raise ValueError("missing recovery outcomes")
    ids = list(dict.fromkeys(r["case"]["case_id"] for r in cases))
    errors = {key: [] for key in ids}
    statuses = {key: {} for key in ids}
    for record in cases:
        key, status = record["case"]["case_id"], record["status"]
        statuses[key][status] = statuses[key].get(status, 0) + 1
        localization = record.get("voxel_beam_localization")
        if localization is not None:
            for beam in localization["beams"]:
                value = np.asarray([np.nan if v is None else v for v in beam["error_xyz_m"]])
                if value.shape != (3,):
                    raise ValueError("localization error must have xyz axes")
                errors[key].append(value)
    out = begin_output(
        archive,
        out,
        dict(
            recovery=(
                "truth-associated windows; per-y,z median across x removed; not blind detection"
            ),
            intervals="no calibrated intervals; missing coordinates remain unmeasured",
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), layout="constrained")
    for i, key in enumerate(ids):
        values = np.asarray(errors[key]).reshape(-1, 3)
        for axis, color in enumerate(("#1b6ca8", "#d1495b", "#65843c")):
            finite = values[:, axis][np.isfinite(values[:, axis])]
            if len(finite):
                axes[0].scatter(
                    np.full(len(finite), i + (axis - 1) * 0.2),
                    finite,
                    color=color,
                    alpha=0.55,
                    s=12,
                    label="xyz"[axis] if i == 0 else None,
                )
                axes[0].plot(i + (axis - 1) * 0.2, finite.mean(), "_", color=color, ms=12)
    axes[0].axhline(0, color="grey", lw=0.7)
    axes[0].set(
        ylabel="Localization error (m)",
        title="Synthetic beam centroid error: xyz; bar is mean bias",
    )
    axes[0].legend(title="Coordinate")
    bottom = np.zeros(len(ids))
    all_statuses = sorted({r["status"] for r in cases})
    for status in all_statuses:
        counts = np.array([statuses[key].get(status, 0) for key in ids])
        axes[1].bar(np.arange(len(ids)), counts, bottom=bottom, label=status)
        bottom += counts
    axes[1].set(
        ylabel="All outcomes (count)",
        title="Failures and boundary censoring retained; no calibrated intervals",
    )
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(np.arange(len(ids)), ids, rotation=20, ha="right")
    save(fig, out, "recovery-errors-and-outcomes")
    (out / "outcomes.json").write_text(json.dumps(statuses, indent=2) + "\n")


if __name__ == "__main__":
    cli(generate)
