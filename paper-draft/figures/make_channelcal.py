"""Per-channel Landau x Gauss MPV of the three raw sources, by ASIC."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

import style

SOURCES = {"pos0": "pos0", "pos1": "pos1", "SKY": "open sky"}
COLORS = {**style.POSITION_COLORS, "SKY": style.MUTED}


def channel_points(rows: list[dict]) -> list[dict]:
    """Fitted channels, inactive ones dropped; any status other than ok is flagged."""
    return [
        {
            "asic": r["asic"],
            "channel": r["channel"],
            "mpv": r["mpv_adc"],
            "flagged": r["status"] != "ok",
        }
        for r in rows
        if r["status"] != "inactive" and r["mpv_adc"] is not None
    ]


def main(argv=None) -> None:
    a, _ = style.setup(argv)
    points = {
        sid: channel_points(
            json.loads((Path(a.runs) / "channelcal" / f"{sid}_channelcal.json").read_text())[
                "channels"
            ]
        )
        for sid in SOURCES
    }
    asics = sorted({p["asic"] for pts in points.values() for p in pts})
    fig, axes = plt.subplots(
        1, len(asics), figsize=(style.FULL_IN, 2.4), sharey=True, layout="constrained"
    )
    for k, (sid, pts) in enumerate(points.items()):
        for ax, asic in zip(axes, asics, strict=True):
            for flagged, marker in ((False, "o"), (True, "x")):
                sel = [p for p in pts if p["asic"] == asic and p["flagged"] == flagged]
                ax.plot(
                    [p["channel"] + 0.2 * (k - 1) for p in sel],
                    [p["mpv"] for p in sel],
                    marker,
                    ms=3,
                    color=COLORS[sid],
                    ls="none",
                    label=SOURCES[sid] if (asic == asics[0] and not flagged) else None,
                )
    for ax, asic in zip(axes, asics, strict=True):
        ax.set(title=f"ASIC {asic}", xlabel="channel")
    axes[0].set_ylabel("MPV net of pedestal (ADC)")
    axes[0].legend(loc="lower left")
    style.save(fig, a)


if __name__ == "__main__":
    main()
