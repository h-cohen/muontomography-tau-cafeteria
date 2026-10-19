"""Isolated retrospective campaigns. No independently calibrated physical accuracy."""

import hashlib
import json
import pickle
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from cafetomo.prediction import (
    _digest,
    _training_grid,
    fit_training_fold,
    poisson_deviance,
    predict_raw_counts,
    score_raw_counts,
    select_and_refit,
    summarize_prediction_replicas,
    training_prediction_replicas,
)
from cafetomo.validation_split import angular_masks, split_counts


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, default=plain, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def settings_document(document):
    settings = dict(document)
    mode = settings.get("mode")
    if mode not in {"pilot", "full", "smoke"}:
        raise ValueError("explicit mode pilot/full/smoke required")
    settings.setdefault("root_seed", 314159)
    settings.setdefault(
        "count", {"train_fraction": 0.5, "outer_seeds": 3, "inner_train_fraction": 0.5}
    )
    settings.setdefault("angular", {"folds": 5, "tile": 5, "guard": 1})
    settings.setdefault(
        "selection",
        {
            "tv_alpha": [0.015, 0.03, 0.06],
            "coverage_damping": [0, 0.05, 0.1],
            "vertical_tv_weight": 0,
        },
    )
    settings.setdefault("features", {"signal_to_error": [2, 3, 4], "background_degree": [1, 2]})
    settings.setdefault(
        "recovery", {"noiseless_cases": 2, "noise_replicas_per_case": 5, "truth_subrays": 16}
    )
    settings.setdefault("stability", {"count_replicas": 5, "contrast_threshold": 0.01})
    settings.setdefault(
        "prediction_uncertainty", {"replicas": 2, "scope": "first_outer_fold_per_protocol"}
    )
    for values in (settings["selection"]["tv_alpha"], settings["selection"]["coverage_damping"]):
        if any(not np.isfinite(x) or x < 0 for x in values):
            raise ValueError("finite nonnegative reconstruction candidates required")
    if len(set(settings["selection"]["tv_alpha"])) != len(settings["selection"]["tv_alpha"]):
        raise ValueError("duplicate TV candidates")
    if len(set(settings["selection"]["coverage_damping"])) != len(
        settings["selection"]["coverage_damping"]
    ):
        raise ValueError("duplicate damping candidates")
    for n in (
        settings["recovery"]["noise_replicas_per_case"],
        settings["stability"]["count_replicas"],
        settings["prediction_uncertainty"]["replicas"],
    ):
        if not isinstance(n, int) or n < 1:
            raise ValueError("positive integer replica budgets required")
    if settings["prediction_uncertainty"]["scope"] != "first_outer_fold_per_protocol":
        raise ValueError("only declared representative-fold uncertainty scope supported")
    if any(not np.isfinite(x) or x <= 0 for x in settings["features"]["signal_to_error"]):
        raise ValueError("positive finite feature thresholds required")
    if any(not isinstance(x, int) or x < 0 for x in settings["features"]["background_degree"]):
        raise ValueError("nonnegative integer background degrees required")
    for name in ("train_fraction", "inner_train_fraction"):
        if not 0 < settings["count"][name] < 1:
            raise ValueError("count fractions must lie strictly between zero and one")
    if settings["count"]["outer_seeds"] < 1 or settings["angular"]["folds"] < 2:
        raise ValueError("positive count seeds and at least two angular folds required")
    if settings["selection"].get("vertical_tv_weight", 0) != 0:
        raise ValueError("declared campaign requires zero vertical TV weight")
    if not settings["selection"]["tv_alpha"] or not settings["selection"]["coverage_damping"]:
        raise ValueError("empty candidate grid")
    if mode == "full" and not settings.get("production_budget"):
        raise ValueError("full mode requires predeclared production_budget")
    return settings


def seed(settings, stage, fold=0, replica=0):
    return int(
        np.random.SeedSequence([settings["root_seed"], stage, fold, replica]).generate_state(1)[0]
    )


def partitions(grid, live, cfg, settings):
    smoke = settings["mode"] == "smoke"
    for i in range(1 if smoke else settings["count"]["outer_seeds"]):
        stream = seed(settings, 1, i)
        part = split_counts(grid, live, fraction=settings["count"]["train_fraction"], seed=stream)
        yield dict(
            name=f"count-{i}",
            protocol="count",
            seed=stream,
            train=part.train,
            test=part.test,
            train_live=part.train_live,
            test_live=part.test_live,
            train_masks=None,
            test_masks=None,
            roof_scope="training_split",
        )
    angular = settings["angular"]
    for i in range(1 if smoke else angular["folds"]):
        train, test = angular_masks(
            (grid.n_bins, grid.n_bins),
            fold=i,
            folds=angular["folds"],
            tile=angular["tile"],
            guard=angular["guard"],
        )
        yield dict(
            name=f"angular-{i}",
            protocol="angular",
            seed=seed(settings, 2, i),
            train=grid,
            test=grid,
            train_live=dict(live),
            test_live=dict(live),
            train_masks={p: train for p in cfg.position_ids},
            test_masks={p: test for p in cfg.position_ids},
            roof_scope="full_reference_conditional",
        )


def source_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def cached_fitter(source):
    def fit(
        grid, live, cfg, *, observation_mask, cache_dir, frozen_nuisance=None, nuisance_cfg=None
    ):
        clean, masks = _training_grid(grid, cfg, observation_mask)
        # opacity_sigma consumes one RNG in source insertion order. The data
        # digest intentionally remains order-independent for historical records.
        source_order = list(clean.counts)
        key = hashlib.sha256(
            (
                source
                + _digest(nuisance_cfg or cfg)
                + _digest(cfg, clean, live, masks)
                + json.dumps(source_order)
                + (frozen_nuisance.choice_hash if frozen_nuisance else "fresh")
            ).encode()
        ).hexdigest()
        target = Path(cache_dir) / "completed-training" / key
        target.mkdir(parents=True, exist_ok=True)
        complete = target / "complete.json"
        payload = target / "fold.pickle"
        if complete.exists() and payload.exists():
            record = json.loads(complete.read_text())
            if record["payload_hash"] == hashlib.sha256(payload.read_bytes()).hexdigest():
                # Trusted local cache generated only by this campaign, never user input.
                return pickle.loads(payload.read_bytes())
        fold = fit_training_fold(
            clean,
            live,
            cfg,
            observation_mask=masks,
            cache_dir=target / "operator",
            frozen_nuisance=frozen_nuisance,
            nuisance_cfg=nuisance_cfg,
        )
        data = pickle.dumps(fold)
        temporary = payload.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(payload)
        write_json(
            complete,
            dict(
                scientific_hash=key,
                source_hash=source,
                source_order=source_order,
                cache_scope="ordered count sources (Poisson RNG consumption)",
                payload_hash=hashlib.sha256(data).hexdigest(),
            ),
        )
        return fold

    return fit


def slab_baseline(part, fold, predictions):
    """Fit a nonnegative uniform horizontal normal column on training only."""
    tx, ty = part["train"].tan_mesh()
    length = np.sqrt(1 + tx**2 + ty**2)
    sid = fold.cfg.sky_reference.id
    roof = part["train"].counts[sid]
    masks = part["train_masks"] or {p: np.ones(roof.shape, bool) for p in fold.cfg.position_ids}

    def objective(column):
        total = 0.0
        for pid in fold.cfg.position_ids:
            mu = roof * part["train_live"][pid] / part["train_live"][sid] * np.exp(-column * length)
            valid = masks[pid] & np.isfinite(predictions[pid]) & (mu > 0)
            total += poisson_deviance(part["train"].counts[pid][valid], mu[valid])
        return total

    result = minimize_scalar(objective, bounds=(0, 10), method="bounded")
    if not result.success:
        raise RuntimeError("slab baseline fit failed")
    return float(result.x), {
        p: np.where(
            np.isfinite(predictions[p]),
            roof * part["test_live"][p] / part["train_live"][sid] * np.exp(-result.x * length),
            np.nan,
        )
        for p in fold.cfg.position_ids
    }


def uncertainty_reference_scope(protocol, reference_id, roof_live_time):
    return dict(
        reference_id=reference_id,
        roof_live_time_s=roof_live_time,
        reference_sampling_scope="one Poisson roof draw shared across all room views per replica",
        reference_counting_scope=(
            "full-reference counting variation included"
            if protocol == "angular"
            else "training-reference counting variation included"
        ),
        point_score_reference_scope=(
            "conditional on observed full roof calibration"
            if protocol == "angular"
            else "conditional on observed training roof split"
        ),
        roof_counting_variation=True,
        roof_time_scope="unchanged full acquisition"
        if protocol == "angular"
        else "thinned training exposure",
    )


def register_fold_inputs(part, index, cfg, settings, out, manifest):
    directory = out / "folds" / part["name"]
    directory.mkdir(parents=True)
    clean, masks = _training_grid(part["train"], cfg, part["train_masks"])
    test_masks = part["test_masks"] or {
        p: np.ones((clean.n_bins, clean.n_bins), bool) for p in cfg.position_ids
    }
    np.savez_compressed(directory / "training-counts.npz", edges=clean.edges, **clean.counts)
    np.savez_compressed(
        directory / "test-counts.npz", edges=part["test"].edges, **part["test"].counts
    )
    np.savez_compressed(directory / "masks.npz", **masks)
    np.savez_compressed(directory / "test-masks.npz", **test_masks)
    fraction = settings["count"]["inner_train_fraction"]
    inner_train_live = {p: t * fraction for p, t in part["train_live"].items()}
    inner_test_live = {p: t * (1 - fraction) for p, t in part["train_live"].items()}
    if part["protocol"] == "angular":
        inner_train_live[cfg.sky_reference.id] = part["train_live"][cfg.sky_reference.id]
        inner_test_live[cfg.sky_reference.id] = part["train_live"][cfg.sky_reference.id]
    registration = dict(
        inner_fraction=fraction,
        inner_train_live=inner_train_live,
        inner_test_live=inner_test_live,
        name=part["name"],
        protocol=part["protocol"],
        partition_seed=part["seed"],
        inner_selection_seed=seed(settings, 3, index),
        train_live=part["train_live"],
        test_live=part["test_live"],
        roof_scope=part["roof_scope"],
        inner_roof_scope="full_reference_conditional"
        if part["protocol"] == "angular"
        else "inner_training_split",
        nuisance_scope=manifest["nuisance_scope"],
        nuisance_config_hash=_digest(cfg),
        source_hash=manifest["source_hash"],
        original_input_hash=manifest["input_hash"],
        training_input_hash=_digest(cfg, clean, part["train_live"], masks),
        raw_training_input_hash=_digest(cfg, part["train"], part["train_live"], masks),
        test_input_hash=_digest(cfg, part["test"], part["test_live"], test_masks),
        provenance=manifest["provenance"],
        training_archive_scope=(
            "excluded room bins sanitized to zero; masks define missingness, never measured zeros"
        ),
        test_archive_scope="raw observations; test-masks define evaluation assignment",
        train_masks="masks.npz",
        test_masks="test-masks.npz",
        uncertainty_reference_scope=uncertainty_reference_scope(
            part["protocol"], cfg.sky_reference.id, part["train_live"][cfg.sky_reference.id]
        ),
    )
    write_json(directory / "inputs.json", registration)
    return registration


def run_campaign(grid, live, cfg, document, *, out, cache, provenance=None):
    settings = settings_document(document)
    out, cache = Path(out).resolve(), Path(cache).resolve()
    protected_roots = {Path.cwd().resolve(), Path(__file__).resolve().parents[2]}
    if provenance and provenance.get("config_path"):
        protected_roots.add(Path(provenance["config_path"]).resolve().parent.parent)
    for protected in protected_roots:
        for path in (out, cache):
            if any(
                path == protected / name or (protected / name) in path.parents
                for name in (
                    "results",
                    "paper-draft",
                    "paper-production",
                    "configs",
                    "src",
                    "tests",
                )
            ):
                raise ValueError("output/cache overlaps protected production artifacts")
            production_runs = protected / "runs"
            isolated_runs = production_runs / "internal-validation"
            if production_runs in path.parents and not (
                path == isolated_runs or isolated_runs in path.parents
            ):
                raise ValueError("campaign paths inside runs must use runs/internal-validation")
            if path == protected or path in protected.parents:
                raise ValueError("campaign path cannot contain the repository")
    if (out / "manifest.json").exists():
        raise ValueError("use a new output root; scientific cache may be reused")
    if "choices" in settings["stability"]:
        from cafetomo.robustness import validate_choices

        validate_choices(settings["stability"]["choices"], cfg)
    start = time.perf_counter()
    manifest = dict(
        status="incomplete",
        mode=settings["mode"],
        publication_output=False,
        settings=settings,
        source_hash=source_hash(),
        config=asdict(cfg),
        input_hash=_digest(cfg, grid, live, {}),
        provenance=provenance or {},
        seed_streams={
            "count_outer": 1,
            "angular_outer": 2,
            "inner_selection": 3,
            "recovery": 4,
            "count_stability": 5,
            "choice_stability": 6,
            "prediction_uncertainty": 7,
        },
        nuisance_scope="original prior/search bounds and original selfcal objective throughout",
        retrospective=True,
        failures=[],
        folds=[],
        stages={},
    )
    write_json(out / "manifest.json", manifest)
    try:
        candidates = [
            replace(
                cfg,
                reconstruction=replace(
                    cfg.reconstruction, tv_alpha=a, coverage_damping=d, tv_z_weight=0
                ),
            )
            for a in settings["selection"]["tv_alpha"]
            for d in settings["selection"]["coverage_damping"]
        ]
        if settings["mode"] == "smoke":
            candidates = candidates[:1]
        predictions, support_records = [], []
        first = None
        uncertainty_protocols = set()
        planned_parts = list(partitions(grid, live, cfg, settings))
        manifest["registered_folds"] = [
            register_fold_inputs(part, index, cfg, settings, out, manifest)
            for index, part in enumerate(planned_parts)
        ]
        manifest["stages"].update({part["name"]: "registered" for part in planned_parts})
        write_json(out / "manifest.json", manifest)
        for index, part in enumerate(planned_parts):
            manifest["stages"][part["name"]] = "running"
            write_json(out / "manifest.json", manifest)
            stage_out = out / "folds" / part["name"]
            fold, chosen, scores = select_and_refit(
                part["train"],
                part["train_live"],
                candidates,
                cache_dir=cache,
                seed=seed(settings, 3, index),
                fraction=settings["count"]["inner_train_fraction"],
                observation_mask=part["train_masks"],
                shared_nuisance=True,
                nuisance_cfg=cfg,
                full_roof_id=cfg.sky_reference.id if part["protocol"] == "angular" else None,
                fitter=cached_fitter(manifest["source_hash"]),
            )
            mu = predict_raw_counts(fold, part["train"], part["train_live"], part["test_live"])
            records = score_raw_counts(part["test"].counts, mu, observation_mask=part["test_masks"])
            column, slab = slab_baseline(part, fold, mu)
            baseline = score_raw_counts(
                part["test"].counts, slab, observation_mask=part["test_masks"]
            )
            record = dict(
                name=part["name"],
                protocol=part["protocol"],
                seed=part["seed"],
                roof_scope=part["roof_scope"],
                point_score_reference_scope=manifest["registered_folds"][index][
                    "uncertainty_reference_scope"
                ]["point_score_reference_scope"],
                train_live=part["train_live"],
                test_live=part["test_live"],
                selected_index=chosen,
                inner_scores=scores,
                training=fold.metadata,
                train_hash=fold.train_hash,
                choice_hash=fold.choice_hash,
            )
            for pid in cfg.position_ids:
                np.savez_compressed(stage_out / f"{pid}-prediction.npz", **records[pid])
            np.save(stage_out / "rho.npy", fold.volume.rho3())
            write_json(stage_out / "solverinfo.json", fold.volume.info)
            if part["train_masks"] is not None:
                np.savez_compressed(stage_out / "masks.npz", **part["train_masks"])
                np.savez_compressed(stage_out / "test-masks.npz", **part["test_masks"])
            record["positions"] = {
                p: {k: v for k, v in records[p].items() if np.isscalar(v)} for p in records
            }
            record["slab_normal_column"] = column
            record["slab_deviance"] = sum(v["deviance"] for v in baseline.values())
            write_json(stage_out / "fold.json", record)
            predictions.append(record)
            manifest["folds"].append(record)
            write_json(out / "manifest.json", manifest)
            support_records.append(run_support(part, fold, stage_out, settings))
            if part["protocol"] not in uncertainty_protocols:
                n_uncertainty = settings.get("prediction_uncertainty", {}).get("replicas", 2)
                replicas = training_prediction_replicas(
                    part["train"],
                    part["train_live"],
                    replace(cfg, reconstruction=fold.cfg.reconstruction),
                    target_live=part["test_live"],
                    observation_mask=part["train_masks"],
                    cache_dir=cache,
                    nuisance_cfg=cfg,
                    artifact_dir=stage_out / "uncertainty-replicas",
                    reference_scope=manifest["registered_folds"][index][
                        "uncertainty_reference_scope"
                    ],
                    n_replicas=n_uncertainty,
                    seed=seed(settings, 7, index),
                )
                uncertainty = summarize_prediction_replicas(replicas)
                for pid, values in uncertainty.items():
                    np.savez_compressed(stage_out / f"{pid}-conditional-uncertainty.npz", **values)
                record["uncertainty_scope"] = "representative first outer fold per protocol"
                record["uncertainty_replicas"] = n_uncertainty
                record["uncertainty_reference_scope"] = manifest["registered_folds"][index][
                    "uncertainty_reference_scope"
                ]
                uncertainty_protocols.add(part["protocol"])
            manifest["stages"][part["name"]] = "complete"
            write_json(stage_out / "fold.json", record)
            write_json(out / "manifest.json", manifest)
            if first is None:
                part["original_cfg"] = cfg
                first = (part, fold)
        results = out / "pending-results"
        write_json(
            results / "prediction.json",
            dict(
                folds=predictions,
                deviance=sum(v["deviance"] for r in predictions for v in r["positions"].values()),
                covered_bins=sum(
                    v["n_covered"] for r in predictions for v in r["positions"].values()
                ),
                interpretation=(
                    "retrospective conditional prediction; common raw coverage slab comparison"
                ),
            ),
        )
        write_json(
            results / "support.json",
            dict(
                folds=support_records, fold_count=len(support_records), envelope_approximation=True
            ),
        )
        manifest["stages"]["recovery"] = "running"
        write_json(out / "manifest.json", manifest)
        run_recovery(grid, live, cfg, settings, out, results)
        manifest["stages"]["recovery"] = "complete"
        manifest["stages"]["stability"] = "running"
        write_json(out / "manifest.json", manifest)
        run_stability(first, settings, out, results)
        manifest["stages"]["stability"] = "complete"
        results.rename(out / "results")
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["failures"].append(dict(type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        manifest["runtime_s"] = time.perf_counter() - start
        manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(out / "manifest.json", manifest)
    return manifest


def run_support(part, fold, stage_out, settings):
    from cafetomo.robustness import training_data
    from cafetomo.support import feature_mask_diagnostics, geometry_support, support_rows

    data = training_data(part["train"], part["train_live"], fold, part["train_masks"])
    np.savez_compressed(
        stage_out / "training-weights.npz",
        w=data.w,
        sx=data.rows.sx,
        sy=data.rows.sy,
        pos_of_row=data.rows.pos_of_row,
    )
    records = []
    for degree in settings["features"]["background_degree"]:
        for threshold in settings["features"]["signal_to_error"]:
            diagnostics = feature_mask_diagnostics(
                data, background_degree=degree, threshold=threshold
            )
            rows, masks = support_rows(data, diagnostics)
            support = geometry_support(rows, masks, fold.cfg, fold.volume.grid, tangent_sigma=0)
            name = f"support-degree{degree}-threshold{threshold:g}"
            np.savez_compressed(
                stage_out / f"{name}.npz",
                joint=support.joint_support,
                coverage=support.jointly_covered,
                **diagnostics.masks,
                **{f"coefficients_{p}": v for p, v in diagnostics.coefficients.items()},
                **{f"excluded_{p}": v for p, v in diagnostics.excluded.items()},
                **{f"eligible_{p}": v for p, v in diagnostics.eligible.items()},
                powers=np.asarray(diagnostics.powers, dtype=np.int64),
            )
            records.append(
                dict(
                    degree=degree,
                    threshold=threshold,
                    jointly_covered=int(support.jointly_covered.sum()),
                    significant_rows={p: int(v.sum()) for p, v in diagnostics.masks.items()},
                )
            )
    return dict(
        coincidence_width_diagnostic=coincidence_width_diagnostic(part["train"], fold.cfg),
        fold=part["name"],
        sensitivity=records,
        selection_scope="all predeclared thresholds reported; no outer outcome selection",
        envelope_approximation=True,
    )


def run_recovery(grid, live, cfg, settings, out, results):
    from cafetomo.recovery import (
        direct_controls,
        precision_budget,
        run_recovery_case,
        simulate_counts,
        standard_cases,
    )

    recovery = settings["recovery"]
    n = 1 if settings["mode"] == "smoke" else recovery["noise_replicas_per_case"]
    cases = (
        standard_cases()
        if settings["mode"] == "full"
        else [standard_cases()[0], standard_cases()[3]]
    )
    budget = precision_budget(n)
    write_json(out / "recovery" / "budget.json", budget)
    direct = direct_controls(
        cfg, out_dir=out / "recovery" / "direct", n_sub=recovery.get("truth_subrays", 16)
    )
    records = []
    for i, case in enumerate(cases):
        for replica in range(n):
            stream = seed(settings, 4, i, replica)
            synthetic = simulate_counts(
                case, cfg, grid, live, seed=stream, n_sub=recovery.get("truth_subrays", 16)
            )
            records.append(
                run_recovery_case(
                    case,
                    synthetic,
                    live,
                    cfg,
                    seed=stream,
                    out_dir=out / "recovery" / case.case_id / str(replica),
                )
            )
    write_json(
        results / "recovery.json",
        dict(
            noisy_outcomes=len(records),
            fit_failures=sum(r.get("status") == "fit_failure" for r in records),
            direct_controls=direct,
            cases=records,
            budget=budget,
            convergence_scope=(
                "converged refers to box fit only; voxel fit uses a fixed iteration budget, "
                "with per-case solver diagnostics in linked recovery artifacts"
            ),
        ),
    )


def run_stability(first, settings, out, results):
    from cafetomo.robustness import run_count_replicas, run_robustness

    part, fold = first
    prior_cfg = replace(part["original_cfg"], reconstruction=fold.cfg.reconstruction)
    clean, _ = _training_grid(part["train"], prior_cfg, part["train_masks"])
    n = 1 if settings["mode"] == "smoke" else settings["stability"]["count_replicas"]
    threshold = settings["stability"]["contrast_threshold"]
    count = run_count_replicas(
        clean,
        part["train_live"],
        prior_cfg,
        n_replicas=n,
        seed=seed(settings, 5),
        out_dir=out / "stability" / "count",
        contrast_threshold=threshold,
        pose_mode="reestimated_pose",
        nuisance_cfg=part["original_cfg"],
        observation_mask=part["train_masks"],
    )
    choices = settings["stability"].get(
        "choices",
        [
            dict(
                group="algorithm",
                overrides={"reconstruction": {"tv_alpha": a}},
                contrast_threshold=threshold,
            )
            for a in settings["selection"]["tv_alpha"]
        ],
    )
    choices = list(choices)
    skipped = []
    for parameter, values in settings["stability"].get("algorithm_target_sweeps", {}).items():
        if parameter not in {"tv_alpha", "coverage_damping"}:
            raise ValueError("unsupported algorithm sweep")
        for value in values:
            if value == getattr(fold.cfg.reconstruction, parameter):
                skipped.append(
                    dict(parameter=parameter, value=value, reason="identical to baseline")
                )
                continue
            choices.append(
                dict(
                    group="algorithm",
                    contrast_threshold=threshold,
                    overrides={"reconstruction": {parameter: value}},
                )
            )
    sensitivity = run_robustness(
        clean,
        part["train_live"],
        prior_cfg,
        choices,
        seed=seed(settings, 6),
        out_dir=out / "stability" / "choices",
        nuisance_cfg=part["original_cfg"],
    )
    write_json(
        results / "stability.json",
        dict(
            count_replicas=n,
            count=count,
            choices=sensitivity,
            duplicate_baseline_targets=skipped,
            scope=part["name"],
            conditional_accuracy=False,
            interpretation=(
                "density persistence and background-subtracted beam excess remain distinct"
            ),
        ),
    )


def coincidence_width_diagnostic(grid, cfg):
    """Raw angular corners quantify the bin-centre footprint approximation."""
    tx, ty = grid.tan_mesh()
    aperture = cfg.detector.aperture_m
    dz = cfg.detector.layer_dz_cm / 100
    centres = np.stack([tx, ty], axis=-1)
    widths = aperture - dz * np.abs(centres)
    accepted = (widths > 0).all(axis=-1)
    lower, upper = grid.edges[:-1], grid.edges[1:]
    variations = []
    crossing = np.zeros(tx.shape, bool)
    for x in (lower, upper):
        for y in (lower, upper):
            xx, yy = np.meshgrid(x, y, indexing="ij")
            corner = np.maximum(0, aperture - dz * np.abs(np.stack([xx, yy], axis=-1)))
            crossing |= (corner <= 0).any(axis=-1)
            relative = np.abs(corner - widths) / np.maximum(widths, np.finfo(float).eps)
            variations.append(relative.max(axis=-1))
    maximum = np.maximum.reduce(variations)
    near_boundary = accepted & (widths.min(axis=-1) <= aperture * 0.25)
    return dict(
        accepted_bins=int(accepted.sum()),
        boundary_corner_crossings=int((accepted & crossing).sum()),
        maximum_relative_width_change=float(maximum[accepted].max()) if accepted.any() else None,
        near_boundary_bins=int(near_boundary.sum()),
        near_boundary_maximum_relative_width_change=float(maximum[near_boundary].max())
        if near_boundary.any()
        else None,
        interpretation=(
            "bin-centre coincidence rectangle; angular support uses envelope approximation"
        ),
    )
