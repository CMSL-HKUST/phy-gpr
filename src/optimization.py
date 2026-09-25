from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from .data_prep import load_data
from . import model_runner as gp
from .utils import DEFAULT_OUTPUT_DIR, RANDOM_SEED, ensure_output_dirs, set_seed

TARGET_LOSS_TOL = 1e-6


@dataclass(frozen=True)
class SelectedOptimizationConfig:
    selected_repeat: int = 5
    test_groups: int = 6
    ng: int = 3
    restarts: int = 5
    output_dir: Path = DEFAULT_OUTPUT_DIR / "selected_model"
    group_ids: tuple[int, ...] = (5, 18, 29)


class _TargetReached(Exception):
    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, dtype=float)


def _selected_split(df: pd.DataFrame, seed: int, selected_repeat: int, test_groups: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = None
    for _ in range(1, selected_repeat + 1):
        ids = rng.choice(df.group_id.to_numpy(), test_groups, replace=False)
    if ids is None:
        raise ValueError("selected_repeat must be at least 1")
    train = df[~df.group_id.isin(ids)].reset_index(drop=True)
    test = df[df.group_id.isin(ids)].reset_index(drop=True)
    return train, test, np.sort(ids.astype(int))


def train_selected_bundle(config: SelectedOptimizationConfig | None = None) -> dict:
    cfg = config or SelectedOptimizationConfig()
    set_seed(RANDOM_SEED)
    df = load_data()
    train, test, test_ids = _selected_split(df, RANDOM_SEED, cfg.selected_repeat, cfg.test_groups)

    xtr = train[["P", "v"]].to_numpy(float)
    xte = test[["P", "v"]].to_numpy(float)
    ytr = train.yield_mean.to_numpy(float)

    step1 = gp._step1(train, True, RANDOM_SEED, cfg.restarts)
    step1_all = gp._step1(df, True, RANDOM_SEED, cfg.restarts)
    physics_train, physics_test = gp._scale(gp._features(step1, xtr), gp._features(step1, xte))
    better_train, better_test = gp._scale(gp._features(step1_all, xtr), gp._features(step1_all, xte))

    x_scaler = StandardScaler().fit(xtr)
    y_scaler = StandardScaler().fit(ytr[:, None])
    zx = x_scaler.transform(xtr)
    zy = y_scaler.transform(ytr[:, None]).ravel()

    # Match the default selected-model protocol: USE_DIAG_ALPHA=False.
    alpha = np.zeros(len(train))
    physics_model = gp._physics(zx, zy, physics_train, alpha, False, ("h", "p"), RANDOM_SEED, cfg.restarts)
    process_model = gp._gp(xtr, ytr, False, alpha, RANDOM_SEED, restarts=cfg.restarts)

    return {
        "dataset": "al",
        "selected_repeat": cfg.selected_repeat,
        "test_group_ids": test_ids.tolist(),
        "train_groups": train,
        "test_groups": test,
        "step1_models": step1,
        "physics_model": physics_model,
        "process_model": process_model,
        "x_scaler_step2": x_scaler,
        "y_scaler_step2": y_scaler,
        "physics_features_train": physics_train,
        "physics_features_test": physics_test,
        "better_upstream_features_train": better_train,
        "better_upstream_features_test": better_test,
        "config": {
            "test_groups": cfg.test_groups,
            "ng": cfg.ng,
            "use_diag_alpha": False,
            "restarts": cfg.restarts,
        },
    }


def physics_predict(bundle: dict, x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    step1 = bundle["step1_models"]
    x_scaler = bundle["x_scaler_step2"]
    y_scaler = bundle["y_scaler_step2"]
    f_raw = gp._features(step1, x_raw)
    x_train_raw = bundle["train_groups"][["P", "v"]].to_numpy(float)
    f_train_raw = gp._features(step1, x_train_raw)
    _, features = gp._scale(f_train_raw, f_raw)

    x = x_scaler.transform(x_raw)
    model = bundle["physics_model"]
    k = gp._kernel(x, model["x"], features, model["f"], model["q"], model["active"], None)
    mean = k @ cho_solve(model["c"], model["y"], check_finite=False)
    v = cho_solve(model["c"], k.T, check_finite=False)
    kss = gp._kernel(x, x, features, features, model["q"], model["active"], None)
    var = np.maximum(np.diag(kss) - (k * v.T).sum(1), 1e-12)
    return y_scaler.inverse_transform(mean[:, None]).ravel(), np.sqrt(var) * y_scaler.scale_[0]


def _optimize_groups(bundle: dict, groups: pd.DataFrame, label: str, bounds: list[tuple[float, float]], x0: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    history = []
    for _, row in groups.iterrows():
        target = float(row.yield_mean)
        for model_name in ("physics", "process"):
            losses: list[float] = []

            def objective(x: np.ndarray) -> float:
                xx = np.asarray(x, dtype=float).reshape(1, 2)
                if model_name == "physics":
                    pred, _ = physics_predict(bundle, xx)
                else:
                    pred, _ = gp._pred(bundle["process_model"], xx)
                loss = float((pred[0] - target) ** 2)
                losses.append(loss)
                if loss <= TARGET_LOSS_TOL:
                    raise _TargetReached(np.asarray(x, dtype=float))
                return loss

            try:
                res = minimize(objective, x0.copy(), method="L-BFGS-B", bounds=bounds, options={"maxiter": 50, "maxfun": 50})
                xstar = np.asarray(res.x, dtype=float)
                stop_reason = "optimizer"
                success = bool(res.success)
                n_iterations = int(res.nit)
            except _TargetReached as reached:
                xstar = reached.x
                stop_reason = "target_reached"
                success = True
                n_iterations = 0

            if model_name == "physics":
                pred_star = float(physics_predict(bundle, xstar.reshape(1, 2))[0][0])
            else:
                pred_star = float(gp._pred(bundle["process_model"], xstar.reshape(1, 2))[0][0])

            summary.append(
                {
                    "split": label,
                    "group_id": int(row.group_id),
                    "parameter_set_excel": int(row.group_id) + 1,
                    "model": model_name,
                    "target_yield_strength": target,
                    "P0": float(x0[0]),
                    "v0": float(x0[1]),
                    "P_star": float(xstar[0]),
                    "v_star": float(xstar[1]),
                    "pred_yield_strength": pred_star,
                    "abs_deviation": abs(pred_star - target),
                    "input_shift": float(np.linalg.norm(xstar - x0)),
                    "success": success,
                    "n_iterations": n_iterations,
                    "n_function_evals": len(losses),
                    "stop_reason": stop_reason,
                    "target_loss_tolerance": TARGET_LOSS_TOL,
                }
            )
            history.extend(
                {
                    "split": label,
                    "group_id": int(row.group_id),
                    "parameter_set_excel": int(row.group_id) + 1,
                    "model": model_name,
                    "target_yield_strength": target,
                    "evaluation": i + 1,
                    "loss": loss,
                }
                for i, loss in enumerate(losses)
            )
    return pd.DataFrame(summary), pd.DataFrame(history)


def _plot_history(history: pd.DataFrame, groups: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3), squeeze=False)
    for j, gid in enumerate(sorted(groups.group_id.astype(int))):
        ax = axes[0, j]
        for model_name, color in [("physics", "tab:blue"), ("process", "tab:orange")]:
            q = history[(history.group_id == gid) & (history.model == model_name)]
            ax.plot(q.evaluation, q.loss, color=color, label=model_name)
        ax.set_yscale("log")
        ax.set_title(f"group {gid}")
        ax.set_xlabel("evaluation")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_pv_mean_variance(bundle: dict, raw: pd.DataFrame, path: Path) -> None:
    p = np.linspace(raw.P.min(), raw.P.max(), 100)
    v = np.linspace(raw.v.min(), raw.v.max(), 100)
    pp, vv = np.meshgrid(p, v)
    grid = np.c_[pp.ravel(), vv.ravel()]

    physics_mean, physics_std = physics_predict(bundle, grid)
    process_mean, process_std = gp._pred(bundle["process_model"], grid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], physics_mean, "Physics mean"),
        (axes[0, 1], physics_std**2, "Physics variance"),
        (axes[1, 0], process_mean, "Process-only mean"),
        (axes[1, 1], process_std**2, "Process-only variance"),
    ]
    for ax, z, title in panels:
        cf = ax.contourf(pp, vv, z.reshape(pp.shape), levels=20, cmap="viridis")
        ax.scatter(raw.P, raw.v, c="white", edgecolors="k", s=18)
        ax.set(xlabel="P (W)", ylabel="v (mm/s)", title=title)
        fig.colorbar(cf, ax=ax)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_selected_optimization(config: SelectedOptimizationConfig | None = None) -> dict:
    cfg = config or SelectedOptimizationConfig()
    ensure_output_dirs(cfg.output_dir)
    bundle = train_selected_bundle(cfg)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.concat([bundle["train_groups"], bundle["test_groups"]]).drop_duplicates("group_id")
    train = bundle["train_groups"]
    test = bundle["test_groups"]
    groups = test[test.group_id.isin(cfg.group_ids)].sort_values("group_id").reset_index(drop=True)
    missing = sorted(set(cfg.group_ids) - set(groups.group_id.astype(int)))
    if missing:
        raise ValueError(f"Selected repeat {cfg.selected_repeat} does not contain requested test group ids: {missing}")

    pv_path = out / "pv_mean_variance.png"
    _plot_pv_mean_variance(bundle, raw, pv_path)

    bounds = [(raw.P.min(), raw.P.max()), (raw.v.min(), raw.v.max())]
    x0 = np.array([train.P.mean(), train.v.mean()])
    _, history = _optimize_groups(bundle, groups, "test", bounds, x0)

    figure_path = out / "inverse_optimization_convergence.png"
    _plot_history(history, groups, figure_path, "AlSi10Mg: selected test groups")

    print(f"Saved mean/variance figure: {pv_path}")
    print(f"Saved optimization figure: {figure_path}")
    print(f"Selected repeat {cfg.selected_repeat}, plotted groups {list(cfg.group_ids)}")
    return {"figure_path": figure_path, "pv_path": pv_path, "output_dir": out, "bundle": bundle}
