"""Reproducible memory and uncertainty experiments for chapters 3 and 4.

Synthetic dimensionless kinematics, delta_t = 1. Fixed design and seeds;
held-out data are evaluated only after the predetermined training run.
The explicit shift register is a designed memory mechanism, NOT a trained RNN.
"""

from __future__ import annotations

import os

for _thread_setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_setting] = "1"

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader


BOOK = Path(__file__).resolve().parents[2]
OUT = BOOK / "outputs" / "chapters-three-six" / "v7" / "experiments-memory"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#777777"
WIDTH = 138 / 25.4
CONFIG = {
    "train_position_seed": 7321,
    "test_position_seed": 9248,
    "network_seed": 1701,
    "train_base_positions": 256,
    "test_base_positions": 512,
    "current_position_interval": [-2.0, 2.0],
    "velocities": [-1.0, 1.0],
    "time_interval": 1.0,
    "hidden_relu_units": 8,
    "input_divisor": 3.0,
    "full_batch_updates": 10000,
    "learning_rate": 0.03,
    "objective": "mean(0.5 * (prediction - target)**2)",
    "selection": "one fixed run per architecture; no validation or test selection",
    "noise_sigma": 0.75,
    "observed_previous_position": -0.6,
}


def generate(seed: int, base_positions: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    current = np.repeat(rng.uniform(-2.0, 2.0, base_positions), 2)
    velocity = np.tile(np.array([-1.0, 1.0]), base_positions)
    previous = current - velocity
    following = current + velocity
    return dict(previous=previous, current=current, velocity=velocity, following=following)


def initialize(input_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(CONFIG["network_seed"])
    return {
        "w1": rng.normal(0, np.sqrt(2 / input_size), (input_size, 8)),
        "b1": np.full(8, 0.3),
        "w2": rng.normal(0, np.sqrt(2 / 8), (8, 1)),
        "b2": np.zeros(1),
    }


def forward(x: np.ndarray, params: dict[str, np.ndarray]):
    preactivation = x @ params["w1"] + params["b1"]
    hidden = np.maximum(0, preactivation)
    prediction = hidden @ params["w2"] + params["b2"]
    return prediction, preactivation, hidden


def loss_gradients(x: np.ndarray, target: np.ndarray, params: dict[str, np.ndarray]):
    prediction, preactivation, hidden = forward(x, params)
    error = prediction - target
    loss = float(np.mean(0.5 * error**2))
    output_gradient = error / len(x)
    hidden_gradient = (output_gradient @ params["w2"].T) * (preactivation > 0)
    gradients = {
        "w1": x.T @ hidden_gradient,
        "b1": np.sum(hidden_gradient, axis=0),
        "w2": hidden.T @ output_gradient,
        "b2": np.sum(output_gradient, axis=0),
    }
    return loss, gradients


def check_gradients(input_size: int) -> dict[str, float | int]:
    rng = np.random.default_rng(8200 + input_size)
    x = rng.uniform(-0.7, 0.7, (7, input_size))
    target = rng.normal(size=(7, 1))
    params = initialize(input_size)
    _, preactivation, _ = forward(x, params)
    assert np.min(np.abs(preactivation)) > 1e-4, "Do not check a ReLU kink."
    _, gradients = loss_gradients(x, target, params)
    epsilon = 1e-6
    discrepancies = []
    for key, array in params.items():
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + epsilon
            plus = loss_gradients(x, target, params)[0]
            array[index] = original - epsilon
            minus = loss_gradients(x, target, params)[0]
            array[index] = original
            numeric = (plus - minus) / (2 * epsilon)
            discrepancies.append(abs(numeric - gradients[key][index]))
    maximum = float(max(discrepancies))
    assert maximum < 1e-7, maximum
    return {"parameters_checked": len(discrepancies), "max_absolute_error": maximum}


def features(data: dict[str, np.ndarray], history: bool) -> np.ndarray:
    columns = [data["previous"], data["current"]] if history else [data["current"]]
    return np.column_stack(columns) / CONFIG["input_divisor"]


def train(data: dict[str, np.ndarray], history: bool):
    x = features(data, history)
    target = data["following"][:, None]
    params = initialize(x.shape[1])
    records = []
    for step in range(CONFIG["full_batch_updates"] + 1):
        loss, gradients = loss_gradients(x, target, params)
        if step % 25 == 0:
            records.append((step, np.sqrt(2 * loss)))
        if step == CONFIG["full_batch_updates"]:
            break
        for key in params:
            params[key] -= CONFIG["learning_rate"] * gradients[key]
    return params, np.array(records)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction.ravel() - target.ravel()) ** 2)))


def posterior_right(observation: np.ndarray | float):
    # p(y|v=+1)=N(-1,sigma^2), p(y|v=-1)=N(+1,sigma^2), equal priors.
    return 1 / (1 + np.exp(2 * np.asarray(observation) / CONFIG["noise_sigma"] ** 2))


def binary_entropy(probability: np.ndarray | float):
    p = np.clip(np.asarray(probability), 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def designed_memory_checks() -> dict:
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    b = np.array([0.0, 1.0])
    readout = np.array([-1.0, 2.0])
    count = 0
    max_error = 0.0
    for start in (-2.0, -0.3, 0.0, 1.7):
        for velocity in (-1.0, 1.0):
            values = start + velocity * np.arange(12)
            state = values[:2].copy()
            for following in values[2:]:
                max_error = max(max_error, abs(float(readout @ state - following)))
                previous_current = state[1]
                state = a @ state + b * following
                assert np.allclose(state, [previous_current, following], atol=1e-12)
                count += 1
    assert max_error < 1e-12
    return {
        "status": "hand-designed exact shift register; not a trained recurrent neural network",
        "state": "s_n = [q_(n-1), q_n]",
        "update": "s_(n+1) = A s_n + B q_(n+1)",
        "A": a.tolist(), "B": b.tolist(), "readout": readout.tolist(),
        "checked_updates": count, "max_prediction_absolute_error": max_error,
    }


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.titleweight": "bold", "figure.titleweight": "bold",
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.65,
        "lines.linewidth": 1.5, "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
    })


def finish(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def figures(train_data, test_data, current_model, history_model, current_curve, history_curve, results):
    style()
    # Same current position really is compatible with two opposite trajectories.
    fig, ax = plt.subplots(figsize=(WIDTH, 87 / 25.4))
    for velocity, color, marker, label in [(1, BLUE, "o", "Rightward: v = +1"), (-1, ORANGE, "^", "Leftward: v = -1")]:
        ax.plot([-1, 0], [-velocity, 0], color=color, marker=marker, markersize=6, label=label)
        ax.plot([0, 1], [0, velocity], color=color, linestyle="--", marker=marker, markersize=6)
    ax.annotate("Same current position", xy=(0, 0), xytext=(0, -0.64), ha="center",
                arrowprops={"arrowstyle": "-", "color": GREY}, color="black", fontsize=9.5)
    ax.set(xticks=[-1, 0, 1], xticklabels=["Previous\n(n - 1)", "Current\n(n)", "Next\n(n + 1)"],
           yticks=[-1, 0, 1], ylabel="Position q", ylim=(-1.3, 1.6), xlim=(-1.13, 1.13))
    ax.axhline(0, color="#CCCCCC", linewidth=0.6, zorder=0)
    ax.legend(loc="upper center", frameon=False, ncol=1, handlelength=2)
    ax.set_title("One position, two possible directions", pad=9)
    finish(fig, "fig-M01-same-position-different-histories")

    fig, ax = plt.subplots(figsize=(WIDTH, 84 / 25.4))
    ax.semilogy(current_curve[:, 0], current_curve[:, 1], color=ORANGE, label="Current position only")
    ax.semilogy(history_curve[:, 0], history_curve[:, 1], color=BLUE, linestyle="--", label="Previous + current")
    ax.axhline(1, color=GREY, linestyle=":", linewidth=1)
    ax.set(xlabel="Full-batch gradient-descent updates", ylabel="Training RMSE (position units)",
           xlim=(0, 10000), ylim=(0.012, 3.2), xticks=[0, 2500, 5000, 7500, 10000])
    ax.ticklabel_format(axis="x", style="plain")
    ax.set_title("Measured training, with and without history", pad=9)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    finish(fig, "fig-M02-measured-learning-curves")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 89 / 25.4), sharex=True, sharey=True)
    for ax, model, history, title, key in zip(axes, [current_model, history_model], [False, True],
                                            ["Current only", "Two frames"], ["current_only", "history"]):
        prediction = forward(features(test_data, history), model)[0].ravel()
        for velocity, color, marker in [(-1, ORANGE, "^"), (1, BLUE, "o")]:
            mask = test_data["velocity"] == velocity
            ax.scatter(test_data["following"][mask][::4], prediction[mask][::4], color=color,
                       marker=marker, s=11, alpha=0.6, edgecolors="none")
        ax.plot([-3, 3], [-3, 3], color="black", linestyle="--", linewidth=0.9, zorder=0)
        ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[-3, 0, 3], yticks=[-3, 0, 3],
               xlabel="True next position")
        ax.set_title(title, pad=8)
        ax.text(0.05, 0.95, f"RMSE {results['models'][key]['test_rmse']:.4f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1})
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted next position")
    fig.suptitle("New trajectories: measured held-out predictions", fontsize=10.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=BLUE, label="v = +1"),
               plt.Line2D([], [], marker="^", linestyle="none", color=ORANGE, label="v = -1")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    finish(fig, "fig-M03-held-out-predictions")

    observation = CONFIG["observed_previous_position"]
    sigma = CONFIG["noise_sigma"]
    grid = np.linspace(-3, 3, 801)
    gaussian = lambda y, mean: np.exp(-0.5 * ((y - mean) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    p_right = float(posterior_right(observation))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 89 / 25.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(grid, gaussian(grid, -1), color=BLUE, label="Rightward")
    ax.plot(grid, gaussian(grid, 1), color=ORANGE, linestyle="--", label="Leftward")
    ax.plot([observation, observation], [0, 0.55], color="black", linestyle=":", linewidth=1)
    ax.plot(observation, gaussian(observation, -1), marker="o", color=BLUE, markersize=5)
    ax.plot(observation, gaussian(observation, 1), marker="^", color=ORANGE, markersize=5)
    ax.set(xlabel="Older measurement y", ylabel="Likelihood density", xlim=(-3, 3),
           xticks=[-3, 0, 3], ylim=(0, 0.72), yticks=[0, 0.3, 0.6])
    ax.set_title("Two hypotheses", pad=8)
    ax.legend(frameon=False, loc="upper center", handlelength=1.2)
    ax.text(observation - 0.1, 0.025, "y = -0.6", rotation=90, va="bottom", ha="right", fontsize=9.5)
    ax = axes[1]
    ax.bar([-1, 1], [1-p_right, p_right], width=0.65, color=[ORANGE, BLUE], edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.text(0, 0.52, "Prior", ha="center", va="bottom", fontsize=9.5, color=GREY)
    for x, prob in [(-1, 1-p_right), (1, p_right)]:
        ax.text(x, prob+0.035, f"{prob:.3f}", ha="center", fontsize=9.5)
    ax.set(xlabel="Next position", ylabel="Posterior probability", xticks=[-1, 1],
           xticklabels=["-1", "+1"], yticks=[0, 0.5, 1], ylim=(0, 1.07), xlim=(-1.65, 1.65))
    ax.set_title("Possible futures", pad=8)
    fig.suptitle("Current position is known exactly: q = 0", fontsize=10.5)
    finish(fig, "fig-M04-likelihoods-and-possible-futures")

    fig, ax = plt.subplots(figsize=(WIDTH, 80 / 25.4))
    entropy = binary_entropy(posterior_right(grid))
    observed_entropy = float(binary_entropy(p_right))
    ax.plot(grid, entropy, color="black")
    ax.axhline(1, color=GREY, linestyle=":", linewidth=1)
    ax.text(2.94, 1.018, "Prior uncertainty: 1 bit", ha="right", va="bottom", fontsize=9.5, color=GREY)
    ax.plot(observation, observed_entropy, color=BLUE, marker="o", markersize=6)
    ax.plot([observation, observation], [0, observed_entropy], color=BLUE, linestyle="--", linewidth=0.9)
    ax.annotate(f"y = -0.6\nH = {observed_entropy:.3f} bits", xy=(observation, observed_entropy),
                xytext=(-2.9, 0.73), fontsize=9.5,
                arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.8})
    ax.set(xlabel="Older measurement y", ylabel="Posterior entropy (bits)",
           xlim=(-3, 3), ylim=(0, 1.16), yticks=[0, 0.5, 1], xticks=[-3, -2, -1, 0, 1, 2, 3])
    ax.set_title("How strongly does a measurement resolve direction?", pad=9)
    finish(fig, "fig-M05-posterior-entropy")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_data = generate(CONFIG["train_position_seed"], CONFIG["train_base_positions"])
    test_data = generate(CONFIG["test_position_seed"], CONFIG["test_base_positions"])
    assert not np.intersect1d(train_data["current"], test_data["current"]).size
    for data in (train_data, test_data):
        assert np.array_equal(data["current"][::2], data["current"][1::2])
        assert np.allclose(2*data["current"] - data["previous"], data["following"], atol=1e-12)
        assert np.allclose(data["following"].reshape(-1, 2).mean(axis=1), data["current"][::2])
    gradient_checks = {"current_only": check_gradients(1), "history": check_gradients(2)}
    current_model, current_curve = train(train_data, history=False)
    history_model, history_curve = train(train_data, history=True)
    # Held-out predictions are made only now, after both fixed training runs.
    metrics = {}
    for key, model, history in [("current_only", current_model, False), ("history", history_model, True)]:
        metrics[key] = {
            "parameters": int(sum(value.size for value in model.values())),
            "train_rmse": rmse(forward(features(train_data, history), model)[0], train_data["following"]),
            "test_rmse": rmse(forward(features(test_data, history), model)[0], test_data["following"]),
        }
    y = CONFIG["observed_previous_position"]
    sigma = CONFIG["noise_sigma"]
    likelihood_right = float(np.exp(-0.5*((y+1)/sigma)**2) / (sigma*np.sqrt(2*np.pi)))
    likelihood_left = float(np.exp(-0.5*((y-1)/sigma)**2) / (sigma*np.sqrt(2*np.pi)))
    p_right = float(posterior_right(y))
    assert np.isclose(p_right, likelihood_right/(likelihood_right+likelihood_left), atol=1e-14)
    assert np.isclose(posterior_right(0), 0.5)
    assert np.isclose(posterior_right(-y), 1-p_right)
    assert np.isclose(binary_entropy(0.5), 1)
    baseline_rmse = rmse(test_data["current"], test_data["following"])
    assert np.isclose(baseline_rmse, 1)
    current_prediction = forward(features(test_data, False), current_model)[0].ravel()
    # Exact finite-sample decomposition because both velocities occur at every q.
    current_mse_decomposition_error = abs(
        np.mean((current_prediction-test_data["following"])**2)
        - (1 + np.mean((current_prediction-test_data["current"])**2))
    )
    assert current_mse_decomposition_error < 1e-12
    results = {
        "configuration": CONFIG,
        "data": {"train_trajectories": len(train_data["current"]), "test_trajectories": len(test_data["current"]),
                 "independent_unit": "a sampled current position with both equally weighted velocity trajectories",
                 "noiseless_experiment": True, "split": "independent position draws with no overlap"},
        "models": metrics,
        "current_only_conditional_mean": {"formula": "E[q_(n+1)|q_n] = q_n", "test_rmse": baseline_rmse,
                                          "analytic_population_rmse": 1.0},
        "exact_two_frame_predictor": {"formula": "q_(n+1) = 2*q_n - q_(n-1)",
                                      "test_rmse": rmse(2*test_data["current"]-test_data["previous"], test_data["following"])},
        "gradient_checks": gradient_checks,
        "current_only_mse_decomposition": {
            "identity": "MSE = 1 + mean((f(q_n)-q_n)^2) for the paired balanced test data",
            "absolute_check_error": float(current_mse_decomposition_error),
        },
        "designed_memory": designed_memory_checks(),
        "bayes": {"assumptions": "q_n=0 known exactly; v in {-1,+1}, equal prior; y=-v+N(0,sigma^2)",
                  "posterior_right_formula": "1/(1+exp(2*y/sigma^2))",
                  "likelihood_density_right": likelihood_right, "likelihood_density_left": likelihood_left,
                  "posterior_right": p_right, "posterior_left": 1-p_right,
                  "prior_entropy_bits": 1.0, "posterior_entropy_bits": float(binary_entropy(p_right)),
                  "realized_entropy_reduction_bits": float(1-binary_entropy(p_right)),
                  "note": "This realized entropy reduction is not an average mutual-information estimate."},
        "limitations": ["Synthetic constant-speed motion only; no acceleration, collisions, or physical uncertainty in part one.",
                        "One fixed seed per model; not a model-ranking study or estimate of typical performance.",
                        "The two-frame network is feedforward; the separate memory update is hand-designed.",
                        "Bayesian uncertainty refers to direction under the stated observation model, not thermodynamic entropy."],
        "figure_width_mm": 138,
        "figures": ["fig-M01-same-position-different-histories", "fig-M02-measured-learning-curves",
                    "fig-M03-held-out-predictions", "fig-M04-likelihoods-and-possible-futures", "fig-M05-posterior-entropy"],
    }
    payload = {f"train_{key}": value for key, value in train_data.items()}
    payload.update({f"test_{key}": value for key, value in test_data.items()})
    payload.update({f"current_model_{key}": value for key, value in current_model.items()})
    payload.update({f"history_model_{key}": value for key, value in history_model.items()})
    payload.update(current_learning_curve=current_curve, history_learning_curve=history_curve)
    payload["test_prediction_current"] = forward(features(test_data, False), current_model)[0].ravel()
    payload["test_prediction_history"] = forward(features(test_data, True), history_model)[0].ravel()
    np.savez_compressed(OUT / "data.npz", **payload)
    figures(train_data, test_data, current_model, history_model, current_curve, history_curve, results)
    pdf_checks = []
    for figure in results["figures"]:
        reader = PdfReader(OUT / f"{figure}.pdf")
        assert len(reader.pages) == 1
        page = reader.pages[0]
        width_mm = float(page.mediabox.width) * 25.4 / 72
        height_mm = float(page.mediabox.height) * 25.4 / 72
        assert abs(width_mm-138) < 1e-6 and height_mm <= 110
        assert len(page.extract_text()) > 30
        images = list(page.images)
        assert not images, "Scientific plots must remain vector PDF content."
        pdf_checks.append({"file": f"{figure}.pdf", "width_mm": width_mm,
                           "height_mm": height_mm, "embedded_raster_images": len(images)})
    results["pdf_checks"] = pdf_checks
    (OUT / "results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"models": metrics, "gradient_checks": gradient_checks, "bayes": results["bayes"],
                      "output_directory": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
