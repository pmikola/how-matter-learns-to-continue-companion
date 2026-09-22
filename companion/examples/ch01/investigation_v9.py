"""Reader lab: inspect or change copies of the archived stroke model. Never writes files."""
from pathlib import Path
import argparse
import json
import numpy as np

BOOK = Path(__file__).resolve().parents[2]
ARCHIVE = BOOK / "outputs/chapter-one-draft/v3/experiment/data-and-model.npz"

def forward(x, params):
    W, b, v, c = params
    a = x @ W + b
    h = np.maximum(a, 0)
    z = h @ v + c
    p = np.exp(-np.logaddexp(0, -z))
    return a, h, z, p

def mean_loss(x, y, params):
    z = forward(x, params)[2]
    return float(np.mean(np.logaddexp(0, z) - y * z))

def update(x, y, params):
    W, b, v, c = params
    a, h, z, p = forward(x, params)
    dz = (p - y) / len(y)
    dh = dz[:, None] * v * (a > 0)
    grads = [x.T @ dh, dh.sum(axis=0), h.T @ dz, np.array(dz.sum())]
    return [value - .1 * grad for value, grad in zip(params, grads)]

def inspect(x, params):
    a, h, z, p = forward(x, params)
    return {
        "hidden_sums": a[0].tolist(),
        "hidden_responses": h[0].tolist(),
        "score": float(z[0]),
        "probability_vertical": float(p[0]),
        "chosen_label": "vertical" if z[0] >= 0 else "horizontal",
        "pixel_2_to_unit_1_weight": float(params[0][1, 0]),
    }

def run(stage="trained", pixel=None, weight_delta=0., step=False):
    with np.load(ARCHIVE) as archive:
        tx, ty = archive["train_x"].copy(), archive["train_y"].copy()
        prefix = "" if stage == "trained" else "initial_"
        params = [archive[prefix + k].copy() for k in ("W", "b", "v", "c")]
    if stage == "300":
        for _ in range(300):
            params = update(tx, ty, params)
    x = np.array([[0., 1., 0., 0., 1., 0., 0., 1., 0.]])
    if pixel is not None:
        index, value = pixel
        x[0, index - 1] = value
    params[0][1, 0] += weight_delta
    result = {
        "stage": stage, "input": x[0].tolist(),
        "manual_weight_delta": weight_delta,
        "training_update_requested": step,
        "before": inspect(x, params),
        "mean_training_loss_before": mean_loss(tx, ty, params),
    }
    if step:
        params = update(tx, ty, params)
        result["after"] = inspect(x, params)
        result["mean_training_loss_after"] = mean_loss(tx, ty, params)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["initial", "300", "trained"], default="trained")
    parser.add_argument("--pixel", nargs=2, type=float, metavar=("INDEX", "VALUE"))
    parser.add_argument("--weight-delta", type=float, default=0.)
    parser.add_argument("--step", action="store_true", help="One full-batch update on copied parameters.")
    parser.add_argument("--json", action="store_true", help="Machine-readable output for verification.")
    args = parser.parse_args()
    pixel = None
    if args.pixel:
        index, value = args.pixel
        if not (np.isfinite(index) and index.is_integer() and 1 <= index <= 9):
            parser.error("Pixel index must be an integer from 1 to 9.")
        if not (np.isfinite(value) and 0 <= value <= 1):
            parser.error("Pixel darkness must be between 0 and 1.")
        pixel = (int(index), value)
    if not np.isfinite(args.weight_delta):
        parser.error("Weight change must be finite.")
    result = run(args.stage, pixel, args.weight_delta, args.step)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("Archived stage:", result["stage"])
    print("Input grid:\n", np.array(result["input"]).reshape(3, 3))
    print("Manual change to the pixel-2 / unit-1 weight:", args.weight_delta)
    for label in ("before", "after"):
        if label not in result:
            continue
        item = result[label]
        print("\n" + label.upper())
        print("Sums before ReLU:", np.round(item["hidden_sums"], 5))
        print("Responses after ReLU:", np.round(item["hidden_responses"], 5))
        print(f"Score: {item['score']:.8f}")
        print(f"Probability assigned to vertical: {item['probability_vertical']:.8f}")
        print("Chosen label:", item["chosen_label"])
        print(f"Mean training loss: {result['mean_training_loss_' + label]:.8f}")
    print("\nNo archive was changed. Pixel edits affect the probe, not the training batch.")
    if not args.step:
        print("Prediction only. No learning update was performed.")

if __name__ == "__main__":
    main()
