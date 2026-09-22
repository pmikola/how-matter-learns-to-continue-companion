"""Chapter 13 checks; default is read-only. --figure creates versioned assets.

Original models match examples/ch08_11/figures_v18.py; that archived generator
is never executed or imported. A separate no-repair exercise holds fatal risk
fixed and is not substituted for either historical plot.
"""
import argparse
from math import exp


def history(repair=0.2, intervals=1000):
    if not 0 <= repair <= 0.999:
        raise ValueError("repair must be in [0, 0.999]")
    transition = ((.979, .020, .001), (repair, .999-repair, .001), (0., 0., 1.))
    state = (1., 0., 0.)
    values = [state]
    for step in range(1, intervals + 1):
        state = tuple(sum(state[row] * transition[row][col] for row in range(3)) for col in range(3))
        assert abs(sum(state) - 1) < 1e-12
        assert abs(sum(state[:2]) - .999**step) < 1e-12
        values.append(state)
    return values


def figure():
    from pathlib import Path
    import hashlib
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(__file__).resolve().parent
    book = destination.parents[2]
    archive = book / "outputs/investigation-v18/data.npz"
    saved = np.load(archive)
    n = np.arange(1001)
    survival = .99**n
    repair = np.array(history())
    assert np.allclose(saved["survival"], survival, rtol=0, atol=1e-14)
    assert np.allclose(saved["repair_states"], repair, rtol=0, atol=1e-13)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
        "axes.labelsize":9, "axes.titlesize":9.5, "xtick.labelsize":8.5,
        "ytick.labelsize":8.5, "pdf.fonttype":42,
        "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2, 1, figsize=(138/25.4, 4.7), layout="constrained")
    blue, orange, ink, grey = "#0072B2", "#D55E00", "#202020", "#777777"
    axes[0].plot(n, survival, color=orange, label="Fixed fatal risk q = 0.01")
    axes[0].plot(n, np.exp(-.1*(1-2.**(-n.astype(float)))), color=blue, ls="--", label="Summable, decreasing fatal risks")
    axes[0].plot(n, np.ones_like(n), color=grey, ls=":", label="Zero fatal risk, stipulated")
    axes[0].set(xlabel="Number of intervals N", ylabel="Survival probability", ylim=(0,1.07), title="A. Survival depends on the assumed risk")
    axes[0].legend(frameon=False, loc="center right", fontsize=8.5)
    axes[1].plot(n, repair[:,0], color=blue, label="Functioning")
    axes[1].plot(n, repair[:,:2].sum(axis=1), color=ink, ls="--", label="Not irreversibly failed")
    axes[1].plot(n, repair[:,2], color=orange, ls=":", label="Irreversibly failed")
    axes[1].set(xlabel="Number of intervals N", ylabel="Probability", ylim=(0,1.07), title="B. Separate repair model: fatal risk q = 0.001")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.savefig(destination / "survival-and-repair.pdf", metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(destination / "survival-and-repair.png", dpi=160)
    plt.close(fig)
    provenance = {"kind":"analytic redraw", "source":"examples/ch08_11/figures_v18.py",
        "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
        "data_unchanged":True, "archive_arrays_checked":["survival","repair_states"],
        "change":"Panel identifiers, explicit separate-model label and lower-panel fatal risk; legible typography."}
    (destination / "survival-and-repair.json").write_text(json.dumps(provenance,indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", action="store_true", help="write only versioned Ch13 figure assets")
    args = parser.parse_args()
    print(f"Fixed risk: P100={.99**100:.7f}; P1000={.99**1000:.8f}")
    print(f"Decreasing-risk limit: {exp(-.1):.6f}")
    for repair in (.2, 0.):
        state = history(repair)[-1]
        print(f"repair={repair:.3f}: " + ", ".join(f"{value:.6f}" for value in state))
    if args.figure:
        figure()
