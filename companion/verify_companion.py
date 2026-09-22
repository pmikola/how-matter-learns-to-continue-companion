"""Read-only integrity and reader-mode smoke checks for the local v40 bundle.

Outputs JSON to stdout. This is not whole-book or experimental certification.
Run from any directory: paths are located relative to this script.
"""
from hashlib import sha256
from itertools import product
from math import comb, isclose, cos, sin, sqrt
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, tolerance=1e-10):
    require(isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            f"{label}: {actual} != {expected}")


def integrity():
    manifest = json.loads((ROOT / "bundle-manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        require(path.is_relative_to(ROOT), "Manifest path escapes bundle")
        content = path.read_bytes()
        require(len(content) == entry["bytes"] and sha256(content).hexdigest() == entry["sha256"],
                f"Integrity mismatch: {entry['path']}")
    return len(manifest["files"])


def check_frozen_energy():
    """Read archived arrays and re-evaluate formulas, never import the producer."""
    arrays_path = ROOT / "outputs/investigation-v24/figure-data.json"
    source_path = ROOT / "examples/investigation_v24.py"
    require(source_path.is_file(), "Frozen v24 provenance source missing")
    data = json.loads(arrays_path.read_text(encoding="utf-8"))["conserved-energy"]
    keys = ("t_s", "q1_m", "q2_m", "kinetic_J", "wall_J", "coupling_J")
    require(all(len(data[k]) == 1001 for k in keys), "Energy-array lengths")
    close(data["total_J"], .75, "Recorded total energy")
    errors, conservation = [], []
    w = sqrt(2)
    for i,t in enumerate(data["t_s"]):
        close(t, 22*i/1000, "Saved time grid", 1e-13)
        q1,q2 = .5*(cos(t)+cos(w*t)), .5*(cos(t)-cos(w*t))
        v1,v2 = -.5*(sin(t)+w*sin(w*t)), -.5*(sin(t)-w*sin(w*t))
        expected = (q1,q2,.5*(v1*v1+v2*v2),.5*(q1*q1+q2*q2),.25*(q1-q2)**2)
        for key,value in zip(keys[1:],expected):
            errors.append(abs(data[key][i]-value))
            close(data[key][i],value,"Frozen energy array "+key,1e-12)
        total = sum(data[k][i] for k in ("kinetic_J","wall_J","coupling_J"))
        conservation.append(abs(total-.75))
        close(total,.75,"Conserved archived total",1e-12)
    return {"passed":True,"sample_count":1001,"expected_total_J":.75,
            "max_formula_error":max(errors),"max_conservation_error_J":max(conservation),
            "source_sha256":sha256(source_path.read_bytes()).hexdigest(),
            "arrays_sha256":sha256(arrays_path.read_bytes()).hexdigest(),
            "producer_imported_or_executed":False,"files_written":False}


def tree_fingerprint():
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest()
            for p in ROOT.rglob("*") if p.is_file()}


def main():
    import numpy as np
    tree_before = tree_fingerprint()
    file_count = integrity()
    energy = check_frozen_energy()
    # Same sparse synchronous rule as the Chapter 8 notes, without a producer.
    from collections import Counter
    def life_step(active):
        counts = Counter((i+di,j+dj) for i,j in active
                         for di in (-1,0,1) for dj in (-1,0,1) if di or dj)
        return {site for site,n in counts.items() if n == 3 or (n == 2 and site in active)}
    glider = {(1,2),(2,3),(3,1),(3,2),(3,3)}
    altered = glider - {(1,2)}
    for _ in range(3):
        altered = life_step(altered)
    still = {(2,2),(2,3),(3,1),(3,4),(4,2),(4,3)}
    require(altered == still and life_step(altered) == still, "Glider deletion still life by update three")
    glider_deletion = {"deleted_site":[1,2], "update_three":sorted(map(list,still)),
                      "update_four_unchanged":True, "rule_changed":False, "training":False}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    runs = []

    def run(relative, arguments=(), structured=True):
        command = [sys.executable, "-B", str(ROOT / relative), *arguments]
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                capture_output=True, check=True, timeout=30)
        output = result.stdout.strip()
        runs.append({"entrypoint": relative, "arguments": list(arguments),
                     "exit_code": result.returncode, "stdout": output})
        return json.loads(output) if structured else output

    stroke = "examples/ch01/investigation_v9.py"
    baseline = run(stroke, ["--json"])
    close(baseline["before"]["probability_vertical"], .99809, "Trained stroke probability", 5e-6)
    pixel = run(stroke, ["--pixel", "5", "0.5", "--json"])
    require(pixel["input"][4] == .5 and pixel["training_update_requested"] is False, "Probe-only pixel edit")
    manual = run(stroke, ["--weight-delta", "0.01", "--json"])
    close(manual["before"]["pixel_2_to_unit_1_weight"] - baseline["before"]["pixel_2_to_unit_1_weight"], .01, "Copied-weight edit")
    update = run(stroke, ["--stage", "300", "--step", "--json"])
    close(update["mean_training_loss_before"], .3579801374061621, "Stage300 batch loss")
    close(update["mean_training_loss_after"], .35671638695637703, "Stage301 batch loss")

    connected = "examples/ch01_04/reader_lab_v11.py"
    represented = run(connected, ["representation", "--index", "50"])
    close(represented["original_score"], -.2006584304644301, "PointB score")
    require(represented["score_error"] < 1e-10 and represented["chosen_label"] == 0 and represented["true_label"] == 1, "PointB retained misclassification")
    run(connected, ["motion", "--current", "0"])
    motion = run(connected, ["motion", "--previous", "-1", "--current", "0"])
    close(motion["hand_built_prediction"], 1., "Constant-velocity hand prediction")
    evidence = run(connected, ["sensor", "--readings", "-0.6", "0.6"])
    close(evidence["stages"][-1]["posterior_right"], .5, "Opposing readings")
    run(connected, ["sensor", "--sigma", "1.5"])
    run(connected, ["sensor", "--prior", "0.2"])

    analytic = "reader_lab_v13.py"
    close(run(analytic, ["walk", "--steps", "16"])["best_accuracy"], .918622171273455, "Walk16")
    close(run(analytic, ["walk", "--steps", "96"])["best_accuracy"], .7092692375999154, "Walk96")
    run(analytic, ["walk", "--steps", "96", "--separation", "12"])
    run(analytic, ["phase", "--gap", "1.8"])
    opposite = run(analytic, ["phase", "--gap", "3.141592653589793"])
    close(opposite["final_gap"], opposite["initial_gap"], "Opposite equilibrium")
    uncoupled = run(analytic, ["phase", "--gap", "1.8", "--coupling", "0"])
    close(uncoupled["final_gap"], 1.8, "Uncoupled gap")
    close(run(analytic, ["sensor", "--mode", "x"])["best_accuracy"], .5, "Mirror ambiguity")
    run(analytic, ["sensor", "--mode", "xy"])
    run(analytic, ["sensor", "--mode", "xy", "--sx", "0.2", "--sy", "0.6"])
    run(analytic, ["sensor", "--angle-a", "0", "--angle-b", "0.12"])
    run(analytic, ["sensor", "--readings", "10"])
    for p, q, expected in ((.45, .5, .6922359155005648), (.01, .06, .9320070105399788)):
        result = run(analytic, ["bernoulli", "--p", str(p), "--q", str(q), "--readings", "100"])
        close(result["best_accuracy"], expected, "Binomial100 accuracy")
        # Independent convolution checks the log-gamma implementation's count law.
        a, b = np.array([1.]), np.array([1.])
        for _ in range(100):
            a, b = np.convolve(a, [1-p,p]), np.convolve(b, [1-q,q])
        close(.5*np.maximum(a,b).sum(), expected, "Independent convolution accuracy")
        # At N8 compare the count decision to every one of 256 ordered sequences.
        ordered = .5*sum(max(p**sum(bits)*(1-p)**(8-sum(bits)), q**sum(bits)*(1-q)**(8-sum(bits))) for bits in product((0,1),repeat=8))
        counted = .5*sum(comb(8,k)*max(p**k*(1-p)**(8-k),q**k*(1-q)**(8-k)) for k in range(9))
        close(ordered, counted, "Ordered/count decision equivalence")
    # Check the full-path form as well as the short alias printed in Chapters5–7.
    full_path = run("examples/ch05_07/reader_lab_v13.py", ["walk", "--steps", "16"])
    close(full_path["best_accuracy"], .918622171273455, "Canonical analytic path")

    assignment = run("outputs/publication-v29/ch10/assignment_check.py", structured=False)
    require("0.2993803913" in assignment and "0.3710703619" in assignment, "Assignment result")
    continuation = run("outputs/publication-v29/ch11/continuation_checks.py", structured=False)
    require("m=1.2835, y=0.7165" in continuation and "M(35)=30.1194" in continuation, "Continuation result")
    survival = run("outputs/publication-v29/ch13/survival_checks.py", structured=False)
    require("0.3660323" in survival and "0.334269, 0.033427, 0.632305" in survival, "Survival result")

    learned = run("examples/resource_learning_v31.py")
    require(learned["verified"] and learned["writes"] is False, "Read-only learned-policy check")
    model = learned["model"]
    recorded = json.loads((ROOT / "reviews/publication-v31/learning-resource-model.json").read_text(encoding="utf-8"))
    for key in ("calibration_error", "conditional_continuation", "strategies", "independent_verification"):
        require(model[key] == recorded[key], "Resource record agreement: "+key)
    close(model["strategies"]["learn_five_labels"]["decimal"], .990098261528608, "Learned continuation", 1e-14)
    close(model["strategies"]["learn_then_unannounced_reversal"]["decimal"], .010373401471392, "Reversed environment", 1e-14)
    compared = run("examples/model_comparison_v31.py")
    close(compared["passive_Y"]["total_variation"], 0., "Passive model distance")
    close(compared["passive_Y"]["best_equal_prior_single_draw_success"], .5, "Passive decoder")
    close(compared["force_X_on_then_read_Y"]["total_variation"], .5, "Intervention model distance")
    close(compared["force_X_on_then_read_Y"]["best_equal_prior_single_draw_success"], .75, "Intervention decoder")

    e3 = run("verify_e3.py")
    require(e3["passed"] and e3["checks"]["baseline_artifacts_verified"] == 8 and e3["outcome"] == "e3_transfer_null", "E3 frozen evidence")
    framework_record_check = run("verify_framework_record.py")
    require(framework_record_check["passed"], "Inherited framework-record consistency")
    require(run("examples/memory_reproduction_v39.py")["passed"], "Frozen global memory comparison")
    require(run("examples/memory_reproduction_v39.py", ["--retrain"])["passed"], "In-memory memory-training reproduction")
    require(run("examples/exploratory/local_rule_profiles_v39.py")["passed"], "Exploratory local profiles, separate from E3")
    require(run("verify_v39_evidence.py")["passed"], "Retained v39 record consistency")
    current_binding = json.loads((ROOT / "evidence/framework-forward-snippets.json").read_text(encoding="utf-8"))
    require(sha256((ROOT / current_binding["source"]).read_bytes()).hexdigest() == current_binding["source_sha256"], "Current v40 notes source binding")
    precision_record = json.loads((ROOT / "reviews/publication-v39/framework-precision.json").read_text(encoding="utf-8"))
    for configuration in precision_record["configurations"]:
        current_snippet = current_binding["snippets"][configuration["framework"]]
        require(sha256(current_snippet.encode()).hexdigest() == configuration["printed_snippet_sha256"], "Current v40 snippet differs from retained v39 configuration")
    require(integrity() == file_count, "Reader modes changed packaged files")
    tree_after = tree_fingerprint()
    require(tree_before == tree_after, "Reader modes added, removed or changed files in the extracted tree")
    result = {"passed": True, "python": sys.version.split()[0], "numpy": np.__version__,
              "working_directory": str(ROOT), "files_verified_before_and_after": file_count,
              "command_count": len(runs), "runs": runs,
              "frozen_energy_arrays":energy,"glider_deletion":glider_deletion,
              "extracted_tree_unchanged":True,"tree_files_verified_before_and_after":len(tree_before),
              "scope": "Fresh v40 portable reader runs, two in-memory Chapter 3 training reproductions using the original v39 reader, frozen arithmetic and a retained exploratory diagnostic. Inherited v34/v39 framework record consistency and current snippet bindings only; no fresh framework or E3 execution."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
