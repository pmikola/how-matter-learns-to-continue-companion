"""Read-only analytic experiments. No training and no output-file writes."""
import argparse
import json
import math
import numpy as np


def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def walk(steps=96, separation=8):
    if not 0 <= steps <= 512 or separation not in range(2, 41, 2):
        raise ValueError('Use 0-512 updates and an even separation from 2 to 40.')
    transition = np.zeros((41, 41))
    for source in range(41):
        transition[source, source] = .5
        transition[(source-1) % 41, source] = .25
        transition[(source+1) % 41, source] = .25
    state = np.zeros((41, 2))
    state[20-separation//2, 0] = 1
    state[20+separation//2, 1] = 1
    for _ in range(steps):
        state = transition @ state
    mixture = state.mean(axis=1)
    return dict(updates=steps, starting_sites=[-separation//2, separation//2],
                possible_final_sites=int(np.count_nonzero(mixture)),
                position_entropy_bits=entropy(mixture),
                mutual_information_bits=entropy(mixture)-sum(entropy(state[:, j]) for j in (0, 1))/2,
                best_accuracy=float(np.max(state*.5, axis=1).sum()))


def phase(gap=1.8, coupling=.6, time=6.):
    if not all(math.isfinite(v) for v in (gap, coupling, time)):
        raise ValueError('Phase inputs must be finite.')
    if not -math.pi <= gap <= math.pi or not 0 <= coupling <= 10 or not 0 <= time <= 100:
        raise ValueError('Use gap in [-pi, pi], coupling in [0, 10], time in [0, 100].')
    final = gap if abs(gap) == math.pi else 2*math.atan(math.tan(gap/2)*math.exp(-2*coupling*time))
    return dict(initial_gap=gap, final_gap=final, coupling=coupling, time=time,
                alignment=abs(math.cos(final/2)), common_phase_change=time,
                status='Specified first-order phase rule, omega=1. Not a physical measurement.')


def sensor(angle_a=math.pi/3, angle_b=-math.pi/3, sx=.1, sy=.3, readings=1, mode='xy'):
    if not all(math.isfinite(v) for v in (angle_a, angle_b, sx, sy)):
        raise ValueError('Sensor inputs must be finite.')
    if not 1e-6 <= sx <= 1e6 or not 1e-6 <= sy <= 1e6 or not 1 <= readings <= 10000 or mode not in ('x', 'xy'):
        raise ValueError('Use positive noise scales 1e-6 to 1e6 and 1-10000 readings, mode x or xy.')
    means = np.array([[math.cos(t), math.sin(t)] for t in (angle_a, angle_b)])
    scales = np.array([sx, sy])
    dims = 1 if mode == 'x' else 2
    distance = float(np.linalg.norm(((means[0]-means[1])/scales)[:dims]))
    fisher = [math.sin(t)**2/sx**2 + (math.cos(t)**2/sy**2 if mode == 'xy' else 0) for t in (angle_a, angle_b)]
    accuracy = .5*(1+math.erf(math.sqrt(readings)*distance/(2*math.sqrt(2))))
    return dict(mode=mode, readings=readings, candidate_angles=[angle_a, angle_b],
                mean_readings=means[:, :dims].tolist(), noise_scales=scales[:dims].tolist(),
                standardized_mean_distance=distance, best_accuracy=accuracy,
                fisher_per_reading=fisher, status='Equal-prior fixed Gaussian candidates, independent repeated readings.')


def counts(n, p):
    if not 1 <= n <= 500 or not math.isfinite(p) or not 0 < p < 1:
        raise ValueError('Use 1-500 trials and an interior probability.')
    # Log probabilities avoid overflow in the binomial coefficient.
    return np.exp([math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                   + k*math.log(p)+(n-k)*math.log1p(-p) for k in range(n+1)])


def bernoulli(p=.45, q=.5, readings=100):
    a, b = counts(readings, p), counts(readings, q)
    return dict(p=p, q=q, readings=readings,
                best_accuracy=float(.5*np.maximum(a, b).sum()),
                fisher_rao_distance=2*abs(math.asin(math.sqrt(q))-math.asin(math.sqrt(p))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    w = sub.add_parser('walk')
    w.add_argument('--steps', type=int, default=96)
    w.add_argument('--separation', type=int, default=8)
    p = sub.add_parser('phase')
    p.add_argument('--gap', type=float, default=1.8)
    p.add_argument('--coupling', type=float, default=.6)
    p.add_argument('--time', type=float, default=6.)
    s = sub.add_parser('sensor')
    s.add_argument('--angle-a', type=float, default=math.pi/3)
    s.add_argument('--angle-b', type=float, default=-math.pi/3)
    s.add_argument('--sx', type=float, default=.1)
    s.add_argument('--sy', type=float, default=.3)
    s.add_argument('--readings', type=int, default=1)
    s.add_argument('--mode', choices=['x', 'xy'], default='xy')
    b = sub.add_parser('bernoulli')
    b.add_argument('--p', type=float, default=.45)
    b.add_argument('--q', type=float, default=.5)
    b.add_argument('--readings', type=int, default=100)
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        result = dict(walk=walk, phase=phase, sensor=sensor, bernoulli=bernoulli)[command](**args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
