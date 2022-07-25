import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def lorenz(x: float, sigma: float = 0.001) -> float:
    return np.log(1 + (x / sigma) ** 2 / 2)

def calc_distance(
        x: np.array or list, y: np.array or list, method: str = "l1", sigma: float = 0.001
) -> float:
    if method == "l1":
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    elif method == "l2":
        return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    elif method == "ll2" or method == "lorenz":
        return lorenz(np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2), sigma)
    elif method == "inverse":
        return 1 / calc_distance(x, y, "l2")

def who_mt(mu: float, sigma: float) -> float:
    x0 = 0.092
    y0 = 0.0
    alpha = 0.6
    k = 0.12
    if mu == 0:
        mu = 0.0001
    mt = pow((k * pow(((sigma - y0) / mu), (alpha - 1))), 1 / alpha) + x0
    return mt

def compute_stochastic_position(new_position, old_position, sigma):
    mu = calc_distance(new_position, old_position, "l1")
    mt = who_mt(mu, sigma)
    sampled_position = np.clip(np.random.normal(new_position, sigma), 0.0, 1.0)
    return sampled_position, mt

def minjerk_trajectory(t, t_total, start_point, end_point):
    t = t/t_total
    x = start_point[0] + (end_point[0] - start_point[0]) * (6 * t**5 - 15 * t**4 + 10 * t**3)
    y = start_point[1] + (end_point[1] - start_point[1]) * (6 * t**5 - 15 * t**4 + 10 * t**3)
    return np.array([x, y])