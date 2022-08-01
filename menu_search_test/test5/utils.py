import numpy as np
from interface import Interface

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

def minjerk_trajectory(t: float, t_total: float, start_point: np.array or list, end_point: np.array or list) -> np.array: 
    """
    Calculating the position of a mass point at time t along the minjerk trajectory
    Input
    _____
    t: current time
    t_total: total time taken from start point to end point
    start_point: start point of the trajectory
    end_point: end point of the trajectory

    Return
    ______
    The position of the mass point at time t
    """
    t = t/t_total
    x = start_point[0] + (end_point[0] - start_point[0]) * (6 * t**5 - 15 * t**4 + 10 * t**3)
    y = start_point[1] + (end_point[1] - start_point[1]) * (6 * t**5 - 15 * t**4 + 10 * t**3)
    return np.array([x, y])

def jerk_of_minjerk_trajectory(t_total: float, start_point: np.array or list, end_point: np.array or list) -> float:
    jerk_x = 720 * (end_point[0] - start_point[0])**2 / t_total**5
    jerk_y = 720 * (end_point[1] - start_point[1])**2 / t_total**5
    return jerk_x + jerk_y

def trial_name_string(trial) -> str:
    env_config = trial.config["env_config"]
    keys = list(env_config.keys())
    trial_name = f"{trial.trial_id}"
    for key in keys:
        trial_name += f"-{key}_{env_config[key]}"
    return trial_name

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def fast_intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def compute_width_distance_fast(position, interface: Interface, button_id):
    x = interface.button_normalized_position(interface.ui[button_id])[0]
    y = interface.button_normalized_position(interface.ui[button_id])[1]
    h = interface.button_normalized_size(interface.ui[button_id])[0]
    w = interface.button_normalized_size(interface.ui[button_id])[1]
    target_center = np.asarray(
        [x + h / 2, y + w / 2])
    L1 = line(position, target_center)
    points = np.asarray(
        [[x, x + h, x + h, x, x],
         [y, y, y + w, y + w, y]])

    intersections = []
    for i in range(4):
        A = points[:, i]
        B = points[:, i + 1]
        L2 = line(A, B)
        R = fast_intersection(L1, L2)
        if R:
            x = R[0]
            y = R[1]
            x_on_segment = A[0] - 1e-3 <= x <= B[0] + 1e-3 or B[0] - 1e-3 <= x <= A[0] + 1e-3
            y_on_segment = A[1] - 1e-3 <= y <= B[1] + 1e-3 or B[1] - 1e-3 <= y <= A[1] + 1e-3
            on_segement = x_on_segment and y_on_segment
            if on_segement:
                intersections.append(R)

    intersections = np.asarray(intersections)
    width = calc_distance(
        intersections[0, :],
        intersections[1, :],
        'l1',
    )
    distance = calc_distance(
        position,
        target_center,
        'l1',
    )
    return width, distance