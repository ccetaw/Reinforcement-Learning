import numpy as np
from sympy.geometry import *


def check_within_element(position: np.array or list, element: dict) -> bool:
    if (
            element["x"] + element["w"] > position[0] > element["x"]
            and element["y"] + element["h"] > position[1] > element["y"]
    ):
        return True
    return False


def generate_item_to_slot(n_items, n_slots) -> np.ndarray:
    assigned = np.zeros((n_items, n_slots), dtype=np.int8)
    assigned[:n_slots, :] = np.eye(n_slots, dtype=np.int8)
    np.random.shuffle(assigned)
    return assigned


def generate_goal(selection, n=1, curriculum=False):
    n_items = len(selection)

    if curriculum:
        n = np.random.randint(1, min(n, n_items)+1)
    elif n == 'random':
        n = np.random.randint(1, n_items + 1)

    indx = list(range(n_items))
    to_change = np.random.choice(indx, size=n, replace=False)
    for c in to_change:
        selection[c] = 1 - selection[c]
    return selection, len(to_change)


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


def lorenz(x: float, sigma: float = 0.001) -> float:
    return np.log(1 + (x / sigma) ** 2 / 2)


def clip_normalize_reward(reward, r_min, r_max):
    reward = max(min(reward, r_max), r_min)
    return (reward - r_min) / (r_max - r_min)


def scale(sigma, low, high):
    return (sigma * (high - low)) + low


def scale_sigma(sigma, low, high):
    return scale(sigma, low, high)


def coordinate_screen(
        coordinate: np.ndarray or list, width=1920.0, height=1080.0
) -> np.array:
    return np.multiply(coordinate, [width, height])


def coordinate_normalized(coordinate: np.ndarray or list, width=1920., height=1080.) -> np.array:
    if isinstance(coordinate, (np.ndarray, list)):
        return np.divide(coordinate, [width, height])
    else:
        for key in coordinate:
            if key in ["x", "w"]:
                coordinate[key] /= width
            else:
                coordinate[key] /= height
        return coordinate


def array_to_flat(array: np.ndarray) -> np.array:
    return array.flatten()


def flat_to_array(n_items, n_slots, one_hot: np.ndarray) -> np.array:
    return one_hot.reshape((n_items, n_slots))


def who_mt(mu: float, sigma: float) -> float:
    x0 = 0.092
    y0 = 0.0
    alpha = 0.6
    k = 0.12
    if mu == 0:
        mu = 0.0001
    mt = pow((k * pow(((sigma - y0) / mu), (alpha - 1))), 1 / alpha) + x0
    return mt


def obs_to_dict(obs: np.array, env) -> dict:
    obs_space = env.observation_dict_user_lower
    sorted_obs_space = sorted(obs_space.keys(), key=lambda x: x.lower())
    obs_dict = {}
    lower_key = 0
    for key in sorted_obs_space:
        length = sum(obs_space[key].shape)
        value = obs[lower_key: lower_key + length]
        lower_key += length
        obs_dict[key] = value
    return obs_dict


def compute_stochastic_position(new_position, old_position, sigma):
    mu = calc_distance(new_position, old_position, "l1")
    mt = who_mt(mu, sigma)
    sampled_position = np.clip(np.random.normal(new_position, sigma), 0.0, 1.0)
    return sampled_position, mt


def generate_target_assignment(n_slots: int) -> np.array:
    target = np.zeros(n_slots, dtype=np.int8)
    idx = np.random.randint(low=0, high=n_slots)
    target[idx] = 1
    return target


def generate_goal_assignment(n_items: int, n_attributes: int, uniform: bool = False) -> np.array:
    goal = generate_tool_selection(n_items, n_attributes, uniform)
    return goal


def generate_slot(interface) -> np.ndarray:
    slots = np.zeros((interface.n_slots, 4))
    x = 0
    w = 1 / interface.n_slots
    for i in range(len(slots)):
        slots[i, 0] = x
        slots[i, 1] = interface.y_pos
        slots[i, 2] = w
        slots[i, 3] = interface.menu_height
        x += w
    return slots


def generate_item_to_slot_assignment(n_items, n_slots) -> np.ndarray:
    assigned = np.zeros((n_items, n_slots), dtype=np.int8)
    assigned[:n_slots, :] = np.eye(n_slots, dtype=np.int8)
    np.random.shuffle(assigned)
    return assigned


def update_tool_selection(
        tool_selection: np.ndarray,
        item_to_slot: np.ndarray,
        target: int,
        n_items: int,
        n_slots: int,
        n_attr: int,
) -> np.ndarray:
    slot = item_to_slot[:, target]
    item = np.where(slot == 1)[0]
    if item.size == 0:
        return tool_selection

    item = item[0]
    items_per_attr = [3, 3, 2, 2]
    for i in range(n_attr):
        bounds = [sum(items_per_attr[:i]), sum(items_per_attr[:(i + 1)])]
        if bounds[0] <= item < bounds[1]:
            tool_selection[bounds[0]:bounds[1]] = 0
            tool_selection[item] = 1
            break
    return tool_selection


def generate_tool_selection(n_items: int, n_attributes: int, uniform: bool = False) -> np.ndarray:
    items_per_attr = [3, 3, 2, 2]
    tool_selection = np.zeros((n_items,), dtype=np.int8)
    for i in range(n_attributes):
        tool_selection[sum(items_per_attr[:i]) + np.random.randint(items_per_attr[i])] = 1
    return tool_selection


def setup_expirement(n_items, n_attributes):
    n_differences = np.random.randint(1, 5)
    tools = generate_tool_selection(n_items, n_attributes)
    goal = tools.copy()
    attr_order = [0, 1, 2, 3]
    np.random.shuffle(attr_order)
    items_per_attr = [3, 3, 2, 2]
    for diff in range(n_differences):
        attr = attr_order[diff]
        bounds = [sum(items_per_attr[:attr]), sum(items_per_attr[:attr + 1])]
        goal[bounds[0]:bounds[1]] = 0
        tool = np.where(tools[bounds[0]:bounds[1]] == 1)[0][0]
        pos = list(range(items_per_attr[attr]))
        pos.pop(tool)
        goal[bounds[0] + np.random.choice(pos)] = 1
    return tools, goal, n_differences


def change_item_to_slot(item_to_slot: np.ndarray) -> np.ndarray:
    indexs = np.asarray(list(range(item_to_slot.shape[1])))
    np.random.shuffle(indexs)
    c = 0
    for i in range(item_to_slot.shape[0]):
        if sum(item_to_slot[i, :]) == 0:
            item_to_slot[i, indexs[c]] = 1
            c += 1
        else:
            item_to_slot[i, :] = 0
    return item_to_slot


def get_target_from_array(target_state: np.ndarray):
    target = np.where(target_state == 1)[0][0]
    return target


def generate_prices(interface):
    number_of_prices = 1
    prices = np.random.randint(1, 10, size=(number_of_prices, 6))
    prices[:, 2] = interface.elements["."]
    prices[:, -1] = interface.elements["enter"]
    return prices


def element_intersections(position, element):
    target_center = (element['x'] + element['w'] / 2, element['y'] + element['h'] / 2)
    p0 = Point(position[0], position[1], evaluate=False)
    p1 = Point(target_center[0], target_center[1], evaluate=False)
    position_target = Line(p0, p1)

    points = np.asarray(
        [[element['x'], element['x'] + element['w'], element['x'] + element['w'], element['x'], element['x']],
         [element['y'], element['y'], element['y'] + element['h'], element['y'] + element['h'], element['y']]])

    intersections = []
    for i in range(4):
        A = points[:, i]
        B = points[:, i + 1]
        p2 = Point(A[0], A[1], evaluate=False)
        p3 = Point(B[0], B[1], evaluate=False)
        segment = Segment(p2, p3)
        pt_int = intersection(segment, position_target)
        if len(pt_int) > 0:
            pt_int = pt_int[0]
            intersections.append([float(pt_int[0]), float(pt_int[1])])
    return np.asarray(intersections)


def compute_width_distance(position, element):
    target_center = np.asarray(
        [element['x'] + element['w'] / 2, element['y'] + element['h'] / 2])
    distance = calc_distance(
        position,
        target_center,
        'l1',
    )
    intersects = element_intersections(position, element)
    width = calc_distance(
        intersects[0, :],
        intersects[1, :],
        'l1',
    )
    return width, distance


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


def compute_width_distance_fast(position, element):
    target_center = np.asarray(
        [element['x'] + element['w'] / 2, element['y'] + element['h'] / 2])
    L1 = line(position, target_center)
    points = np.asarray(
        [[element['x'], element['x'] + element['w'], element['x'] + element['w'], element['x'], element['x']],
         [element['y'], element['y'], element['y'] + element['h'], element['y'] + element['h'], element['y']]])

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
