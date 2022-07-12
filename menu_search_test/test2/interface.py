import numpy as np
from gym import spaces


class Interface:
    def __init__(self, env_config):
        self.screen_width = 1920
        self.screen_height = 1080
        self.ui: dict = {}
        self.n_slots = env_config['n_items']
        self.n_items = env_config['n_items']
        self.mid = env_config['mid']
        self.low = env_config['low']
        self.diff = env_config['n_diff']
        self.relative = env_config['relative']
        self.decision_time = 0.4

    def generate_random_ui(self):
        cells_width = int(self.screen_width / 15)
        cells_height = int(self.screen_width / 15)
        size = 0.1
        margin = 0.01
        _element = {
            "x": None,
            "y": None,
            "w": size,
            "h": size,
            "item": None
        }
        pixels = np.ones((cells_width, cells_height), dtype=np.int8)
        pixel_margins = int(margin * cells_width)
        pixels[:pixel_margins, :] = 0
        pixels[-(pixel_margins + int(size * cells_width)):, :] = 0
        pixels[:, :pixel_margins] = 0
        pixels[:, -(pixel_margins + int(size * cells_height)):] = 0
        ui ={}
        for target, item in zip(range(self.n_slots), range(self.n_items)):
            element = _element.copy()
            element['item'] = item
            options = np.where(pixels == 1)[:2]
            idx = np.random.randint(len(options[0]))
            x = options[0][idx]
            y = options[1][idx]
            element['x'] = x
            element['y'] = y
            lower_x = max(x - int(element['w'] * cells_width) - pixel_margins, 0)
            higher_x = min(x + int(element['w'] * cells_width) + pixel_margins, cells_width)
            lower_y = max(y - int(element['h'] * cells_height) - pixel_margins, 0)
            higher_y = min(y + int(element['h'] * cells_height) + pixel_margins, cells_height)
            pixels[lower_x:higher_x, lower_y:higher_y] = 0
            element['x'] /= cells_width
            element['y'] /= cells_height

            ui[target] = element
        return ui

    def generate_static_ui(self):
        size = 0.15
        margin = 0.05
        _element = {
            "x": None,
            "y": None,
            "w": size,
            "h": size,
            "item": None
        }

        max_per_direction = np.floor((1 - margin) / (size + margin))
        assert self.n_slots <= max_per_direction ** 2
        col = 0
        row = 0
        for target, item in zip(range(self.n_slots), range(self.n_items)):
            element = _element.copy()
            element['item'] = item
            element['x'] = size * row + (row + 1) * margin
            element['y'] = size * col + (col + 1) * margin
            self.ui[target] = element

            row += 1
            if row % max_per_direction == 0:
                col += 1
                row = 0

    def get_menu_location(self):
        result = []
        for element in self.ui:
            result.append(self.ui[element]['x'] + self.ui[element]['w'] / 2)
            result.append(self.ui[element]['y'] + self.ui[element]['h'] / 2)
        return np.asarray(result, dtype=np.float64)

    def update_page(self):
        pass

    def render(self):
        pass

    def screenshot(self, suffix):
        pass

    def update_folder(self, name):
        pass

    def to_gif(self):
        pass


def main():

    env_config = {
        'low': 'coordinate',  # True, False, 'heuristic', 'coordinate'
        'mid': 'heuristic',  # None, 'multi', 'single', 'heuristic'
        'random':True,
        'relative': False,
        'n_items':15,
        'n_diff': 'curriculum'
    }
    interface = Interface(env_config)
    while True:
        print("-")
        interface.generate_random_ui()


if __name__ == '__main__':
    main()
