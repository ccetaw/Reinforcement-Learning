import numpy as np
from gym import spaces


class Interface:
    def __init__(self, env_config):
        self.screen_width = 1920
        self.screen_height = 1080
        self.ui: dict = {}
        self.n_slots = env_config['n_items']
        self.n_items = env_config['n_items']
        self.random = env_config['random']
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

            self.ui[target] = element

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



if __name__ == '__main__':
    env_config = {
        'random':True,
        'n_items':18,
    }
    interface = Interface(env_config)
    interface.generate_random_ui()

    import pygame
    from pygame import gfxdraw
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((interface.screen_width, interface.screen_height))
    clock = pygame.time.Clock()
    for _ in range(90):
        surf = pygame.Surface((interface.screen_width, interface.screen_height))
        surf.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 16)

        for element in interface.ui:
                rect = pygame.Rect(interface.ui[element]['x'] * interface.screen_width,
                                interface.ui[element]['y'] * interface.screen_height,
                                interface.ui[element]['w'] * interface.screen_width,
                                interface.ui[element]['h'] * interface.screen_height)
                gfxdraw.rectangle(surf, rect, (100, 100, 100))

        screen.blit(surf, (0, 0))
        pygame.event.pump()
        clock.tick(30)
        pygame.display.flip()
    pygame.display.quit()
    pygame.quit()