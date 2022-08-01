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

class Button():
    """
    Buttons are rectangles and have several kind of different sizes. An id is associated with one button.
    Button size is always times of grid size.

    Attribute:
    position:
        postiton of the button in pixel
    size:
        size of the button in pixel
    id:
        the id of the button
    """

    size_options = [80, 160, 240] # small, medium, big (in pixels)

    def __init__(self, args) -> None:
        self.position = np.array(args['position']) # The top-left corner of the button
        self.size = np.array([self.size_options[args['height']], self.size_options[args['width']]])
        self.id = args['id']
    
    def relative_size(self, grid_size):
        return np.array([int(self.size[0] / grid_size), int(self.size[1] / grid_size)])

    def status(self):
        _status = {
            'position': self.position,
            'height': int(np.nonzero(self.size_options == self.size[0])[0][0]),
            'width': int(np.nonzero(self.size_options == self.size[1])[0][0]),
            'id': self.id
        }
        return _status
        
class Interface():
    build_in_mode = ['normal', 'central', 'mutual_exclu', 'all_exclu']

    def __init__(self, config):
        self.screen_width = 1920
        self.screen_height = 1080
        self.grid_size = 40
        self.interval = 1
        self.margin_size = 40
        self.n_buttons = config['n_buttons']
        self.random = config['random']
        self.button_id = np.array(range(self.n_buttons))
        self.ui = []
        self.mode = config['mode'] # normal, central, mutual_exclu, all_exclu
        self.button_group = {
            'normal': None,
            'central': None,
            'mutual_exclu': None,
            'all_exclu': None
        }
        if self.random:
            self.generate_random_ui()
        else:
            self.generate_static_ui()
        self.generate_control_button_group()


    def generate_static_ui(self):
        grid = np.ones(shape=(int((self.screen_height-2*self.margin_size)/self.grid_size+1), int((self.screen_width-2*self.margin_size)/self.grid_size)+1))
        for id in self.button_id:
            not_occupied = np.nonzero(grid)
            button_size = 2
            grid_button_position = np.array([not_occupied[0][0], not_occupied[1][0]])
            button_args = {
                'position': self.get_pixel(grid_button_position),
                'height': button_size,
                'width': button_size,
                'id': id
            }
            grid_button_size = Button(button_args).relative_size(self.grid_size)
            for i in range(len(not_occupied[0])):
                if self.is_available_grid_point(grid_button_position, grid_button_size, grid):
                    break
                else:
                    grid_button_position = np.array([not_occupied[0][i], not_occupied[1][i]])
            
            button_args['position'] = self.get_pixel(grid_button_position)
            button = Button(button_args)
            self.ui.append(button)
            self.update_grid(grid_button_position, grid_button_size, grid)

    def generate_random_ui(self):
        grid = np.ones(shape=(int((self.screen_height-2*self.margin_size)/self.grid_size+1), int((self.screen_width-2*self.margin_size)/self.grid_size+1)))
        for id in self.button_id:
            not_occupied = np.nonzero(grid)
            button_height = np.random.randint(low=0, high=len(Button.size_options))
            button_width = np.random.randint(low=0, high=len(Button.size_options))
            rand_postion = np.random.randint(low=0, high=np.size(not_occupied[0]))
            grid_button_position = np.array([not_occupied[0][rand_postion], not_occupied[1][rand_postion]])
            button_args = {
                'position': self.get_pixel(grid_button_position),
                'height': button_height,
                'width': button_width,
                'id': id
            }
            grid_button_size = Button(button_args).relative_size(self.grid_size)

            while not self.is_available_grid_point(grid_button_position, grid_button_size, grid):
                rand_postion = np.random.randint(low=0, high=np.size(not_occupied[0]))
                grid_button_position = np.array([not_occupied[0][rand_postion], not_occupied[1][rand_postion]])

            button_args['position'] = self.get_pixel(grid_button_position)
            button = Button(button_args)
            self.ui.append(button)
            self.update_grid(grid_button_position, grid_button_size, grid)

    def generate_control_button_group(self):
        if self.mode not in self.build_in_mode:
            self.mode = 'normal'
        elif self.mode == 'normal':
            self.button_group['normal'] = self.button_id
        elif self.mode == 'central':
            self.button_group['central'] = 0
            self.button_group['normal'] = []
            for id in self.button_id:
                if id != self.button_group['central']:
                    self.button_group['normal'].append(id)
        elif self.mode == 'mutual_exclu':
            self.button_group['mutual_exclu'] = np.array(range(int(self.n_buttons/2)))
            self.button_group['normal'] = []
            for id in self.button_id:
                if id not in self.button_group['mutual_exclu']:
                    self.button_group['normal'].append(id)
        elif self.mode == 'all_exclu':
            self.button_group['all_exclu'] = 0
            self.button_group['normal'] = []
            for id in self.button_id:
                if id != self.button_group['all_exclu']:
                    self.button_group['normal'].append(id)


    def is_available_grid_point(self, position, size, grid):
        if position[0]+size[0] < grid.shape[0] and position[1]+size[1] < grid.shape[1]:
            to_occupy = grid[position[0]:position[0]+size[0]+1, position[1]:position[1]+size[1]+1]
            if np.array_equal(to_occupy, np.ones(np.shape(to_occupy))):
                return True
            else:
                return False
        else:
            return False

    def update_grid(self, position, size, grid):
        top = np.max([0, position[0] - self.interval])
        left = np.max([0, position[1] - self.interval])
        bottom = np.min([position[0]+size[0]+1+self.interval, grid.shape[0]-1])
        right = np.min([position[1]+size[1]+1+self.interval, grid.shape[1]-1])
        grid[top:bottom, left:right] = 0


    def get_pixel(self, grid_point):
        pixel = np.array([self.margin_size + grid_point[0]*self.grid_size, self.margin_size + grid_point[1]*self.grid_size])
        return pixel

    def check_within_button(self, position):
        for button in self.ui:
            lefttop = self.button_normalized_position(button)
            size = self.button_normalized_size(button)
            if lefttop[0]  < position[0] < lefttop[0] + size[0] and lefttop[1] < position[1] < lefttop[1] + size[1]:
                return button.id
        return None

    def press(self, button_id: int, state: np.array):
        if button_id is not None:
            if self.mode == 'normal':
                state[button_id] = 1 - state[button_id]
            if self.mode == 'central':
                state[button_id] = 1 - state[button_id]
                if self.button_group['central'] == button_id:
                    state[self.button_group['normal']] = state[button_id] 
                else: 
                    if np.array_equal(state[self.button_group['normal']], np.ones(self.n_buttons-1)):
                        state[self.button_group['central']] = 1
                    # elif np.array_equal(state[self.button_group['normal']], np.zeros(self.n_buttons-1)):
                    #     state[self.button_group['central']] = 0
                    else:
                        state[self.button_group['central']] = 0
            if self.mode == 'mutual_exclu':
                state[button_id] = 1 - state[button_id]
                if button_id in self.button_group['mutual_exclu']:
                    if state[button_id] == 1:
                        for id in self.button_group['mutual_exclu']: 
                            if id != button_id:
                                state[id] = 0
            if self.mode == 'all_exclu':
                state[button_id] = 1 - state[button_id]
                if self.button_group['all_exclu'] == button_id:
                    state[self.button_group['normal']] = 1 - state[button_id]
                else:
                    if np.array_equal(state[self.button_group['normal']], np.zeros(self.n_buttons-1)):
                        state[self.button_group['all_exclu']] = 1
                    else:
                        state[self.button_group['all_exclu']] = 0

    def sample_possible_pattern(self):
        sample = np.random.randint(low=0, high=2, size=self.n_buttons)
        while not self.check_legal_pattern(sample):
            sample = np.random.randint(low=0, high=2, size=self.n_buttons)
        return sample

    def check_legal_pattern(self, pattern: np.array):
        if self.mode == 'central':
            if pattern[self.button_group['central']] == 1:
                if not np.array_equal(pattern[self.button_group['normal']], np.ones(self.n_buttons -1)):
                    return False
            elif pattern[self.button_group['central']] == 0:
                if np.array_equal(pattern[self.button_group['normal']], np.ones(self.n_buttons -1)):
                    return False
        elif self.mode == 'mutual_exclu':
            if np.count_nonzero(pattern[self.button_group['mutual_exclu']]) > 1:
                return False
        elif self.mode == 'all_exclu':
            if pattern[self.button_group['all_exclu']] == 1:
                if not np.array_equal(pattern[self.button_group['normal']], np.zeros(self.n_buttons -1)):
                    return False
            elif pattern[self.button_group['all_exclu']] == 0:
                if np.array_equal(pattern[self.button_group['normal']], np.zeros(self.n_buttons -1)):
                    return False
        return True
                
    def button_normalized_position(self, button: Button):
        return button.position  / np.array([self.screen_height, self.screen_width])

    def button_normalized_size(self, button: Button):
        return button.size  / np.array([self.screen_height, self.screen_width])

    def status(self):
        _status = {
            'screen_height': self.screen_height,
            'screen_width': self.screen_width,
            'grid_size': self.grid_size,
            'interval': self.interval,
            'margin_size': self.margin_size,
            'n_buttons': self.n_buttons,
            'button_id': self.button_id,
            'buttons': []
        }
        for button in self.ui:
            _status['buttons'].append(button.status())
        return _status

    def save(self, name: str):
        with open(name, 'w') as file_to_write:
            json.dump(self.status(), file_to_write, indent=4, cls=NpEncoder)
         
    
    def load(self, path: str):
        with open(path, 'r') as file_to_read:
            parameters = json.load(file_to_read)
            self.screen_height = parameters['screen_height']
            self.screen_width = parameters['screen_width']
            self.grid_size = parameters['grid_size']
            self.interval = parameters['interval']
            self.interval = parameters['margin_size']
            self.n_buttons = parameters['n_buttons']
            self.button_id = np.array(parameters['button_id'])
            self.ui.clear()
            for button_parameters in parameters['buttons']:
                self.ui.append(Button(button_parameters))
        

    

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
    import pygame
    from pygame import gfxdraw

    env_config = {
        'random': True,
        'n_buttons': 10,
    }
    interface = Interface(env_config)
    # print(interface.status())
    # interface.save('./test_interface.json')

    # interface.load('./test_interface.json')
    # print(interface.status())



    pygame.init()
    pygame.display.init()
    font = pygame.font.Font('freesansbold.ttf', 16)
    screen = pygame.display.set_mode((interface.screen_width, interface.screen_height))
    clock = pygame.time.Clock()
    for _ in range(20):
        # interface = Interface(env_config)
        surf = pygame.Surface((interface.screen_width, interface.screen_height))
        surf.fill((255, 255, 255))
            
        
        for button in interface.ui:
            rect = pygame.Rect(button.position[1],
                                button.position[0],
                                button.size[1],
                                button.size[0])
            gfxdraw.rectangle(surf, rect, (200, 0, 0))
        screen.blit(surf, (0, 0))

        for button in interface.ui:
            t = f"Button {button.id}"
            text = font.render(t, True, (0, 0, 0))
            screen.blit(text, (button.position[1], button.position[0]))

        pygame.event.pump()
        clock.tick(2)
        pygame.display.flip()
    pygame.display.quit()
    pygame.quit()

