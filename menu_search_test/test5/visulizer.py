from interface import Interface
import pygame
from pygame import gfxdraw

env_config = {
    'random': True,
    'n_buttons': 6,
    'mode': 'all_exclu'
}

interface = Interface(env_config)

pygame.init()
pygame.display.init()
font = pygame.font.Font('freesansbold.ttf', 16)
screen = pygame.display.set_mode((interface.screen_width, interface.screen_height))
clock = pygame.time.Clock()
for i in range(10):
    interface.load('./best_ui/n_buttons_5-mode_all_exclu_top'+str(i))
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