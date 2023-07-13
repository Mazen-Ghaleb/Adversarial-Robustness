# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import os

try:
    import cv2
except ImportError:
    raise RuntimeError(
        'cannot import cv2, make sure cv2 package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- Bounding Box HUD -----------------------------------------------------------------------
# ==============================================================================

class BBHUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._show_info = True
        self._info_text = []

    def tick(self, world):
        if not self._show_info:
            return

        self._info_text = [
            "{:<29}".format('Classification Bounding Box:'),
            world.modelManager.model_image,
            '',
            "{:<29}".format('Attack Bounding Box:'),
            world.modelManager.attack_model_image,
            '',
            "{:<29}".format('Defense Bounding Box:'),
            world.modelManager.defense_model_image
        ]

    def toggle_info(self):
        self._show_info = not self._show_info

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (self.dim[0] - info_surface.get_width(), 0))
            v_offset = 4
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                
                if isinstance(item, np.ndarray):
                    dim = (190, 190)
                    resized = cv2.resize(item, dim)
                    resized = np.swapaxes(resized, 0, 1)
                    
                    rect = pygame.Rect((0, 0), dim)
                    display.blit(pygame.surfarray.make_surface(resized),
                                 (display.get_width() - surface.get_width() - 8, v_offset), rect)
                    v_offset += dim[1]
                    item = None
                    
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (display.get_width() - surface.get_width() - 8, v_offset))
                v_offset += 18
