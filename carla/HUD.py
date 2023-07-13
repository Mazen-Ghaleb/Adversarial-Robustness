# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import os
import carla
import datetime
import math
from GlobalMethods import get_actor_display_name

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')
    
# Variable imports
from HelpText import helpText
from HelpText import modelHelpText

# Class Imports
from HelpText import HelpText
from FadingText import FadingText

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height, helpText)
        self.modelHelp = HelpText(pygame.font.Font(mono, 16), width, height, modelHelpText)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return

        t = world.player.get_transform()
        v = world.player.get_velocity()
        v_magnitude = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        speed_limits = world.world.get_actors().filter('traffic.speed_limit.*')
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (v_magnitude),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of Speed Signs: %d' % len(speed_limits)]

        if len(speed_limits) > 1:
            self._info_text += ['Nearby Speed Signs:']

            def distance(l): return math.sqrt((l.x - t.location.x) **
                                              2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            speed_limits = [(distance(x.get_location()), x)
                            for x in speed_limits if x.id != world.player.id]
            no_printedSpeedlimit = 0
            for d, speed_limit in sorted(speed_limits, key=lambda speed_limits: speed_limits[0]):
                if d > 200.0 or no_printedSpeedlimit > 1:
                    break
                speed_limits_type = get_actor_display_name(
                    speed_limit, truncate=22)+" Sign"
                self._info_text.append('% 4dm %s' % (d, speed_limits_type))
                no_printedSpeedlimit += 1

        self._info_text += ['Number of vehicles: %d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt((l.x - t.location.x) **
                                              2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            no_printedVehicle = 0
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0 or no_printedVehicle > 1:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
                no_printedVehicle += 1

        if (world.modelManager.model_flag):
            if (world.modelManager.model_speed is not None):
                self._info_text += ['Model Speed: %d'% world.modelManager.model_speed]

            if (world.modelManager.model_confidence is not None):
                self._info_text += ['Model Confidence: %.3f'%
                                    world.modelManager.model_confidence]

        if (world.modelManager.attack_model_flag):
            if (world.modelManager.attack_model_speed is not None):
                self._info_text += ['Attacked Speed: %d'% world.modelManager.attack_model_speed]

            if (world.modelManager.attack_model_confidence is not None):
                self._info_text += ['Attacked Confidence: %.3f'%
                                    world.modelManager.attack_model_confidence]
        
        if (world.modelManager.defense_model_flag):
            if (world.modelManager.defense_model_speed is not None):
                self._info_text += ['Defense Speed: %d'% world.modelManager.defense_model_speed]

            if (world.modelManager.defense_model_confidence is not None):
                self._info_text += ['Defense Confidence: %.3f'%
                                    world.modelManager.defense_model_confidence]

    def toggle_info(self):
        self._show_info = not self._show_info
        
    def toggle_help(self):
        if (self.modelHelp._render):
            self.modelHelp.toggle()
        self.help.toggle()
        
    def toggle_modelHelp(self):
        if (self.help._render):
            self.help.toggle()
        self.modelHelp.toggle()

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)
        self.modelHelp.render(display)