# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import carla

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_F2
    from pygame.locals import K_F3
    from pygame.locals import K_F4
    from pygame.locals import K_F5
    from pygame.locals import K_F6
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_e
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_j
    from pygame.locals import K_k
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_u
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_y
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# Class Imports
from World import World

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world: World, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world: World, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_F2:
                    world.bbhud.toggle_info()
                elif event.key == K_F3:
                    world.modelManager.toggle_modelWindow()
                elif event.key == K_F4:
                    world.modelManager.toggle_attackWindow()
                elif event.key == K_F5:
                    world.modelManager.toggle_defenseWindow()
                elif event.key ==K_F6:
                    if (world.modelManager.decouple_flag):
                        world.modelManager.set_decouple_flag(False)
                        world.hud.notification("Disabled decoupling of the models")
                    else:
                        world.modelManager.set_decouple_flag(True)
                        world.modelManager.empty_model_results()
                        world.hud.notification("Enabled decoupling of the models")
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.toggle_help()
                elif event.key == K_k:
                    world.hud.toggle_modelHelp()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_y:
                    if (world.modelManager.isOverrideSpeed):
                        world.modelManager.isOverrideSpeed = False
                        world.agentManager.set_agentStatus(False)
                        world.hud.notification("Disabled Auto-Speed limit")
                    else:
                        world.modelManager.isOverrideSpeed = True
                        if not world.agentManager.agentStatus:
                            world.agentManager.set_agentRandomDestination(world.map.get_spawn_points())
                            world.agentManager.set_agentStatus(True)
                        # Activates model if not activated
                        if (not world.modelManager.model_flag):
                            world.modelManager.set_model_flag(True, world.vehicle_camera, world.hud, world.agentManager)
                            world.hud.notification(
                                "Auto-Speed limit enabled and Speed-limit Sign detection enabled")
                        else:
                            world.hud.notification(
                                "Auto-Speed limit enabled")
                elif event.key == K_u:
                    if (world.modelManager.model_flag):
                        world.modelManager.set_model_flag(False, world.vehicle_camera, world.hud, world.agentManager)
                        world.modelManager.set_attack_model_flag(False)
                        world.modelManager.set_defense_model_flag(False)
                        world.modelManager.model_image = world.modelManager.getEmptyImage()
                        world.modelManager.attack_model_image = world.modelManager.getEmptyImage()
                        world.modelManager.defense_model_image = world.modelManager.getEmptyImage()
                        world.hud.notification(
                            "Speed-limit Sign detection disabled")
                    else:
                        world.modelManager.set_model_flag(True, world.vehicle_camera, world.hud, world.agentManager)
                        world.hud.notification(
                            "Speed-limit Sign detection enabled")
                    if (world.modelManager.decouple_flag):
                        world.modelManager.empty_model_results()
                elif event.key == K_e and pygame.key.get_mods() & KMOD_SHIFT:
                    if (world.modelManager.model_flag):
                        if (world.modelManager.defense_model_flag):
                            world.modelManager.set_defense_model_flag(False)
                            world.modelManager.defense_model_image = world.modelManager.getEmptyImage()
                            world.hud.notification("{} Defense Sign detection disabled".format(world.modelManager.defense_methods[world.modelManager.defense_currentMethodIndex]))
                        else:
                            world.modelManager.set_defense_model_flag(True)
                            world.hud.notification("{} Defense Sign detection enabled".format(world.modelManager.defense_methods[world.modelManager.defense_currentMethodIndex]))
                        if (world.modelManager.decouple_flag):
                            world.modelManager.empty_model_results()
                    else:
                        world.hud.notification(
                            "Can't enable Defense Sign detection while Sign detection is disabled")
                elif event.key == K_e:
                    if (world.modelManager.model_flag):
                        if (world.modelManager.attack_model_flag):
                            world.modelManager.set_attack_model_flag(False)
                            world.modelManager.attack_model_image = world.modelManager.getEmptyImage()
                            world.hud.notification("{} Attack Sign detection disabled".format(world.modelManager.attack_methods[world.modelManager.attack_currentMethodIndex]))
                        else:
                            world.modelManager.set_attack_model_flag(True)
                            world.hud.notification("{} Attack Sign detection enabled".format(world.modelManager.attack_methods[world.modelManager.attack_currentMethodIndex]))
                        if (world.modelManager.decouple_flag):
                            world.modelManager.empty_model_results()
                    else:
                        world.hud.notification(
                            "Can't enable Attack Sign detection while Sign detection is disabled")
                elif event.key == K_f and pygame.key.get_mods() & KMOD_SHIFT:
                    world.modelManager.defense_currentMethodIndex = (world.modelManager.defense_currentMethodIndex+1) % len(world.modelManager.defense_methods)
                    world.hud.notification("Changed Defense method to {}".format(world.modelManager.defense_methods[world.modelManager.defense_currentMethodIndex]))
                elif event.key == K_f:
                    world.modelManager.attack_currentMethodIndex = (world.modelManager.attack_currentMethodIndex+1) % len(world.modelManager.attack_methods)
                    world.hud.notification("Changed Attack method to {}".format(world.modelManager.attack_methods[world.modelManager.attack_currentMethodIndex]))
                elif event.key == K_o and pygame.key.get_mods() & KMOD_SHIFT:
                    if (world.modelManager.model_flag):
                        if(world.modelManager.modelRecord_flag):
                            world.modelManager.set_modelRecord_flag(False)
                            world.hud.notification(
                            "Disabled model image recording")
                        else:
                            world.modelManager.set_modelRecord_flag(True)
                            world.hud.notification(
                            "Enabled model image recording")
                    else:
                        world.hud.notification(
                            "Speed-limit Sign detection is not enabled")
                elif event.key == K_o:
                    if (world.modelManager.model_flag):
                        world.modelManager.modelPicture_flag = True
                    else:
                        world.hud.notification(
                            "Speed-limit Sign detection is not enabled")
                elif event.key == K_j:
                    if (world.modelManager.model_flag):
                        world.modelManager.modelClassificationPicture_flag = True
                    else:
                        world.hud.notification(
                            "Speed-limit Sign detection is not enabled")
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification(
                            "Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(
                            carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification(
                            "Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(
                        event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(
                        "Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec",
                                       world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        if self._autopilot_enabled:
                            world.hud.notification('Autopilot %s' % ('On'))
                        else:
                            world.hud.notification(
                                    'Autopilot %s' % ('Off'))

                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(
                    pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights:  # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(
                        carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(
                    pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods(
            ) & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)