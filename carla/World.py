# Copyright (c) 2023 Mazen Mostafa Ghaleb, Mostafa Lotfy Mostafa, Safia Medhat Abdulaziz, Youssef Maher Nader
#
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import sys
import carla
import random
from GlobalMethods import get_actor_display_name, get_actor_blueprints, find_weather_presets

# Class Imports
from ModelManager import ModelManager
from AgentManager import AgentManager
from HUD import HUD
from BBHUD import BBHUD
from CollisionSensor import CollisionSensor
from LaneInvasionSensor import LaneInvasionSensor
from GnssSensor import GnssSensor
from IMUSensor import IMUSensor
from RadarSensor import RadarSensor
from CameraManager import CameraManager

# initial_spawn_point = carla.Transform(carla.Location(x=396.4, y=67.2, z=2), carla.Rotation(yaw=270)) # Town01
initial_spawn_point = carla.Transform(carla.Location(x=-13.6, y=5.8, z=11), carla.Rotation(yaw=180)) # Town04

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, args, hud: HUD = None, bbhud: BBHUD = None, agentManager:AgentManager =None, 
                 modelManager:ModelManager = ModelManager()):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        
        if hud is None:
            hud = HUD(args.width, args.height)    
        
        if bbhud is None:
            bbhud = BBHUD(args.width, args.height)    
        
        if agentManager is None:
            agentManager = AgentManager(args.agent, args.behavior)    
            
        self.hud = hud
        self.bbhud = bbhud
        self.player = None
        self.agentManager = agentManager
        self.modelManager = modelManager
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart(initial_spawn_point)
        # self.restart(carla.Transform(random.choice(self.map.get_spawn_points()).location, carla.Rotation(yaw=90))) # Random spawn point
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.vehicle_camera = None
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self, spawn_point = None):
        """Restart the world"""
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(get_actor_blueprints(
            self.world, self._actor_filter, self._actor_generation))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(
                blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(
                blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            if spawn_point is None:
                spawn_point = self.player.get_transform()
                spawn_point.location.z += 2.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            if spawn_point is None:
                spawn_points = self.map.get_spawn_points()
                spawn_point = random.choice(
                    spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(x=0, z=4))
        self.vehicle_camera = self.world.spawn_actor(
            camera_bp, camera_init_trans, attach_to=self.player)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
            
        self.agentManager.create_agent(self.player)
        
        if self.modelManager.model_flag:
            self.vehicle_camera.stop()
            self.modelManager.set_model_flag(True, self.vehicle_camera, self.hud, self.agentManager)
        
    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)
        self.bbhud.tick(self)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)
        self.bbhud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.vehicle_camera,
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                if sensor.is_listening:
                    sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()