# Copyright (c) 2023 Mazen Mostafa Ghaleb, Mostafa Lotfy Mostafa, Safia Medhat Abdulaziz, Youssef Maher Nader
#
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import sys
import carla
import matplotlib.image
import random
from timeit import default_timer as timer   
from GlobalMethods import get_actor_display_name, get_actor_blueprints, find_weather_presets

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

# Class Imports
from HUD import HUD
from BBHUD import BBHUD
from CollisionSensor import CollisionSensor
from LaneInvasionSensor import LaneInvasionSensor
from GnssSensor import GnssSensor
from IMUSensor import IMUSensor
from RadarSensor import RadarSensor
from CameraManager import CameraManager
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent 

initial_spawn_point = carla.Transform(carla.Location(x=396.4, y=67.2, z=2), carla.Rotation(yaw=270)) 

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud: HUD, bbhud: BBHUD, args):
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
        self.hud = hud
        self.bbhud = bbhud
        self.player = None
        self.agent = None
        self.agentMode = args.agent
        self.agentBehavior = args.behavior
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

        self.model_currentTick = 0
        self.model_tickRate = 20
        self.window_first_stats = [True, True, True]

        self.model_flag = False
        self.attack_model_flag = False
        self.defense_model_flag = False
        self.modelPicture_flag = False
        self.modelClassificationPicture_flag = False

        self.detector = None
        self.isOverrideSpeed = False

        self.model_result = None
        self.model_speed = None
        self.model_confidence = None
        self.model_image = np.zeros((640, 640, 3), dtype = np.uint8)
        self.model_image_window = False

        self.attack_methods = []
        self.attack_currentMethodIndex = 0
        
        self.defense_methods = []
        self.defense_currentMethodIndex = 0

        self.attack_model_result = None
        self.attack_model_speed = None
        self.attack_model_confidence = None
        self.attack_model_image = np.zeros((640, 640, 3), dtype = np.uint8)
        self.attack_model_image_window = False
        
        self.defense_model_result = None
        self.defense_model_speed = None
        self.defense_model_confidence = None
        self.defense_model_image = np.zeros((640, 640, 3), dtype = np.uint8)
        self.defense_model_image_window = False

    def toggle_modelWindow(self):
        self.model_image_window = not self.model_image_window
        self.window_first_stats[0] = True
    
    def toggle_attackWindow(self):
        self.attack_model_image_window = not self.attack_model_image_window
        self.window_first_stats[1] = True
        
    def toggle_defenseWindow(self):
        self.defense_model_image_window = not self.defense_model_image_window
        self.window_first_stats[2] = True

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
            
        self.create_agent()
        
    def create_agent(self):
        if self.agentMode != "None":
            if self.agentMode == "Basic":
                self.agent = BasicAgent(self.player)
            else:
                self.agent = BehaviorAgent(self.player, behavior= self.agentBehavior)
            self.agent.ignore_traffic_lights(True)
            self.agent.set_target_speed(15)

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

    def toggle_model(self, flag):  # True to activate model and False to stop model
        if (flag):
            self.vehicle_camera.listen(
                lambda image: calculate_model(image, self))
        else:
            self.vehicle_camera.stop()
            self.model_result = None
        self.model_flag = flag

    def toggle_attack_model(self, flag:bool):  # True to activate attack model and False to stop attack model
        self.attack_model_flag = flag
    
    def toggle_defense_model(self, flag:bool):  # True to activate defense model and False to stop defense model
        self.defense_model_flag = flag

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
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- Model functions ----------------------------------------------------------
# ==============================================================================
            
def to_bgr_array(image):
    """Convert a CARLA raw image to a BGR numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array

def drawBoundingBox(boxes, image: np.ndarray, classifications, confidences, color =(0, 255, 0), font = cv2.FONT_HERSHEY_SIMPLEX, thickness = 2):
    for box,classification,confidence in zip(boxes,classifications,confidences):
        detected_image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
        detected_image = cv2.putText(detected_image, "{}: {:.2f} %".format(int(classification), confidence*100), 
        (int(box[0]), int(box[1] - 5)), font, 0.6, color, 1)
    return detected_image

def calculate_model(image, world: World):
    total_start = timer()
    if (world.model_currentTick == 0):  # Performs the calculation every world.model_tickRate ticks
        image = to_bgr_array(image)
        world.detector.preprocess(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if world.modelPicture_flag:
            matplotlib.image.imsave('../out/test.png', image)
            world.hud.notification("Saved current view of Speed-limit Sign detection")
            world.modelPicture_flag = False

        calculate_classification(world, np.array(image, copy=True))
        
        if world.defense_model_flag and world.attack_model_flag:
            calculate_attack(world, np.array(image, copy=True))
            calculate_defense(world, np.array(image, copy=True))
            calculate_overrideSpeed(world, world.defense_model_speed)
        elif world.defense_model_flag:
            calculate_defense(world, np.array(image, copy=True))
            calculate_overrideSpeed(world, world.defense_model_speed)
        elif world.attack_model_flag:
            calculate_attack(world, np.array(image, copy=True))
            calculate_overrideSpeed(world, world.attack_model_speed)
        else:
            calculate_overrideSpeed(world, world.model_speed)

        print("{:<25}".format("Total Model time"),": {:.3f}s".format(timer()-total_start))
        
    world.model_currentTick = (world.model_currentTick+1) %world.model_tickRate

def calculate_classification(world: World, image):
    detection_start = timer()
    world.model_result = world.detector.run_without_attack()

    if world.model_result is not None:
        world.model_speed = world.model_result[0][0]
        world.model_confidence = world.model_result[1][0]
        world.model_image = drawBoundingBox(world.model_result[2], image, world.model_result[0], world.model_result[1])

        if world.modelClassificationPicture_flag:
            matplotlib.image.imsave('../out/bounding_test.png', world.model_image)
            world.hud.notification("Saved classified view of Speed-limit Sign detection")
            world.modelClassificationPicture_flag = False

        print("{:<25}".format("Classification Model time"),": {:.3f}s".format(timer()-detection_start),
        "Label:{:<3} Confidence:{:.3f}".format(int(world.model_speed), world.model_confidence))
    else:
        world.model_speed = None
        world.model_confidence = None
        world.model_image = np.zeros((640, 640, 3), dtype = np.uint8)
        print("{:<25}".format("Classification Model time"),": {:.3f}s No Sign Detected".format(timer()-detection_start))

def calculate_attack(world: World, image):
    attack_start = timer()
    world.attack_model_result = world.detector.run_with_attack(world.attack_methods[world.attack_currentMethodIndex])
    
    if world.attack_model_result is not None:
        world.attack_model_speed = world.attack_model_result[0][0]
        world.attack_model_confidence = world.attack_model_result[1][0]
        world.attack_model_image = drawBoundingBox(world.attack_model_result[2], image, world.attack_model_result[0], world.attack_model_result[1])
        
        print("{:<25}".format("{} Attack Model time".format(world.attack_methods[world.attack_currentMethodIndex])),
        ": {:.3f}s".format(timer()-attack_start),
        "Label:{:<3} Confidence:{:.3f}".format(int(world.attack_model_speed), world.attack_model_confidence))
    else:
        world.attack_model_speed = None
        world.attack_model_confidence = None
        world.attack_model_image = np.zeros((640, 640, 3), dtype = np.uint8)
        print("{:<25}".format("{} Attack Model time".format(world.attack_methods[world.attack_currentMethodIndex])),
        ": {:.3f}s No Sign Detected".format(timer()-attack_start))

def calculate_defense(world: World, image):
    defense_start = timer()
    world.defense_model_result = world.detector.run_with_defense(world.defense_methods[world.defense_currentMethodIndex],
                                                                 world.attack_methods[world.attack_currentMethodIndex])
    
    if world.defense_model_result is not None:
        world.defense_model_speed = world.defense_model_result[0][0]
        world.defense_model_confidence = world.defense_model_result[1][0]
        world.defense_model_image = drawBoundingBox(world.defense_model_result[2], image, world.defense_model_result[0], world.defense_model_result[1])
        
        print("{:<25}".format("{} Defense Model time".format(world.defense_methods[world.defense_currentMethodIndex])),
        ": {:.3f}s".format(timer()-defense_start),
        "Label:{:<3} Confidence:{:.3f}".format(int(world.defense_model_speed), world.defense_model_confidence))
    else:
        world.defense_model_speed = None
        world.defense_model_confidence = None
        world.defense_model_image = np.zeros((640, 640, 3), dtype = np.uint8)
        print("{:<25}".format("{} Defense Model time".format(world.defense_methods[world.defense_currentMethodIndex])),
        ": {:.3f}s No Sign Detected".format(timer()-defense_start))

def calculate_overrideSpeed(world: World, detectedSpeed):
    if world.isOverrideSpeed:
        if detectedSpeed:
            # Over 3.6 to convert it from km/h to m/s because constant velocity takes it in m/s
            print("{:<25}".format("Overriding the speed with"),": {:.3f} km/h".format(detectedSpeed))
            SpeedOfOverride = detectedSpeed /3.6
            world.player.enable_constant_velocity(carla.Vector3D(SpeedOfOverride, 0, 0))
            
