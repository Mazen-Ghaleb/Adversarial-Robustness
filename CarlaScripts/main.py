#!/usr/bin/env python

# Copyright (c) 2023 Mazen Mostafa Ghaleb, Mostafa Lotfy Mostafa, Safia Medhat Abdulaziz, Youssef Maher Nader
#
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.

# ==============================================================================
# -- Find CARLA Module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys
from carlaPath import carlaPath

try:
    sys.path.append(glob.glob(carlaPath+'/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import carla
import argparse
import logging

try:
    import cv2
except ImportError:
    raise RuntimeError(
        'cannot import cv2, make sure cv2 package is installed')

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

# Variable imports
from HelpText import helpText
from HelpText import modelHelpText
from World import initial_spawn_point

# Class imports
from KeyboardControl import KeyboardControl
from World import World
from demo import Demo

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)

        #sim_world = client.get_world()
        sim_world = client.load_world('Town01_Opt')

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        if (args.fullscreen):
            display = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
            args.width = 1920
            args.height = 1080
        else:
            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        world = World(sim_world, args=args)

        for sign in world.world.get_actors().filter('traffic.speed_limit.90'):
            sign.destroy()
        
        if (world.modelManager.detector is None):
            world.modelManager.detector = Demo()
            
            world.modelManager.attack_methods.append(("FGSM"))
            world.modelManager.attack_methods.append(("IT-FGSM"))
            world.modelManager.defense_methods.append(("HGD"))
            
            # Initial cache of function
            world.modelManager.detector.preprocess(cv2.imread("../out/sample.png"))
            temp_cache = world.modelManager.detector.run_without_attack()
            
            # Initial cache of function
            temp_cache = world.modelManager.detector.run_with_attack(world.modelManager.attack_methods[0])
            #temp_cache = it_fgsm(world.modelManager.detector.model, temp_cache_img, world.device, 4, True, batch= False)

        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()

        camera_bp = world.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(x=0, z=4))
        world.vehicle_camera = world.world.spawn_actor(
            camera_bp, camera_init_trans, attach_to=world.player)


        if (world.agentManager.agent is not None):    
            destination =carla.Location(x=291.9, y=-2.1, z=2) 
            world.agentManager.agent.set_destination(destination)
            # world.agentManager.set_agentRandomDestination(world.map.get_spawn_points())
        
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(30)
            if controller.parse_events(client, world, clock, args.sync):
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            
            if(world.modelManager.model_image_window):
                if cv2.getWindowProperty('Model Image', cv2.WND_PROP_VISIBLE) == 0 and not world.modelManager.window_first_stats[0]:
                    world.modelManager.toggle_modelWindow()
                else:
                    # Display the Model Image in window
                    cv2.imshow('Model Image', cv2.cvtColor(world.modelManager.model_image, cv2.COLOR_BGR2RGB))
                    world.modelManager.window_first_stats[0] = False
                
            if(world.modelManager.attack_model_image_window):
                if cv2.getWindowProperty('Attack Image', cv2.WND_PROP_VISIBLE) == 0 and not world.modelManager.window_first_stats[1]:
                    world.modelManager.toggle_attackWindow()
                else:
                    # Display the Attack Image in window
                    cv2.imshow('Attack Image', cv2.cvtColor(world.modelManager.attack_model_image, cv2.COLOR_BGR2RGB))
                    world.modelManager.window_first_stats[1] = False
            
            if(world.modelManager.defense_model_image_window):
                if cv2.getWindowProperty('Defense Image', cv2.WND_PROP_VISIBLE) == 0 and not world.modelManager.window_first_stats[2]:
                    world.modelManager.toggle_defenseWindow()
                else:
                    # Display the Defense Image in window
                    cv2.imshow('Defense Image', cv2.cvtColor(world.modelManager.defense_model_image, cv2.COLOR_BGR2RGB))
                    world.modelManager.window_first_stats[2] = False
            
            if world.agentManager.agentStatus:
                if world.agentManager.agent.done():
                    if args.loop:
                        world.restart(initial_spawn_point)
                        world.agentManager.agent.set_destination(destination)
                        world.hud.notification("The target has been reached, resetting vehicle position", seconds=4.0)
                        print("The target has been reached, resetting vehicle position")
                    elif world.modelManager.isOverrideSpeed:
                        world.agentManager.set_agentRandomDestination(world.map.get_spawn_points())
                        world.hud.notification("The target has been reached, resetting vehicle position to random point", seconds=4.0)
                        print("The target has been reached, resetting vehicle position")
                    else:
                        # world.agentManager.set_agentRandomDestination(world.map.get_spawn_points())
                        world.hud.notification("The target has been reached, stopping the simulation", seconds=4.0)
                        print("The target has been reached, stopping the simulation")
                        break

                control = world.agentManager.agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()
        cv2.destroyAllWindows()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.lincoln.mkz_2020',
        help='actor filter (default: "vehicle.lincoln.mkz_2020")')
        # default='vehicle.*',
        # help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "--agent", type=str,
        choices=["Behavior", "Basic", "None"],
        help="select which agent to run",
        default="None")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--fullscreen',
        action='store_true',
        help='Activate fullscreen mode')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(helpText, modelHelpText)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
