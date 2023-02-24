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
import HelpText as Ht

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

try:
    import torch
except ImportError:
    raise RuntimeError(
        'cannot import torch, make sure torch package is installed')

# Method imports
from attack.fast_attacks import fgsm, it_fgsm

# Class imports
from KeyboardControl import KeyboardControl
from HUD import HUD
from BBHUD import BBHUD
from World import World
from model.speed_limit_detector import SpeedLimitDetector

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

        hud = HUD(args.width, args.height)
        bbhud = BBHUD(args.width, args.height)
        world = World(sim_world, hud, bbhud, args)
        for sign in world.world.get_actors().filter('traffic.speed_limit.90'):
            sign.destroy()

        world.cuda_available = torch.cuda.is_available()
        if world.cuda_available:
            print("CUDA IS WORKING")
            world.device = torch.device("cuda")
        else:
            world.device = torch.device("cpu")
            print("CUDA ISNT WORKING")
        
        if (world.detector is None):
            world.detector = SpeedLimitDetector(world.device)
            
            # Initial cache of function
            temp_cache_img = world.detector.preprocess(cv2.imread("../out/sample.png"))
            temp_cache = world.detector.detect_sign(temp_cache_img)
            
            # Initial cache of function
            world.attack_methods.append((fgsm,"FGSM"))
            world.attack_methods.append((it_fgsm,"IT-FGSM"))
                        
            temp_cache = fgsm(world.detector.model, temp_cache_img, world.device, 4, True, batch= False)
            #temp_cache = it_fgsm(world.detector.model, temp_cache_img, world.device, 4, True, batch= False)

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

        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(30)
            if controller.parse_events(client, world, clock, args.sync):
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


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
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
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

    print(Ht.__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()