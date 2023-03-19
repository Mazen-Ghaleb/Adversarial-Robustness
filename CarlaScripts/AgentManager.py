# Copyright (c) 2023 Mazen Mostafa Ghaleb, Mostafa Lotfy Mostafa, Safia Medhat Abdulaziz, Youssef Maher Nader
#
# This work is licensed under the terms of the MIT license.
# For a copy, see https://opensource.org/licenses/MIT.

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================
import random

# Class Imports
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
 
# ==============================================================================
# -- AgentManager --------------------------------------------------------------
# ==============================================================================

class AgentManager(object):
    """Class representing the agent responsible for motion planning."""

    def __init__(self, agentMode, agentBehavior):
        self.agent = None
        self.agentMode = agentMode
        self.agentBehavior = agentBehavior
        self.agentStatus = False
    
    def create_agent(self, player):
        if self.agentMode != "None":
                self.agentStatus = True
            
        if self.agentMode == "Basic":
            self.agent = BasicAgent(player)
        else:
            self.agent = BehaviorAgent(player, behavior= self.agentBehavior)
            
        self.agent.ignore_traffic_lights(True)
        self.agent.set_target_speed(15)
        
    def toggle_agentStatus(self):
            self.agentStatus = not self.agentStatus

    def set_agentStatus(self, flag:bool):
            self.agentStatus = flag
            
    def set_agentRandomDestination(self, spawn_points):
        
        # Set the agent random destination
        destination = random.choice(spawn_points).location
        
        self.agent.set_destination(destination)