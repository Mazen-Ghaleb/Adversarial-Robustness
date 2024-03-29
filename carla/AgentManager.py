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
            # self.agent = BasicAgent(player, opt_dict={"target_speed": 30, "ignore_traffic_lights": True, "ignore_stop_signs": True, "sampling_resolution": 1.0})
            self.agent = BasicAgent(player)
        else:
            self.agent = BehaviorAgent(player, behavior= self.agentBehavior)
            
        self.agent.set_target_speed(30)
        self.agent.ignore_traffic_lights(True)
        self.agent.ignore_stop_signs(True)
        self.agent._sampling_resolution = 1.0
            
    def toggle_agentStatus(self):
            self.agentStatus = not self.agentStatus

    def set_agentStatus(self, flag:bool):
            self.agentStatus = flag
            
    def set_agentRandomDestination(self, spawn_points):
        
        # Set the agent random destination
        destination = random.choice(spawn_points).location
        
        self.agent.set_destination(destination)