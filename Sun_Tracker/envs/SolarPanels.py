import numpy as np
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math
from math import floor, ceil
import torch
from typing import Optional
import pygame
from pygame import gfxdraw
import random  
import pygame
import sys
import math
import numpy as np

class SolarPanel:
    def __init__(self, initial_tilt_angle):
        self.tilt_angle = 0  
        #Solar panels icon for human rendering
        self.alpha_angle = 0 
        self.theta_angle = 0
        self.window_size = (800, 600)
        self.panel_position = (self.window_size[0] // 2, self.window_size[1] - 100)
        
    #get the tilts angles (alpha and theta) and the actual solar panels angles in step_time (15 minutes)
    def get_tilt_angle(self, action):
        self.alpha_tilt_angle = action[0]   
        self.theta_tilt_angle = action[1]
        return self.alpha_tilt_angle, self.theta_tilt_angle
    
    def get_actual_current_angle(self, action):
        self.alpha_angle += action[0]
        self.theta_angle += action[1]
        return self.alpha_angle, self.theta_angle

    #function for human rendering of the solar panel 
    #2 Dimensional coordinates system (with fixed radius)
    def get_Solar_Panel_coordinates(self, action, theta, phi):
        theta, phi = SolarPanel.get_actual_current_angle(self, action)[0], SolarPanel.get_actual_current_angle(self, action)[1]
        theta, phi = math.radians(theta), math.radians(phi)
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        return x, y, z