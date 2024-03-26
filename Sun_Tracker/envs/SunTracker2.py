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
from Sun_Tracker.envs.Sun import Sun
from Sun_Tracker.envs.SolarPanels import SolarPanel
from gym.utils import seeding
from gym.spaces import Box

class SolarTrackingEnv(gym.Env):
    
    def __init__(self):
        super(SolarTrackingEnv, self).__init__()

        self.N = 1
        sun_params, initial_tilt_angle = (self.N, 1), 0
        self.sun = Sun(*sun_params)
        self.solar_panel = SolarPanel(initial_tilt_angle)
        self.energies = []
        self.energies.append(5000)
        self.tryouts_same_time_step = 0 
                
        # [solar_radiation, alpha, theta]
        self.observation_space = Box(
                    low=np.array([0, 0, 0]), 
                    high=np.array([15000, 180, 180]),  
                    shape = (3,),
                    dtype=np.float32)

        #action is relative to the current position of the solar panel [alpha,theta]
        self.action_space = Box(
                    low=np.array([5, -5]), 
                    high=np.array([15, 15]), 
                    shape=(2,), 
                    dtype=np.float32)     
           
        #At sunrise, if the wind does not exceed the speed max_wind_speed, the solar panels start off by tilting towards the position that is perpendicular to sun
        self.current_step = 1
        self.PV_Az = self.sun.get_energy_perpendicular_to_sun(self.current_step)[3]
        self.PV_Alt = self.sun.get_energy_perpendicular_to_sun(self.current_step)[4]
        self.max_energy = 10000
        #self.time_step_in_same_15_min = 0
        self.actions_history = []
        self.actions_history.append([0,0])
        #Put a limit on the max variation between 2 consecutive time steps
        self.max_alpha_tilt_angle, self.max_theta_tilt_angle = 16, 16
        #calculate steps in one day
        self.steps_in_a_day = len(self.sun.generate_radiation(self.PV_Az, self.PV_Alt)) 
        self.info = {'solar radiation': self.sun.get_radiation(self.PV_Az, self.PV_Alt, self.current_step), 'Maximum radiation': self.max_energy, 'solar panel tilt angle': self.solar_panel.get_tilt_angle([0,0]), 'time step': self.current_step}

    def back_to_initial_state(self):
        self.current_step = 1
        #reset the environment to intial state
        self.PV_Az = self.sun.get_energy_perpendicular_to_sun(self.current_step)[3]
        self.PV_Alt = self.sun.get_energy_perpendicular_to_sun(self.current_step)[4]
        self.solar_panel = SolarPanel(0)
        self.tryouts_same_time_step = 0 
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N += 1
        self.sun = Sun(self.N, 1)
        print(self.N)
        self.back_to_initial_state()
        obs = self._get_observation([0,0])
        return obs, self.info
    
    def step(self, action):  #action = [alpha_angle, theta_angle] = [PV_az, PV_alt] = [east/west , south/north]
        self.actions_history.append(action)
        self.info = {'solar radiation': self.sun.get_radiation(self.PV_Az, self.PV_Alt, self.current_step), 'Maximum radiation': self.max_energy, 'solar panel angles': (self.sun.get_energy_perpendicular_to_sun(self.current_step)[1], self.sun.get_energy_perpendicular_to_sun(self.current_step)[2]), 'time step': self.current_step}

        if abs(action[0]) > self.max_alpha_tilt_angle or abs(action[1]) > self.max_theta_tilt_angle :
            return self._get_observation(action), 0, False, True, self.info
        
        self.max_energy = self.sun.get_energy_perpendicular_to_sun(self.current_step)[0]    
        #print(self.max_energy, "max energy")
        self._get_observation(action)
        reward = self.get_reward(action)
        #Check if the episode is terminated (end of the day)
   
        if reward == 2 or self.tryouts_same_time_step == 30: 
            truncated = True 
            print(self.current_step, "current step")
            print(self.tryouts_same_time_step, "tryouts same time step")
            self.tryouts_same_time_step = 0 
            #make agent go to the perpendicular position to the sun at time step i 
            """self.PV_Az = self.sun.get_energy_perpendicular_to_sun(self.current_step)[3]
            self.PV_Alt = self.sun.get_energy_perpendicular_to_sun(self.current_step)[4]"""
            self.info = {'solar radiation': self.sun.get_radiation(self.PV_Az, self.PV_Alt, self.current_step), 'Maximum radiation': self.max_energy, 'solar panel angles': (self.PV_Az, self.PV_Alt), 'time step': self.current_step}
            self.current_step += 1
        else: 
            truncated = False 
            self.info = {'solar radiation': self.sun.get_radiation(self.PV_Az, self.PV_Alt, self.current_step), 'Maximum radiation': self.max_energy, 'solar panel angles': (self.PV_Az, self.PV_Alt), 'time step': self.current_step}

        terminated = self.current_step == 43

        if terminated: 
            self.reset()
            
        self.tryouts_same_time_step += 1 

        return self._get_observation(action), reward, terminated, truncated, self.info
    
    
    def get_reward(self, action): 
        #We compare the energy currently received by the solar panels to the maximum energy that would have been received if solar panels are perpendicular to the sun
        current_energy = self.sun.get_radiation(self.PV_Az, self.PV_Alt, self.current_step)
        self.energies.append(current_energy)

        if self.current_step <= 25:
            if current_energy >= 0.9 * self.max_energy and current_energy > self.energies[-2] and action[0]>= self.actions_history[-2][0]:
                reward = 2
            elif current_energy >= 0.9 * self.max_energy and current_energy < self.energies[-2]:
                reward = 0.5    
            elif current_energy <= 0.5 * self.max_energy:
                reward = -1
            elif current_energy > 0.5 * self.max_energy and current_energy < 0.9*self.max_energy:
                reward = 0.2
            else: 
                reward = 0

        if self.current_step > 25:  #peak at noon 
            if current_energy >= 0.9 * self.max_energy and current_energy < self.energies[-2] and action[0]< self.actions_history[-2][0]:
                reward = 2
            elif current_energy >= 0.9 * self.max_energy and current_energy > self.energies[-2]:
                reward = 0.5    
            elif current_energy <= 0.5 * self.max_energy:
                reward = -1
            elif current_energy > 0.5 * self.max_energy and current_energy < 0.9*self.max_energy:
                reward = 0.2
            else: 
                reward = 0
        return reward
    
    
    def _get_observation(self, action):
        self.PV_Az = self.sun.get_energy_perpendicular_to_sun(self.current_step - 1)[1] + action[0]
        self.PV_Alt = self.sun.get_energy_perpendicular_to_sun(self.current_step - 1)[2] + action[1]
        solar_radiation = np.clip(self.sun.get_radiation(self.PV_Az, self.PV_Alt, self.current_step), 0, 15000)
        solar_panels_tilt = self.solar_panel.get_tilt_angle(action)
        observation = [solar_radiation, self.PV_Az, self.PV_Alt] 
        return np.array(observation, dtype=np.float32)  
            
    def close(self):
        pass
