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
class Sun:
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, N, Dn, render_mode: Optional[str] = None):
        self.N = 157
        self.Dn = 1
        self.La = 48.427157 
        self.Lo = -68.548517
        self.t_GMT = -4
        self.Rho = 0.9
        self.step = 1 / 60

    def generate_radiation(self, PV_Az, PV_Alt):
        T_p = self.N + (self.Dn - 1) 
        self.PV_Az = PV_Az
        self.PV_Alt = PV_Alt

        Alpha = []
        Theta = []
        Delta = []
        G_tot = []

        for N in range(self.N, T_p + 1):
            A = 1162.12 + 77.0323 * np.cos(np.deg2rad(N * 360 / 365))
            B = 0.171076 - 0.0348944 * np.cos(np.deg2rad(N * 360 / 365))
            C = 0.0897334 - 0.0412439 * np.cos(np.deg2rad(N * 360 / 365))

            Decl_ang = 23.45 * np.sin(np.deg2rad(360 * (N - 81) / 365))
            Bt = 360 * (N - 81) / 365
            EoT = 9.87 * np.sin(np.deg2rad(2 * Bt)) - 7.53 * np.cos(np.deg2rad(Bt)) - 1.5 * np.sin(np.deg2rad(Bt))
            LMST = abs(self.t_GMT) * 15
            if self.Lo >= 0:
                AST_corr = ((4 * (abs(self.Lo) - LMST)) + EoT) / 60
            else:
                AST_corr = ((-4 * (abs(self.Lo) - LMST)) + EoT) / 60
            Omega_sr_ss = np.degrees(np.arccos((-np.tan(np.deg2rad(self.La))) * (np.tan(np.deg2rad(Decl_ang)))))
            AST_sr = abs((((Omega_sr_ss / 15) - 12)))
            AST_ss = (((Omega_sr_ss / 15) + 12))
            Ltsr = AST_sr + abs(AST_corr)
            Ltss = AST_ss + abs(AST_corr)
            Day_L = Ltss - Ltsr
            Mid_L = Ltsr + (Day_L / 2)
            Alpha_n = []
            Theta_n = []
            Delta_n = []
            G_tot_n = []

            for LSt in np.arange(Ltsr, Ltss + self.step, self.step):
                L_Sol_t = LSt + AST_corr
                Omega = (L_Sol_t - 12) * 15
                sin_Alpha = np.cos(np.deg2rad(self.La)) * np.cos(np.deg2rad(Omega)) * np.cos(np.deg2rad(Decl_ang)) + np.sin(
                    np.deg2rad(self.La)) * np.sin(np.deg2rad(Decl_ang))
                if sin_Alpha < 0:
                    sin_Alpha = 0
                Alpha_i = np.degrees(np.arcsin(sin_Alpha))
                Alpha_n = np.append(Alpha_n, Alpha_i)
                cos_Theta = (np.sin(np.deg2rad(Decl_ang)) * np.cos(np.deg2rad(self.La)) - np.cos(
                    np.deg2rad(Decl_ang)) * np.sin(np.deg2rad(self.La)) * np.cos(np.deg2rad(Omega))) / np.cos(
                    np.deg2rad(Alpha_i))
                Theta_i = np.degrees(np.arccos(cos_Theta))
                if LSt > Mid_L:
                    Theta_i = -Theta_i + 360
                elif LSt <= Mid_L:
                    pass
                Theta_n = np.append(Theta_n, Theta_i)
                Gamma = np.abs(Theta_i - self.PV_Az)
                cos_Delta = np.cos(np.deg2rad(Alpha_i)) * np.cos(np.deg2rad(Gamma)) * np.sin(
                    np.deg2rad(self.PV_Alt)) + np.sin(np.deg2rad(Alpha_i)) * np.cos(np.deg2rad(self.PV_Alt))
                Delta_i = np.degrees(np.arccos(cos_Delta))

                Gn_direct = A * (C + sin_Alpha) / np.exp(B / (sin_Alpha))
                G_direct = Gn_direct * max(np.cos(np.deg2rad(Delta_i)), 0)
                Gn_diffuse = C * Gn_direct
                G_diffuse = Gn_diffuse * ((1 + np.cos(np.deg2rad(self.PV_Alt))) / 2)
                G_Tn = G_direct + G_diffuse
                G_refl = G_Tn * self.Rho * ((1 - np.cos(np.deg2rad(self.PV_Alt))) / 2)
                G_tot_i = (G_direct + G_diffuse + G_refl)
                G_tot_n = np.append(G_tot_n, G_tot_i)

        new_G_tot_n = G_tot_n[60 * ceil(Ltsr): 60 * floor(Ltss)]
        #print(len(new_G_tot_n), "len new_G_tot_n")
        return new_G_tot_n
    

    def get_sun_angles_throughout_day(self):
        T_p = self.N + (self.Dn - 1) 
        Alpha = []
        Theta = []
        Delta = []

        for N in range(self.N, T_p + 1):
            A = 1162.12 + 77.0323 * np.cos(np.deg2rad(N * 360 / 365))
            B = 0.171076 - 0.0348944 * np.cos(np.deg2rad(N * 360 / 365))
            C = 0.0897334 - 0.0412439 * np.cos(np.deg2rad(N * 360 / 365))

            Decl_ang = 23.45 * np.sin(np.deg2rad(360 * (N - 81) / 365))
            Bt = 360 * (N - 81) / 365
            EoT = 9.87 * np.sin(np.deg2rad(2 * Bt)) - 7.53 * np.cos(np.deg2rad(Bt)) - 1.5 * np.sin(np.deg2rad(Bt))
            LMST = abs(self.t_GMT) * 15
            if self.Lo >= 0:
                AST_corr = ((4 * (abs(self.Lo) - LMST)) + EoT) / 60
            else:
                AST_corr = ((-4 * (abs(self.Lo) - LMST)) + EoT) / 60
            Omega_sr_ss = np.degrees(np.arccos((-np.tan(np.deg2rad(self.La))) * (np.tan(np.deg2rad(Decl_ang)))))
            AST_sr = abs((((Omega_sr_ss / 15) - 12)))
            AST_ss = (((Omega_sr_ss / 15) + 12))
            Ltsr = AST_sr + abs(AST_corr)
            Ltss = AST_ss + abs(AST_corr)
            Day_L = Ltss - Ltsr
            Mid_L = Ltsr + (Day_L / 2)
            Alpha_n = []
            Theta_n = []
            Delta_n = []
            G_tot_n = []

            for LSt in np.arange(Ltsr, Ltss + self.step, self.step):
                L_Sol_t = LSt + AST_corr
                Omega = (L_Sol_t - 12) * 15
                sin_Alpha = np.cos(np.deg2rad(self.La)) * np.cos(np.deg2rad(Omega)) * np.cos(np.deg2rad(Decl_ang)) + np.sin(
                    np.deg2rad(self.La)) * np.sin(np.deg2rad(Decl_ang))
                if sin_Alpha < 0:
                    sin_Alpha = 0
                Alpha_i = np.degrees(np.arcsin(sin_Alpha))
                Alpha_n = np.append(Alpha_n, Alpha_i)
                cos_Theta = (np.sin(np.deg2rad(Decl_ang)) * np.cos(np.deg2rad(self.La)) - np.cos(
                    np.deg2rad(Decl_ang)) * np.sin(np.deg2rad(self.La)) * np.cos(np.deg2rad(Omega))) / np.cos(
                    np.deg2rad(Alpha_i))
                Theta_i = np.degrees(np.arccos(cos_Theta))
                if LSt > Mid_L:
                    Theta_i = -Theta_i + 360
                elif LSt <= Mid_L:
                    pass
                Theta_n = np.append(Theta_n, Theta_i)

        new_Alpha_n = Alpha_n[60 * ceil(Ltsr): 60 * floor(Ltss)]
        new_Theta_n = Theta_n[60 * ceil(Ltsr): 60 * floor(Ltss)]
        return new_Alpha_n, new_Theta_n
    
    def get_energy_perpendicular_to_sun(self, time_step):
        sun_altitude7, sun_azimuth7 = Sun.get_sun_angles_throughout_day(self)[0][time_step*15 + 7], Sun.get_sun_angles_throughout_day(self)[1][time_step*15 + 7]
        sun_altitude1, sun_azimuth1 = Sun.get_sun_angles_throughout_day(self)[0][time_step*15], Sun.get_sun_angles_throughout_day(self)[1][time_step*15]
        # Calculate solar zenith angle
        solar_zenith_angle1 = 90 - sun_altitude1
        solar_zenith_angle7 = 90 - sun_altitude7
        # Convert sun's azimuth angle to standard angle
        if sun_azimuth1 < 180:
            standard_azimuth1 = sun_azimuth1
        else:
            standard_azimuth1 = 360 - sun_azimuth1
        # Calculate PV_Alt
        pv_alt_perp_to_sun1  = 90 - solar_zenith_angle1
        # Calculate PV_Az
        if standard_azimuth1 < 180:
            pv_az_perp_to_sun1 = 180 - standard_azimuth1
        else:
            pv_az_perp_to_sun1  = standard_azimuth1 - 180
        if sun_azimuth7 < 180:
            standard_azimuth7 = sun_azimuth7
        else:
            standard_azimuth7 = 360 - sun_azimuth7
        # Calculate PV_Alt
        pv_alt_perp_to_sun7  = 90 - solar_zenith_angle7
        # Calculate PV_Az
        if standard_azimuth7 < 180:
            pv_az_perp_to_sun7 = 180 - standard_azimuth7
        else:
            pv_az_perp_to_sun7  = standard_azimuth7 - 180
        
        # We return the sum of energies received by solar panels during the whole period of 15 min 
        #print(Sun.generate_radiation(self, pv_az_perp_to_sun7, pv_alt_perp_to_sun7)[(time_step - 1) * 15 : time_step * 15])
        #print(len(Sun.generate_radiation(self, pv_az_perp_to_sun7, pv_alt_perp_to_sun7)[(time_step - 1) * 15 : time_step * 15]))
        Energy_rad = sum(Sun.generate_radiation(self, pv_az_perp_to_sun7, pv_alt_perp_to_sun7)[(time_step - 1) * 15 : time_step * 15])
        return Energy_rad, pv_alt_perp_to_sun7, pv_az_perp_to_sun7, pv_alt_perp_to_sun1, pv_az_perp_to_sun1

    #get radiation for current time_step (for whole 15 minutes)
    def get_radiation(self, PV_Az, PV_Alt, time_step):
        #print(sum(Sun.generate_radiation(self, PV_Az, PV_Alt)[(time_step - 1) * 15 : time_step * 15]), "sum radiation")
        return sum(Sun.generate_radiation(self, PV_Az, PV_Alt)[(time_step - 1) * 15 : time_step * 15])
    
