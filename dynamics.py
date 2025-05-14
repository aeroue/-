#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete

class CWDynamics:
    """Clohessy-Wiltshire (CW) 方程动力学模型"""
    
    def __init__(self, orbit_altitude=400.0, dt=5.0):
        """
        初始化CW动力学模型
        
        参数:
            orbit_altitude: 参考轨道高度（单位：km）
            dt: 采样时间（单位：s）
        """
        self.G = 6.67430e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
        self.M_earth = 5.972e24  # 地球质量 (kg)
        self.R_earth = 6371.0  # 地球半径 (km)
        self.orbit_altitude = orbit_altitude  # 轨道高度 (km)
        self.r_orbit = (self.R_earth + self.orbit_altitude) * 1000  # 轨道半径 (m)
        self.n = np.sqrt(self.G * self.M_earth / (self.r_orbit ** 3))  # 平均角速度 (rad/s)
        self.dt = dt  # 采样时间 (s)
        
        # 计算连续时间状态空间矩阵
        self.A_cont, self.B_cont = self.calculate_cw_matrices()
        
        # 计算离散时间状态空间矩阵
        self.A_d, self.B_d = self.discretize_system(self.A_cont, self.B_cont, self.dt)
    
    def calculate_cw_matrices(self):
        """计算CW方程的连续时间状态空间矩阵 A 和 B"""
        n = self.n
        
        # 状态矩阵 A
        A_cont = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n**2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -n**2, 0, 0, 0]
        ])
        
        # 控制矩阵 B
        B_cont = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        return A_cont, B_cont
    
    def discretize_system(self, A_cont, B_cont, dt):
        """使用零阶保持器（ZOH）方法离散化状态空间模型"""
        n_states = A_cont.shape[0]
        n_inputs = B_cont.shape[1]
        
        # 构造扩展矩阵
        M = np.zeros((n_states + n_inputs, n_states + n_inputs))
        M[:n_states, :n_states] = A_cont
        M[:n_states, n_states:] = B_cont
        
        # 计算矩阵指数
        M_exp = expm(M * dt)
        
        # 提取离散矩阵
        A_d = M_exp[:n_states, :n_states]
        B_d = M_exp[:n_states, n_states:]
        
        return A_d, B_d
    
    def update_state(self, current_state, control_input):
        """
        更新航天器状态
        
        参数:
            current_state: 当前状态向量 [x, y, z, vx, vy, vz]
            control_input: 控制输入向量 [ax, ay, az]
            
        返回:
            next_state: 更新后的状态向量
        """
        # 确保输入是numpy数组
        current_state = np.array(current_state).reshape(-1)
        control_input = np.array(control_input).reshape(-1)
        
        # 状态更新方程
        next_state = self.A_d @ current_state + self.B_d @ control_input
        
        return next_state
    
    def calculate_delta_v(self, control_sequence):
        """
        计算给定控制序列的Delta-V总量
        
        参数:
            control_sequence: 控制序列，形状为(n_steps, 3)
            
        返回:
            delta_v: 总Delta-V (m/s)
        """
        delta_v = 0.0
        for control in control_sequence:
            delta_v += np.linalg.norm(control) * self.dt
        
        return delta_v
    
    def calculate_fuel_cost(self, delta_v, initial_mass, dry_mass, isp):
        """
        使用齐奥尔科夫斯基火箭方程计算燃料消耗
        
        参数:
            delta_v: 速度变化量 (m/s)
            initial_mass: 初始质量 (kg)
            dry_mass: 干质量 (kg)
            isp: 比冲 (s)
            
        返回:
            fuel_mass: 消耗的燃料质量 (kg)
        """
        g0 = 9.80665  # 标准重力加速度 (m/s^2)
        
        # 确保Delta-V为正数
        if delta_v <= 0:
            return 0.0
        
        # 计算有效排气速度
        ve = isp * g0
        
        # 使用齐奥尔科夫斯基火箭方程
        try:
            mass_ratio = np.exp(delta_v / ve)
            final_mass = initial_mass / mass_ratio
            
            # 确保最终质量不小于干质量
            if final_mass < dry_mass:
                return initial_mass - dry_mass  # 返回最大可用燃料
            
            fuel_mass = initial_mass - final_mass
            return fuel_mass
            
        except OverflowError:
            # 当Delta-V/ve过大时，可能导致指数溢出
            return initial_mass - dry_mass  # 返回最大可用燃料