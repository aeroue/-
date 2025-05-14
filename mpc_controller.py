#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
import time

class SpectralMPC:
    """基于谱分析的模型预测控制器"""
    
    def __init__(self, A, B, Q, R, P, N_horizon, N_control, umax, dt=5.0):
        """
        初始化MPC控制器
        
        参数:
            A: 系统矩阵 (离散时间)
            B: 控制矩阵 (离散时间)
            Q: 状态权重矩阵
            R: 控制权重矩阵
            P: 终端状态权重矩阵
            N_horizon: 预测时域长度
            N_control: 控制时域长度
            umax: 控制输入约束值
            dt: 采样时间 (s)
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.N = N_horizon
        self.Nc = N_control
        self.umax = umax
        self.dt = dt
        
        # 获取系统维度
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        
        # 构建H步可控度矩阵
        self.C_H = self.build_controllability_matrix(A, B, N_horizon)
        
        # 进行谱分析
        self.eigenvalues, self.eigenvectors = self.spectral_analysis()
    
    def build_controllability_matrix(self, A, B, H):
        """构建H步可控度矩阵"""
        n_states = A.shape[0]
        n_inputs = B.shape[1]
        
        # 初始化可控度矩阵
        C_H = np.zeros((n_states, n_inputs * H))
        
        # 计算 B, AB, A^2B, ..., A^(H-1)B
        temp_B = B.copy()
        for i in range(H):
            C_H[:, i*n_inputs:(i+1)*n_inputs] = temp_B
            temp_B = A @ temp_B
        
        return C_H
    
    def spectral_analysis(self):
        """对可控度矩阵进行谱分析"""
        # 计算 C_H * C_H^T
        G = self.C_H @ self.C_H.T
        
        # 确保G是对称矩阵
        G = (G + G.T) / 2.0
        
        # 特征值分解
        eigenvalues, eigenvectors = eigh(G)
        
        # 按特征值降序排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 过滤掉接近零的特征值
        tol = 1e-10
        valid_indices = eigenvalues > tol
        eigenvalues = eigenvalues[valid_indices]
        eigenvectors = eigenvectors[:, valid_indices]
        
        return eigenvalues, eigenvectors
    
    def select_control_modes(self, target_deviation, num_modes=None):
        """
        选择控制模态实现目标状态偏差
        
        参数:
            target_deviation: 目标状态偏差
            num_modes: 要使用的模态数量，如果为None则使用所有有效模态
            
        返回:
            alpha: 模态权重
        """
        if num_modes is None:
            num_modes = self.eigenvalues.size
        else:
            num_modes = min(num_modes, self.eigenvalues.size)
        
        # 计算每个模态的权重
        alpha = np.zeros(num_modes)
        for j in range(num_modes):
            alpha[j] = target_deviation @ self.eigenvectors[:, j]
        
        return alpha
    
    def solve_mpc(self, current_state, target_state_ref):
        """
        求解MPC问题
        
        参数:
            current_state: 当前状态
            target_state_ref: 目标状态参考轨迹，形状为(n_states, N_horizon+1)
            
        返回:
            optimal_u0: 最优控制输入
            optimal_u_sequence: 完整最优控制序列
        """
        start_time = time.time()
        
        # 定义优化变量
        U = cp.Variable((self.n_inputs, self.Nc))
        X = cp.Variable((self.n_states, self.N + 1))
        
        # 初始化代价函数
        cost = 0
        
        # 累积预测时域内的代价
        for k in range(self.N):
            # 确定k时刻的控制输入
            if k < self.Nc:
                u_k = U[:, k]
            else:
                u_k = U[:, self.Nc-1]  # 使用最后一个控制输入
            
            # 状态追踪误差代价
            cost += cp.quad_form(X[:, k+1] - target_state_ref[:, k+1], self.Q)
            
            # 控制输入代价
            if k < self.Nc:
                cost += cp.quad_form(u_k, self.R)
        
        # 终端代价
        cost += cp.quad_form(X[:, self.N] - target_state_ref[:, self.N], self.P)
        
        # 约束条件
        constraints = [X[:, 0] == current_state]  # 初始状态约束
        
        # 系统动态约束
        for k in range(self.N):
            if k < self.Nc:
                u_k = U[:, k]
            else:
                u_k = U[:, self.Nc-1]  # 使用最后一个控制输入
                
            constraints += [X[:, k+1] == self.A @ X[:, k] + self.B @ u_k]
        
        # 控制输入约束
        for k in range(self.Nc):
            constraints += [cp.abs(U[:, k]) <= self.umax]
        
        # 终端状态约束（可选）
        # constraints += [X[:, self.N] == target_state_ref[:, self.N]]
        
        # 定义并求解优化问题
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        # 求解器选项
        solver_options = {
            'max_iter': 2000,
            'eps_abs': 1e-4,
            'eps_rel': 1e-4
        }
        
        try:
            # 首先尝试使用OSQP求解
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, **solver_options)
        except Exception as e:
            print(f"OSQP 求解失败，尝试使用 SCS: {e}")
            try:
                # 如果OSQP失败，尝试使用SCS
                problem.solve(solver=cp.SCS, warm_start=True, verbose=False)
            except Exception as e:
                print(f"SCS 求解也失败: {e}")
                return None, None
        
        # 检查求解状态
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if U.value is not None:
                optimal_u_sequence = U.value
                optimal_u0 = optimal_u_sequence[:, 0]
                
                end_time = time.time()
                #print(f"MPC 求解时间: {end_time - start_time:.4f}秒")
                
                return optimal_u0, optimal_u_sequence
            else:
                print(f"求解状态:{problem.status}，但 U.value 为 None")
                return np.zeros(self.n_inputs), None
        else:
            print(f"MPC 问题求解状态: {problem.status}")
            return np.zeros(self.n_inputs), None
    
    def parameterized_control_law(self, current_state, target_state, K, modal_weights):
        """
        使用参数化控制律计算控制输入
        
        参数:
            current_state: 当前状态
            target_state: 目标状态
            K: 反馈增益矩阵
            modal_weights: 模态权重
            
        返回:
            u: 控制输入
        """
        # 计算误差
        error = target_state - current_state
        
        # 基础反馈控制
        u_fb = K @ error
        
        # 模态控制
        u_modal = np.zeros(self.n_inputs)
        n_modes = min(len(modal_weights), self.eigenvalues.size)
        
        for j in range(n_modes):
            # 将特征向量投影到输入空间
            input_direction = self.B.T @ self.eigenvectors[:, j]
            if np.linalg.norm(input_direction) > 1e-10:
                input_direction = input_direction / np.linalg.norm(input_direction)
                u_modal += modal_weights[j] * input_direction
        
        # 组合控制输入
        u = u_fb + u_modal
        
        # 应用控制约束
        u = np.clip(u, -self.umax, self.umax)
        
        return u
    
    def generate_target_trajectory(self, current_state, target_state, N_steps):
        """
        生成线性参考轨迹
        
        参数:
            current_state: 当前状态
            target_state: 目标状态
            N_steps: 步数
            
        返回:
            trajectory: 参考轨迹，形状为(n_states, N_steps+1)
        """
        trajectory = np.zeros((self.n_states, N_steps+1))
        trajectory[:, 0] = current_state
        
        for i in range(1, N_steps+1):
            alpha = i / N_steps
            trajectory[:, i] = (1 - alpha) * current_state + alpha * target_state
        
        return trajectory