#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import copy
import time

class ActiveInference:
    """基于主动推理的弱通信条件下协同决策"""
    
    def __init__(self, num_robots, num_tasks, observation_error_std=2.0,
                 alpha=0.4, beta=0.6, horizon=5, max_reasoning_level=3,
                 dt=1.0, reward_configs=None):
        """
        初始化主动推理算法
        
        参数:
            num_robots: 航天器数量
            num_tasks: 任务数量
            observation_error_std: 观测误差标准差
            alpha: 认知价值权重
            beta: 实用价值权重
            horizon: 预测步数
            max_reasoning_level: 最大推理层次
            dt: 时间步长
            reward_configs: 有效的联盟配置列表
        """
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.observation_error_std = observation_error_std
        self.alpha = alpha
        self.beta = beta
        self.horizon = horizon
        self.max_reasoning_level = max_reasoning_level
        self.dt = dt
        
        # 设置有效的联盟配置
        if reward_configs is None:
            self.reward_configs = self.identify_reward_configs()
        else:
            self.reward_configs = reward_configs
            
        # 初始化信念
        self.beliefs = np.ones((num_robots, num_robots, num_tasks)) / num_tasks
        
        # 设置机动选项
        self.velocity_options = [0.5, 1.0, 1.5]
        self.heading_options = np.linspace(-np.pi/4, np.pi/4, 5)
        self.num_actions = len(self.velocity_options) * len(self.heading_options)
        
        # 记录历史信息
        self.belief_history = {i: {} for i in range(num_robots)}
        self.action_history = []
        self.free_energy_history = []
    
    def identify_reward_configs(self):
        """
        识别有效的联盟配置
        
        返回:
            reward_configs: 有效配置列表
        """
        import itertools
        
        # 简单情况：每个航天器分配一个任务
        if self.num_robots <= self.num_tasks:
            return list(itertools.permutations(range(self.num_tasks), self.num_robots))
        
        # 复杂情况：多个航天器可能分配到同一任务
        configs = []
        
        # 枚举所有可能的分配
        for config in itertools.product(range(self.num_tasks), repeat=self.num_robots):
            # 确保每个任务至少有一个航天器
            if set(config) == set(range(self.num_tasks)):
                configs.append(config)
        
        return configs
    
    def reset_beliefs(self):
        """重置所有信念为均匀分布"""
        self.beliefs = np.ones((self.num_robots, self.num_robots, self.num_tasks)) / self.num_tasks
        
        # 记录初始信念
        for robot_id in range(self.num_robots):
            for task_id in range(self.num_tasks):
                if task_id not in self.belief_history[robot_id]:
                    self.belief_history[robot_id][task_id] = []
                
                self.belief_history[robot_id][task_id].append(self.beliefs[robot_id, robot_id, :].copy())
    
    def simulate_observation(self, true_position, robot_id, observed_id):
        """
        模拟航天器对其他航天器的观测
        
        参数:
            true_position: 真实位置
            robot_id: 观测者ID
            observed_id: 被观测者ID
            
        返回:
            observation: 观测结果
        """
        # 自身观测无误差
        if robot_id == observed_id:
            return true_position.copy()
        
        # 其他航天器的观测带有随机误差
        error = np.random.normal(0, self.observation_error_std, true_position.shape)
        return true_position + error
    
    def calculate_evidence(self, positions, goals, robot_types, level=1, max_distance=30.0):
        """
        计算各航天器对各任务的证据强度
        
        参数:
            positions: 航天器位置
            goals: 任务位置
            robot_types: 航天器类型列表
            level: 推理层次
            max_distance: 最大距离参考值
            
        返回:
            evidence: 证据矩阵，形状为(num_robots, num_tasks)
        """
        # 计算距离证据
        def compute_distance_evidence(positions, goals, eta=max_distance):
            diff = positions[:, np.newaxis, :] - goals[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
            evidence = np.exp(-distances / eta)
            return evidence
        
        # 计算方向证据
        def compute_direction_evidence(positions, velocities, goals):
            diff = goals[np.newaxis, :, :] - positions[:, np.newaxis, :]
            directions_to_goals = diff / (np.linalg.norm(diff, axis=2)[:, :, np.newaxis] + 1e-10)
            
            velocities_norm = np.linalg.norm(velocities, axis=1)
            valid_velocities = velocities_norm > 1e-6
            
            evidence = np.zeros((positions.shape[0], goals.shape[0]))
            
            for i in range(positions.shape[0]):
                if valid_velocities[i]:
                    velocity_direction = velocities[i] / velocities_norm[i]
                    for j in range(goals.shape[0]):
                        evidence[i, j] = np.dot(velocity_direction, directions_to_goals[i, j])
            
            # 转换为[0, 1]范围
            evidence = (evidence + 1) / 2
            
            return evidence
        
        # 只有位置没有速度，使用距离证据
        distance_evidence = compute_distance_evidence(positions, goals)
        
        if level == 0:  # 零阶推理：只考虑自身对任务的距离
            return distance_evidence
        
        if level == 1:  # 一阶推理：考虑所有航天器对任务的距离
            return distance_evidence
        
        # 高阶推理：考虑航天器对其他航天器的观测和理解
        if level >= 2:
            # 复制一阶证据作为基础
            higher_order_evidence = distance_evidence.copy()
            
            # 对每对航天器计算二阶及以上证据
            for observer in range(self.num_robots):
                for observed in range(self.num_robots):
                    if observer != observed:
                        # 模拟观测者视角下的被观测者
                        observed_position = self.simulate_observation(positions[observed], observer, observed)
                        observed_evidence = compute_distance_evidence(np.array([observed_position]), goals)[0]
                        
                        # 将这一证据与原始证据融合
                        higher_order_evidence[observer] = 0.7 * higher_order_evidence[observer] + 0.3 * observed_evidence
            
            return higher_order_evidence
    
    def softmax(self, x, axis=1):
        """
        计算softmax函数
        
        参数:
            x: 输入数组
            axis: 计算softmax的轴
            
        返回:
            softmax结果
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-16)
    
    def update_belief(self, robot_id, observed_positions, goals, agent_types):
        """
        更新航天器的信念
        
        参数:
            robot_id: 航天器ID
            observed_positions: 观测到的位置
            goals: 任务位置
            agent_types: 航天器类型
            
        返回:
            updated_belief: 更新后的信念
        """
        # 基于不同层次的推理计算证据
        evidence_by_level = {}
        
        for level in range(self.max_reasoning_level + 1):
            evidence_by_level[level] = self.calculate_evidence(
                observed_positions, goals, agent_types, level=level
            )
        
        # 融合不同层次的证据
        # 级别越高，权重越小
        level_weights = {
            0: 0.1,  # 零阶推理权重
            1: 0.3,  # 一阶推理权重
            2: 0.4,  # 二阶推理权重
            3: 0.2   # 三阶推理权重
        }
        
        combined_evidence = np.zeros((self.num_robots, self.num_tasks))
        
        for level, evidence in evidence_by_level.items():
            if level <= self.max_reasoning_level:
                combined_evidence += level_weights[level] * evidence
        
        # 计算新的信念分布
        new_belief = self.softmax(combined_evidence, axis=1)
        
        # 更新机器人自身的信念
        old_belief = self.beliefs[robot_id].copy()
        
        # 使用贝叶斯更新
        for i in range(self.num_robots):
            # 计算后验概率
            posterior = old_belief[i] * new_belief[i]
            
            # 归一化
            if np.sum(posterior) > 0:
                posterior /= np.sum(posterior)
                self.beliefs[robot_id, i] = posterior
            else:
                self.beliefs[robot_id, i] = new_belief[i]
        
        # 记录信念历史
        for task_id in range(self.num_tasks):
            if task_id not in self.belief_history[robot_id]:
                self.belief_history[robot_id][task_id] = []
            
            self.belief_history[robot_id][task_id].append(self.beliefs[robot_id, robot_id, :].copy())
        
        return self.beliefs[robot_id]
    
    def predict_position(self, position, velocity, heading, dt):
        """
        预测航天器的下一个位置
        
        参数:
            position: 当前位置
            velocity: 速度大小
            heading: 航向角
            dt: 时间步长
            
        返回:
            next_position: 预测的下一个位置
        """
        next_position = position.copy()
        
        # 更新位置
        next_position[0] += velocity * np.cos(heading) * dt
        next_position[1] += velocity * np.sin(heading) * dt
        
        # 更新航向
        next_position[2] = heading
        
        return next_position
    
    def calculate_kl_divergence(self, p, q):
        """
        计算KL散度
        
        参数:
            p: 目标分布
            q: 当前分布
            
        返回:
            kl: KL散度
        """
        # 防止除以零和对零取对数
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        return np.sum(p * np.log(p / q))
    
    def calculate_shannon_entropy(self, p):
        """
        计算香农熵
        
        参数:
            p: 概率分布
            
        返回:
            entropy: 香农熵
        """
        # 防止对零取对数
        p = np.clip(p, 1e-10, 1.0)
        
        return -np.sum(p * np.log(p))
    
    def calculate_joint_probability(self, beliefs):
        """
        计算联合概率
        
        参数:
            beliefs: 信念数组
            
        返回:
            joint_prob: 联合概率
        """
        # 初始化联合概率
        joint_prob = np.ones(len(self.reward_configs))
        
        # 聚合所有航天器的信念
        aggregated_beliefs = np.sum(beliefs, axis=0) / self.num_robots
        
        # 计算每个有效配置的概率
        for i, config in enumerate(self.reward_configs):
            prob = 1.0
            
            for robot_id, task_id in enumerate(config):
                prob *= aggregated_beliefs[robot_id, task_id]
            
            joint_prob[i] = prob
        
        # 归一化
        joint_prob /= np.sum(joint_prob) + 1e-10
        
        return joint_prob
    
    def calculate_expected_free_energy(self, robot_id, current_positions, current_belief, action, goals, agent_types):
        """
        计算期望自由能
        
        参数:
            robot_id: 航天器ID
            current_positions: 当前位置
            current_belief: 当前信念
            action: 动作（速度和航向）
            goals: 任务位置
            agent_types: 航天器类型
            
        返回:
            efe: 期望自由能
        """
        # 分解动作
        velocity, heading = action
        
        # 预测下一个状态
        next_positions = current_positions.copy()
        next_positions[robot_id] = self.predict_position(
            current_positions[robot_id], velocity, heading, self.dt
        )
        
        # 计算新的信念
        new_belief = current_belief.copy()
        
        # 模拟下一步的观测和信念更新
        for observer_id in range(self.num_robots):
            for observed_id in range(self.num_robots):
                if observer_id != robot_id and observed_id != robot_id:
                    continue
                
                # 模拟观测
                observed_position = self.simulate_observation(
                    next_positions[observed_id], observer_id, observed_id
                )
                
                # 基于观测更新信念
                evidence = self.calculate_evidence(
                    np.array([observed_position]), goals, [agent_types[observed_id]], level=1
                )[0]
                
                new_belief[observer_id, observed_id] = self.softmax(evidence)
        
        # 计算认知价值（消除不确定性）
        cognitive_value = 0
        
        for observer_id in range(self.num_robots):
            entropy_before = self.calculate_shannon_entropy(current_belief[observer_id, robot_id])
            entropy_after = self.calculate_shannon_entropy(new_belief[observer_id, robot_id])
            cognitive_value += entropy_before - entropy_after
        
        # 计算实用价值（趋近目标）
        current_joint_prob = self.calculate_joint_probability(current_belief)
        new_joint_prob = self.calculate_joint_probability(new_belief)
        
        # 寻找最可能的配置
        most_likely_config_idx = np.argmax(new_joint_prob)
        most_likely_config = self.reward_configs[most_likely_config_idx]
        
        # 计算航天器到目标的距离
        target_task_id = most_likely_config[robot_id]
        target_position = goals[target_task_id]
        
        distance_to_target = np.linalg.norm(next_positions[robot_id][:2] - target_position)
        pragmatic_value = -distance_to_target / 100.0  # 归一化距离
        
        # 总期望自由能
        efe = -(self.alpha * cognitive_value + self.beta * pragmatic_value)
        
        return efe
    
    def choose_best_action(self, robot_id, positions, beliefs, goals, agent_types):
        """
        选择最佳动作
        
        参数:
            robot_id: 航天器ID
            positions: 航天器位置
            beliefs: 信念
            goals: 任务位置
            agent_types: 航天器类型
            
        返回:
            best_action: 最佳动作（速度和航向）
            best_efe: 最佳期望自由能
        """
        start_time = time.time()
        
        best_efe = float('inf')
        best_action = (0, 0)
        
        # 评估所有可能的动作
        for velocity in self.velocity_options:
            for heading in self.heading_options:
                action = (velocity, heading)
                
                # 计算该动作的期望自由能
                efe = self.calculate_expected_free_energy(
                    robot_id, positions, beliefs, action, goals, agent_types
                )
                
                # 如果期望自由能更低，则更新最佳动作
                if efe < best_efe:
                    best_efe = efe
                    best_action = action
        
        end_time = time.time()
        # print(f"动作评估耗时: {end_time - start_time:.4f}秒")
        
        return best_action, best_efe
    
    def make_decision(self, robot_id, positions, beliefs, goals, agent_types):
        """
        航天器决策
        
        参数:
            robot_id: 航天器ID
            positions: 航天器位置
            beliefs: 信念
            goals: 任务位置
            agent_types: 航天器类型
            
        返回:
            best_action: 最佳动作
            updated_belief: 更新后的信念
            best_efe: 最佳期望自由能
        """
        # 更新信念
        updated_belief = self.update_belief(robot_id, positions, goals, agent_types)
        
        # 选择最佳动作
        best_action, best_efe = self.choose_best_action(
            robot_id, positions, updated_belief, goals, agent_types
        )
        
        # 记录历史
        self.action_history.append((robot_id, best_action))
        self.free_energy_history.append((robot_id, best_efe))
        
        return best_action, updated_belief, best_efe
    
    def run_simulation(self, initial_positions, goals, agent_types, max_steps=100, convergence_distance=1.5):
        """
        运行仿真
        
        参数:
            initial_positions: 初始位置
            goals: 任务位置
            agent_types: 航天器类型
            max_steps: 最大步数
            convergence_distance: 收敛距离阈值
            
        返回:
            positions_history: 位置历史
            belief_history: 信念历史
            action_history: 动作历史
            converged: 是否收敛
            steps: 步数
        """
        # 重置信念
        self.reset_beliefs()
        
        # 初始化历史记录
        positions_history = [initial_positions.copy()]
        
        # 当前位置
        current_positions = initial_positions.copy()
        
        print("开始主动推理仿真...")
        
        # 主循环
        for step in range(max_steps):
            print(f"步骤 {step+1}/{max_steps}")
            
            # 每个航天器做决策
            for robot_id in range(self.num_robots):
                # 航天器决策
                best_action, updated_belief, best_efe = self.make_decision(
                    robot_id, current_positions, self.beliefs, goals, agent_types
                )
                
                # 更新位置
                velocity, heading = best_action
                current_positions[robot_id] = self.predict_position(
                    current_positions[robot_id], velocity, heading, self.dt
                )
                
                print(f"  航天器 {robot_id}: 动作=({velocity:.2f}, {heading:.2f}), EFE={best_efe:.4f}")
            
            # 记录历史
            positions_history.append(current_positions.copy())
            
            # 检查收敛
            converged = True
            for robot_id in range(self.num_robots):
                # 获取航天器当前最可能的任务
                most_likely_task = np.argmax(self.beliefs[robot_id, robot_id])
                
                # 计算到该任务的距离
                distance = np.linalg.norm(current_positions[robot_id][:2] - goals[most_likely_task])
                
                if distance > convergence_distance:
                    converged = False
                    break
            
            if converged:
                print(f"仿真在 {step+1} 步后收敛")
                break
        
        if not converged:
            print(f"仿真在最大步数 {max_steps} 后未收敛")
        
        return positions_history, self.belief_history, self.action_history, converged, step + 1