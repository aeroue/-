#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy

class BayesianBeliefUpdate:
    """贝叶斯信念更新与联盟演化"""
    
    def __init__(self, num_robots, num_tasks, num_types, pseudocount=1.0):
        """
        初始化贝叶斯信念更新模块
        
        参数:
            num_robots: 航天器数量
            num_tasks: 任务数量
            num_types: 任务类型数量
            pseudocount: 伪计数
        """
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.num_types = num_types
        self.pseudocount = pseudocount
        
        # 初始化观测矩阵，维度为 [num_robots, num_tasks+1, num_types]
        # 注意：任务ID从1开始，0为虚拟任务
        self.observation_matrices = {}
        
        for i in range(num_robots):
            self.observation_matrices[i] = np.zeros((num_tasks + 1, num_types))
    
    def initialize_beliefs(self, robots, belief_type='uniform'):
        """
        初始化航天器的信念
        
        参数:
            robots: 航天器字典
            belief_type: 信念初始化类型
        """
        for robot_id, robot in robots.items():
            # 初始化为正确的形状
            robot.local_belief = np.zeros((self.num_tasks + 1, self.num_types))
            
            if belief_type == 'uniform':
                # 均匀分布信念
                robot.local_belief[1:, :] = 1.0 / self.num_types
            elif belief_type == 'arbitrary':
                # 随机信念
                for j in range(1, self.num_tasks + 1):
                    random_vector = np.random.rand(self.num_types) + 1e-6
                    robot.local_belief[j, :] = random_vector / np.sum(random_vector)
            else:
                # 默认均匀分布
                robot.local_belief[1:, :] = 1.0 / self.num_types
            
            # 虚拟任务的信念设为0
            robot.local_belief[0, :] = 0
            
            # 初始化观测矩阵
            robot.observation_matrix = np.zeros((self.num_tasks + 1, self.num_types))
    
    def take_observation(self, robot, task, tasks):
        """
        航天器观测任务
        
        参数:
            robot: 航天器对象
            task: 任务ID
            tasks: 任务字典
            
        返回:
            observed_type: 观测到的类型
        """
        if task == 0 or task not in tasks:
            return None
        
        # 获取任务真实类型
        true_type = tasks[task].true_type
        
        # 以一定概率正确观测
        if np.random.rand() < robot.positive_observation_prob:
            return true_type
        else:
            # 随机选择一个错误类型
            possible_false_types = [k for k in range(self.num_types) if k != true_type]
            if not possible_false_types:
                return true_type
            
            return np.random.choice(possible_false_types)
    
    def update_observations(self, robots, tasks, observations_per_round=20):
        """
        更新任务观测
        
        参数:
            robots: 航天器字典
            tasks: 任务字典
            observations_per_round: 每轮观测次数
            
        返回:
            aggregated_observations: 聚合的观测结果
        """
        # 初始化聚合观测矩阵
        aggregated_observations = np.zeros((self.num_tasks + 1, self.num_types))
        
        # 每个航天器观测其当前任务
        for robot_id, robot in robots.items():
            task_id = robot.current_coalition_id
            
            # 跳过虚拟任务或不存在的任务
            if task_id == 0 or task_id not in tasks:
                continue
            
            # 进行多次观测
            for _ in range(observations_per_round):
                observed_type = self.take_observation(robot, task_id, tasks)
                
                if observed_type is not None and 0 <= observed_type < self.num_types:
                    aggregated_observations[task_id, observed_type] += 1
        
        return aggregated_observations
    
    def update_beliefs(self, robots, aggregated_observations):
        """
        更新航天器信念
        
        参数:
            robots: 航天器字典
            aggregated_observations: 聚合的观测结果
        """
        # 更新每个航天器的观测矩阵和信念
        for robot_id, robot in robots.items():
            # 更新观测矩阵
            robot.observation_matrix += aggregated_observations
            
            # 更新每个任务的信念
            for j in range(1, self.num_tasks + 1):
                # 获取当前累积的计数
                current_counts = robot.observation_matrix[j, :]
                
                # 添加伪计数
                alpha = current_counts + self.pseudocount
                
                # 计算新的信念
                sum_alpha = np.sum(alpha)
                
                if sum_alpha > 1e-9:
                    robot.local_belief[j, :] = alpha / sum_alpha
                else:
                    # 如果没有观测，保持均匀分布
                    robot.local_belief[j, :] = 1.0 / self.num_types
    
    def run_belief_update(self, robots, tasks, current_partition, observations_per_round=20):
        """
        执行一轮信念更新
        
        参数:
            robots: 航天器字典
            tasks: 任务字典
            current_partition: 当前联盟分配
            observations_per_round: 每轮观测次数
            
        返回:
            updated_robots: 更新后的航天器字典
        """
        # 复制航天器字典以避免修改原始数据
        updated_robots = copy.deepcopy(robots)
        
        # 更新每个航天器的当前联盟ID
        for robot_id, task_id in current_partition.items():
            if robot_id in updated_robots:
                updated_robots[robot_id].current_coalition_id = task_id
        
        # 获取聚合观测
        aggregated_observations = self.update_observations(
            updated_robots, tasks, observations_per_round
        )
        
        # 更新信念
        self.update_beliefs(updated_robots, aggregated_observations)
        
        return updated_robots