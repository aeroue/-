#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import copy
from collections import defaultdict

class Task:
    """任务类"""
    
    def __init__(self, id, position, true_type):
        """
        初始化任务
        
        参数:
            id: 任务ID
            position: 任务位置
            true_type: 任务真实类型
        """
        self.id = id
        self.position = position
        self.true_type = true_type
        
        # 任务状态相关属性
        self.relative_state = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0])
        self.is_completed = False
        
        # 任务价值和风险参数
        self.TASK_REVENUE = {0: 300, 1: 500, 2: 1000}
        self.TASK_RISK_COST = {0: 10, 1: 30, 2: 80}
    
    def get_revenue(self, task_type):
        """获取任务类型对应的收益"""
        return self.TASK_REVENUE.get(task_type, 0)
    
    def calculate_risk_cost(self, robot, task_type):
        """计算任务类型对应的风险成本"""
        return self.TASK_RISK_COST.get(task_type, 0)
    
    def get_actual_revenue(self):
        """获取任务实际收益"""
        return self.get_revenue(self.true_type)
    
    def calculate_actual_risk_cost(self, robot):
        """计算任务实际风险成本"""
        return self.calculate_risk_cost(robot, self.true_type)

class Spacecraft:
    """航天器类"""
    
    def __init__(self, id, position, num_tasks, num_types, initial_mass=1000.0, 
                dry_mass=200.0, isp=300.0, observation_prob=0.9, type='A'):
        """
        初始化航天器
        
        参数:
            id: 航天器ID
            position: 初始位置
            num_tasks: 任务数量
            num_types: 任务类型数量
            initial_mass: 初始质量 (kg)
            dry_mass: 干质量 (kg)
            isp: 比冲 (s)
            observation_prob: 正确观测概率
            type: 航天器类型
        """
        self.id = id
        self.position = position
        self.num_tasks = num_tasks
        self.num_types = num_types
        self.initial_mass = initial_mass
        self.dry_mass = dry_mass
        self.isp = isp
        self.positive_observation_prob = observation_prob
        self.type = type
        
        # 航天器状态相关属性
        self.relative_state = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0])
        self.current_coalition_id = 0  # 初始分配到虚拟任务
        self.neighbors = []
        
        # 信念相关属性
        self.local_belief = np.zeros((num_tasks + 1, num_types))
        self.observation_matrix = np.zeros((num_tasks + 1, num_types))
        
        # CW动力学相关矩阵
        self.Ad = None
        self.Bd = None
        
        # 轨迹记录
        self.trajectory = [list(self.position)]
        
        # 状态更新标志
        self.needs_update = True
    
    def update_physical_state(self, control_input, dt):
        """
        更新航天器物理状态
        
        参数:
            control_input: 控制输入
            dt: 时间步长
        """
        if self.Ad is None or self.Bd is None:
            # 如果没有初始化动力学矩阵，简单更新位置
            velocity = np.linalg.norm(control_input[:2])
            heading = np.arctan2(control_input[1], control_input[0])
            
            self.position = (
                self.position[0] + velocity * np.cos(heading) * dt,
                self.position[1] + velocity * np.sin(heading) * dt
            )
            self.relative_state[0] = self.position[0]
            self.relative_state[1] = self.position[1]
        else:
            # 使用CW动力学更新状态
            if control_input is None:
                control_input = np.zeros(self.Bd.shape[1])
            
            control_input = np.array(control_input).flatten()
            
            if control_input.shape[0] != self.Bd.shape[1]:
                control_input = np.zeros(self.Bd.shape[1])
            
            try:
                next_state = self.Ad @ self.relative_state + self.Bd @ control_input
                self.relative_state = next_state
                self.position = (self.relative_state[0], self.relative_state[1])
            except Exception as e:
                print(f"状态更新计算错误，航天器{self.id}: {e}")
        
        # 记录轨迹
        self.trajectory.append(list(self.relative_state))
    
    def estimate_delta_v_for_task(self, task):
        """
        估算执行任务所需的Delta-V
        
        参数:
            task: 任务对象
            
        返回:
            delta_v: 估算的Delta-V
        """
        if task is None:
            return 0.0
        
        # 计算直线距离
        distance = np.linalg.norm(np.array(self.position) - np.array(task.position))
        
        # 简单估算：Delta-V与距离成正比
        estimated_delta_v = distance * 0.05
        
        # 限制在合理范围内
        return max(0.0, min(estimated_delta_v, 100.0))
    
    def calculate_fuel_cost(self, delta_v):
        """
        计算燃料成本
        
        参数:
            delta_v: Delta-V
            
        返回:
            fuel_cost: 燃料成本
        """
        # 如果无需速度增量，则无需燃料
        if delta_v <= 0:
            return 0.0
        
        # 如果比冲无效，则无法计算
        if self.isp <= 0:
            return float('inf')
        
        # 计算有效排气速度
        g0 = 9.80665  # 标准重力加速度
        ve = self.isp * g0
        
        # 使用齐奥尔科夫斯基火箭方程
        try:
            exponent = delta_v / ve
            if exponent > 700:  # 防止溢出
                return float('inf')
                
            mass_ratio = np.exp(exponent)
            m_initial = self.initial_mass
            m_final = m_initial / mass_ratio
            
            # 计算消耗的燃料
            fuel_consumed = m_initial - m_final
            
            # 检查燃料是否足够
            max_fuel = self.initial_mass - self.dry_mass
            if fuel_consumed > max_fuel + 1e-9:
                return float('inf')
                
            return fuel_consumed
            
        except (OverflowError, ZeroDivisionError):
            return float('inf')
    
    def calculate_expected_utility(self, task_id, current_partition, all_tasks):
        """
        计算期望效用
        
        参数:
            task_id: 任务ID
            current_partition: 当前联盟分配
            all_tasks: 所有任务的字典
            
        返回:
            utility: 期望效用
        """
        if task_id == 0:  # 虚拟任务的效用为0
            return 0
        
        # 获取当前任务联盟
        coalition = current_partition.get(task_id, []).copy()
        
        # 检查是否已经是成员
        is_member = self.id in coalition
        
        # 计算联盟规模
        if is_member:
            hypothetical_size = len(coalition)
        else:
            hypothetical_size = len(coalition) + 1
        
        # 确保联盟规模至少为1
        if hypothetical_size == 0:
            hypothetical_size = 1
        
        # 获取任务对象
        task = all_tasks.get(task_id)
        if not task:
            return -float('inf')
        
        # 计算期望收益和风险成本
        expected_revenue_term = 0
        expected_risk_cost_term = 0
        
        for k in range(self.num_types):
            belief_ijk = self.local_belief[task_id, k]
            revenue_k = task.get_revenue(k)
            risk_cost_ik = task.calculate_risk_cost(self, k)
            
            expected_revenue_term += belief_ijk * revenue_k
            expected_risk_cost_term += belief_ijk * risk_cost_ik
        
        # 计算共享收益
        expected_shared_revenue = expected_revenue_term / hypothetical_size
        
        # 计算燃料成本
        estimated_delta_v = self.estimate_delta_v_for_task(task)
        fuel_cost = self.calculate_fuel_cost(estimated_delta_v)
        
        if np.isinf(fuel_cost):
            return -float('inf')
        
        # 计算总效用
        utility = expected_shared_revenue - expected_risk_cost_term - fuel_cost
        
        return utility
    
    def calculate_actual_utility(self, task_id, final_partition, all_tasks):
        """
        计算实际效用
        
        参数:
            task_id: 任务ID
            final_partition: 最终联盟分配
            all_tasks: 所有任务的字典
            
        返回:
            utility: 实际效用
        """
        if task_id == 0:  # 虚拟任务的效用为0
            return 0
        
        # 获取任务联盟
        coalition = []
        for robot_id, assigned_task in final_partition.items():
            if assigned_task == task_id:
                coalition.append(robot_id)
        
        # 检查联盟是否为空或自身是否不在联盟中
        coalition_size = len(coalition)
        if coalition_size == 0 or self.id not in coalition:
            return 0
        
        # 获取任务对象
        task = all_tasks.get(task_id)
        if not task:
            return 0
        
        # 获取实际收益和风险成本
        actual_revenue = task.get_actual_revenue()
        actual_risk_cost = task.calculate_actual_risk_cost(self)
        
        # 计算燃料成本
        estimated_delta_v = self.estimate_delta_v_for_task(task)
        fuel_cost = self.calculate_fuel_cost(estimated_delta_v)
        
        if np.isinf(fuel_cost):
            return 0
        
        # 计算共享收益
        actual_shared_revenue = actual_revenue / coalition_size
        
        # 计算总效用
        utility = actual_shared_revenue - actual_risk_cost - fuel_cost
        
        return utility

class Simulation:
    """仿真环境类"""
    
    def __init__(self, env_size=100, comm_range=150, obs_per_round=20, 
                dt=5.0, use_cw_dynamics=True):
        """
        初始化仿真环境
        
        参数:
            env_size: 环境大小
            comm_range: 通信范围
            obs_per_round: 每轮观测次数
            dt: 时间步长
            use_cw_dynamics: 是否使用CW动力学
        """
        self.env_size = env_size
        self.comm_range = comm_range
        self.observations_per_round = obs_per_round
        self.dt = dt
        self.use_cw_dynamics = use_cw_dynamics
        
        # 仿真状态
        self.tasks = {}
        self.spacecrafts = {}
        self.time = 0.0
        
        # 历史记录
        self.assignment_history = []
        self.utility_history = []
        self.position_history = []
        self.belief_history = defaultdict(lambda: defaultdict(list))
    
    def initialize_tasks(self, num_tasks, task_types=None, min_distance=10.0):
        """
        初始化任务
        
        参数:
            num_tasks: 任务数量
            task_types: 任务类型列表
            min_distance: 最小距离
        """
        # 生成任务位置
        task_positions = []
        attempts = 0
        max_attempts = 1000
        
        while len(task_positions) < num_tasks and attempts < max_attempts:
            pos = (random.uniform(0, self.env_size), random.uniform(0, self.env_size))
            
            # 检查与已有任务的距离
            too_close = False
            for existing_pos in task_positions:
                dist = np.linalg.norm(np.array(pos) - np.array(existing_pos))
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                task_positions.append(pos)
            
            attempts += 1
        
        # 设置任务类型
        if task_types is None:
            task_types = [random.randint(0, 2) for _ in range(num_tasks)]
        
        # 创建任务对象
        for j in range(1, num_tasks + 1):
            if j - 1 < len(task_positions) and j - 1 < len(task_types):
                position = task_positions[j - 1]
                task_type = task_types[j - 1]
                self.tasks[j] = Task(j, position, task_type)
        
        print(f"初始化了 {len(self.tasks)} 个任务")
    
    def initialize_spacecrafts(self, num_spacecrafts, num_tasks, num_types, 
                              spacecraft_types=None, dynamics=None):
        """
        初始化航天器
        
        参数:
            num_spacecrafts: 航天器数量
            num_tasks: 任务数量
            num_types: 任务类型数量
            spacecraft_types: 航天器类型列表
            dynamics: 动力学模型
        """
        # 生成航天器位置
        sc_positions = []
        
        for _ in range(num_spacecrafts):
            pos = (random.uniform(0, self.env_size), random.uniform(0, self.env_size))
            sc_positions.append(pos)
        
        # 设置航天器类型
        if spacecraft_types is None:
            spacecraft_types = ['A'] * num_spacecrafts
        
        # 创建航天器对象
        for i in range(num_spacecrafts):
            position = sc_positions[i]
            sc_type = spacecraft_types[i] if i < len(spacecraft_types) else 'A'
            
            # 设置观测概率 (A型最高, B型中等, S型最低)
            if sc_type == 'A':
                obs_prob = random.uniform(0.9, 1.0)
            elif sc_type == 'B':
                obs_prob = random.uniform(0.8, 0.9)
            else:  # S型
                obs_prob = random.uniform(0.7, 0.8)
            
            self.spacecrafts[i] = Spacecraft(
                i, position, num_tasks, num_types, 
                observation_prob=obs_prob, type=sc_type
            )
            
            # 设置动力学矩阵
            if dynamics is not None and self.use_cw_dynamics:
                self.spacecrafts[i].Ad = dynamics.A_d
                self.spacecrafts[i].Bd = dynamics.B_d
        
        print(f"初始化了 {len(self.spacecrafts)} 个航天器")
    
    def update_comm_graph(self):
        """更新通信图"""
        for i, sc_i in self.spacecrafts.items():
            sc_i.neighbors = []
            
            for j, sc_j in self.spacecrafts.items():
                if i == j:
                    continue
                    
                dist = np.linalg.norm(np.array(sc_i.position) - np.array(sc_j.position))
                
                if dist <= self.comm_range:
                    sc_i.neighbors.append(j)
        
        # 确保通信图至少是连通的
        self._ensure_connected_graph()
    
    def _ensure_connected_graph(self):
        """确保通信图至少是连通的"""
        # 构建当前的通信图
        graph = defaultdict(list)
        
        for i, sc in self.spacecrafts.items():
            graph[i] = sc.neighbors
        
        # 检查连通性
        visited = set()
        
        def dfs(node):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        # 从第一个航天器开始深度优先搜索
        if self.spacecrafts:
            dfs(next(iter(self.spacecrafts)))
        
        # 如果没有访问所有航天器，则添加额外连接
        if len(visited) < len(self.spacecrafts):
            # 找出未连接的组件
            remaining = set(self.spacecrafts.keys()) - visited
            
            # 连接每个未访问的组件到已访问的任意节点
            visited_node = next(iter(visited))
            
            for node in remaining:
                # 向航天器的邻居列表添加连接
                self.spacecrafts[visited_node].neighbors.append(node)
                self.spacecrafts[node].neighbors.append(visited_node)
                
                # 更新已访问集合
                visited.add(node)
    
    def record_state(self):
        """记录当前状态"""
        # 记录位置
        positions = []
        for sc_id in sorted(self.spacecrafts.keys()):
            sc = self.spacecrafts[sc_id]
            positions.append(np.array(sc.relative_state))
        
        self.position_history.append(np.array(positions))
        
        # 记录任务分配
        assignment = {}
        for sc_id, sc in self.spacecrafts.items():
            assignment[sc_id] = sc.current_coalition_id
        
        self.assignment_history.append(assignment)
        
        # 记录信念
        for sc_id, sc in self.spacecrafts.items():
            for task_id in range(1, sc.num_tasks + 1):
                self.belief_history[sc_id][task_id].append(sc.local_belief[task_id, :].copy())
    
    def calculate_global_utility(self, assignment=None):
        """
        计算全局效用
        
        参数:
            assignment: 任务分配，如果为None则使用当前分配
            
        返回:
            global_utility: 全局效用
        """
        if assignment is None:
            assignment = {}
            for sc_id, sc in self.spacecrafts.items():
                assignment[sc_id] = sc.current_coalition_id
        
        global_utility = 0.0
        
        for sc_id, sc in self.spacecrafts.items():
            task_id = assignment[sc_id]
            utility = sc.calculate_actual_utility(task_id, assignment, self.tasks)
            global_utility += utility
        
        return global_utility
    
    def take_observations(self):
        """
        执行一轮观测
        
        返回:
            aggregated_observations: 聚合的观测结果
        """
        # 初始化聚合观测矩阵
        num_types = next(iter(self.spacecrafts.values())).num_types if self.spacecrafts else 0
        num_tasks = len(self.tasks)
        aggregated_observations = np.zeros((num_tasks + 1, num_types))
        
        # 每个航天器观测其当前任务
        for sc_id, sc in self.spacecrafts.items():
            task_id = sc.current_coalition_id
            
            # 跳过虚拟任务或不存在的任务
            if task_id == 0 or task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            
            # 进行多次观测
            for _ in range(self.observations_per_round):
                # 以一定概率正确观测
                if random.random() < sc.positive_observation_prob:
                    observed_type = task.true_type
                else:
                    # 随机选择一个错误类型
                    possible_false_types = [k for k in range(num_types) if k != task.true_type]
                    if not possible_false_types:
                        observed_type = task.true_type
                    else:
                        observed_type = random.choice(possible_false_types)
                
                if 0 <= observed_type < num_types:
                    aggregated_observations[task_id, observed_type] += 1
        
        return aggregated_observations
    
    def update_beliefs(self, aggregated_observations):
        """
        更新航天器信念
        
        参数:
            aggregated_observations: 聚合的观测结果
        """
        for sc_id, sc in self.spacecrafts.items():
            # 更新观测矩阵
            sc.observation_matrix += aggregated_observations
            
            # 更新每个任务的信念
            for j in range(1, sc.num_tasks + 1):
                # 获取当前累积的计数
                current_counts = sc.observation_matrix[j, :]
                
                # 添加伪计数
                alpha = current_counts + 1.0  # 伪计数
                
                # 计算新的信念
                sum_alpha = np.sum(alpha)
                
                if sum_alpha > 1e-9:
                    sc.local_belief[j, :] = alpha / sum_alpha
                else:
                    # 如果没有观测，保持均匀分布
                    sc.local_belief[j, :] = 1.0 / sc.num_types
    
    def step(self, actions):
        """
        执行一个仿真步骤
        
        参数:
            actions: 航天器动作字典
            
        返回:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 更新航天器状态
        for sc_id, action in actions.items():
            if sc_id in self.spacecrafts:
                self.spacecrafts[sc_id].update_physical_state(action, self.dt)
        
        # 更新时间
        self.time += self.dt
        
        # 更新通信图
        self.update_comm_graph()
        
        # 计算当前全局效用
        current_utility = self.calculate_global_utility()
        self.utility_history.append(current_utility)
        
        # 记录状态
        self.record_state()
        
        # 返回结果
        next_state = self._get_current_state()
        reward = current_utility
        done = False
        info = {'time': self.time}
        
        return next_state, reward, done, info
    
    def _get_current_state(self):
        """获取当前状态"""
        state = {
            'spacecrafts': {},
            'tasks': {},
            'time': self.time
        }
        
        for sc_id, sc in self.spacecrafts.items():
            state['spacecrafts'][sc_id] = {
                'position': sc.position,
                'relative_state': sc.relative_state,
                'current_coalition_id': sc.current_coalition_id,
                'neighbors': sc.neighbors
            }
        
        for task_id, task in self.tasks.items():
            state['tasks'][task_id] = {
                'position': task.position,
                'true_type': task.true_type,
                'is_completed': task.is_completed
            }
        
        return state
    
    def run_belief_update(self):
        """执行信念更新"""
        # 获取观测结果
        aggregated_observations = self.take_observations()
        
        # 更新信念
        self.update_beliefs(aggregated_observations)
        
        return aggregated_observations
    
    def initialize_beliefs(self, belief_type='uniform'):
        """
        初始化航天器信念
        
        参数:
            belief_type: 信念初始化类型 ('uniform' 或 'arbitrary')
        """
        for sc_id, sc in self.spacecrafts.items():
            # 初始化为均匀分布
            sc.local_belief = np.zeros((sc.num_tasks + 1, sc.num_types))
            
            if belief_type == 'uniform':
                # 均匀分布信念
                sc.local_belief[1:, :] = 1.0 / sc.num_types
            elif belief_type == 'arbitrary':
                # 随机信念
                for j in range(1, sc.num_tasks + 1):
                    random_vector = np.random.rand(sc.num_types) + 1e-6
                    sc.local_belief[j, :] = random_vector / np.sum(random_vector)
            else:
                # 默认均匀分布
                sc.local_belief[1:, :] = 1.0 / sc.num_types
            
            # 虚拟任务的信念设为0
            sc.local_belief[0, :] = 0
            
            # 初始化观测矩阵
            sc.observation_matrix = np.zeros((sc.num_tasks + 1, sc.num_types))