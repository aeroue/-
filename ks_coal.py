#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import random
import time
import networkx as nx

class KSCOAL:
    """K-Serial Stable Coalition Algorithm (KS-COAL)"""
    
    def __init__(self, num_robots, num_tasks, K=None, max_iterations=100, convergence_threshold=3):
        """
        初始化KS-COAL算法
        
        参数:
            num_robots: 航天器数量
            num_tasks: 任务数量 (不包括虚拟任务t_0)
            K: 优化指数列表，若为None则默认设为[2]*num_robots
            max_iterations: 最大迭代次数
            convergence_threshold: 稳定迭代次数阈值
        """
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.total_tasks = num_tasks + 1  # 包括虚拟任务t_0
        
        # 设置优化指数
        if K is None:
            self.K = [2] * num_robots
        else:
            self.K = K
            
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # 初始化局部联盟结构
        self.local_partitions = {}
        self.partition_versions = {}
        self.partition_timestamps = {}
        self.current_coalitions = {}
        self.needs_update = {}
        
        # 初始化通信拓扑
        self.comm_graph = nx.Graph()
        
        # 初始化消息队列
        self.message_queues = {}
        
        # 记录迭代历史
        self.coalition_history = []
        self.utility_history = []
        
    def initialize(self, initial_positions, comm_range=150.0):
        """
        初始化算法
        
        参数:
            initial_positions: 航天器初始位置
            comm_range: 通信范围
        """
        # 初始化通信图
        self.comm_graph.add_nodes_from(range(self.num_robots))
        
        # 基于距离添加通信边
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                dist = np.linalg.norm(np.array(initial_positions[i][:2]) - np.array(initial_positions[j][:2]))
                if dist <= comm_range:
                    self.comm_graph.add_edge(i, j)
        
        # 确保通信图至少是连通的
        if not nx.is_connected(self.comm_graph) and self.num_robots > 1:
            components = list(nx.connected_components(self.comm_graph))
            for i in range(len(components)-1):
                comp1 = list(components[i])[0]
                comp2 = list(components[i+1])[0]
                self.comm_graph.add_edge(comp1, comp2)
        
        # 初始化每个航天器的局部变量
        for i in range(self.num_robots):
            # 初始联盟结构：所有航天器都在虚拟任务t_0中
            partition = {0: list(range(self.num_robots))}
            for j in range(1, self.total_tasks):
                partition[j] = []
            
            self.local_partitions[i] = partition
            self.partition_versions[i] = 0
            self.partition_timestamps[i] = random.random()
            self.current_coalitions[i] = 0  # 初始时所有航天器都在虚拟任务t_0中
            self.needs_update[i] = True
            self.message_queues[i] = []
        
        # 记录初始联盟结构
        self.coalition_history.append(self.get_assignment_dict())
    
    def get_neighbors(self, robot_id):
        """获取航天器的邻居列表"""
        return list(self.comm_graph.neighbors(robot_id))
    
    def calculate_expected_utility(self, robot_id, task_id, local_partition, tasks, robots):
        """
        计算航天器加入任务联盟的期望效用
        
        参数:
            robot_id: 航天器ID
            task_id: 任务ID
            local_partition: 局部联盟结构
            tasks: 任务字典
            robots: 航天器字典
            
        返回:
            utility: 期望效用
        """
        if task_id == 0:  # 虚拟任务的效用为0
            return 0.0
        
        # 获取当前任务联盟
        coalition = local_partition.get(task_id, []).copy()
        is_member = robot_id in coalition
        
        # 计算联盟规模
        if is_member:
            coalition_size = len(coalition)
        else:
            coalition_size = len(coalition) + 1
        
        # 防止除以零
        if coalition_size == 0:
            coalition_size = 1
        
        # 获取任务对象
        task = tasks.get(task_id)
        if task is None:
            return -float('inf')
        
        # 获取航天器对象
        robot = robots.get(robot_id)
        if robot is None:
            return -float('inf')
        
        # 计算预期收益
        expected_revenue_term = 0.0
        expected_risk_cost_term = 0.0
        
        for k in range(robot.num_types):
            belief_ijk = robot.local_belief[task_id, k]
            revenue_k = task.get_revenue(k)
            risk_cost_ik = task.calculate_risk_cost(robot, k)
            
            expected_revenue_term += belief_ijk * revenue_k
            expected_risk_cost_term += belief_ijk * risk_cost_ik
        
        expected_shared_revenue = expected_revenue_term / coalition_size
        
        # 计算燃料成本
        estimated_delta_v = robot.estimate_delta_v_for_task(task)
        fuel_cost = robot.calculate_fuel_cost(estimated_delta_v)
        
        if np.isinf(fuel_cost):
            return -float('inf')
        
        # 计算总效用
        utility = expected_shared_revenue - expected_risk_cost_term - fuel_cost
        
        return utility
    
    def greedy_selection(self, robot_id, tasks, robots):
        """
        航天器的贪婪选择阶段
        
        参数:
            robot_id: 航天器ID
            tasks: 任务字典
            robots: 航天器字典
            
        返回:
            changed: 是否改变了联盟
        """
        local_partition = self.local_partitions[robot_id]
        current_coalition_id = self.current_coalitions[robot_id]
        
        # 计算当前联盟的效用
        current_utility = self.calculate_expected_utility(
            robot_id, current_coalition_id, local_partition, tasks, robots
        )
        
        best_utility = current_utility if not np.isinf(current_utility) else -float('inf')
        best_task_id = current_coalition_id
        changed = False
        
        # 尝试所有可能的任务
        for task_id in range(self.total_tasks):
            utility = self.calculate_expected_utility(
                robot_id, task_id, local_partition, tasks, robots
            )
            
            if not np.isinf(utility) and utility > best_utility + 1e-9:
                best_utility = utility
                best_task_id = task_id
        
        # 如果找到了更好的任务，则转换联盟
        if best_task_id != current_coalition_id:
            old_coalition_id = current_coalition_id
            
            # 从当前联盟中移除
            if robot_id in local_partition.get(old_coalition_id, []):
                local_partition[old_coalition_id].remove(robot_id)
            
            # 加入新联盟
            if best_task_id not in local_partition:
                local_partition[best_task_id] = []
            
            if robot_id not in local_partition[best_task_id]:
                local_partition[best_task_id].append(robot_id)
            
            # 更新状态
            self.current_coalitions[robot_id] = best_task_id
            self.partition_versions[robot_id] += 1
            self.partition_timestamps[robot_id] = random.random()
            changed = True
            self.needs_update[robot_id] = True
        
        return changed
    
    def send_message(self, robot_id):
        """
        发送消息给邻居
        
        参数:
            robot_id: 航天器ID
        """
        partition_copy = copy.deepcopy(self.local_partitions[robot_id])
        version = self.partition_versions[robot_id]
        timestamp = self.partition_timestamps[robot_id]
        
        message = {
            'sender_id': robot_id,
            'partition': partition_copy,
            'version': version,
            'timestamp': timestamp
        }
        
        # 向所有邻居发送消息
        for neighbor_id in self.get_neighbors(robot_id):
            self.message_queues[neighbor_id].append(message)
    
    def process_messages(self, robot_id):
        """
        处理接收到的消息
        
        参数:
            robot_id: 航天器ID
            
        返回:
            changed_by_message: 是否因消息而改变了局部视图
        """
        if not self.message_queues[robot_id]:
            return False
        
        # 构建自身消息
        own_info = {
            'sender_id': robot_id,
            'partition': copy.deepcopy(self.local_partitions[robot_id]),
            'version': self.partition_versions[robot_id],
            'timestamp': self.partition_timestamps[robot_id]
        }
        
        # 收集所有消息
        received_info = [own_info] + [copy.deepcopy(msg) for msg in self.message_queues[robot_id]]
        self.message_queues[robot_id] = []
        
        # 找到主导消息
        dominant_info = own_info
        
        for msg in received_info:
            is_dominant = False
            
            # 比较版本号和时间戳
            if msg['version'] > dominant_info['version']:
                is_dominant = True
            elif msg['version'] == dominant_info['version'] and msg['timestamp'] > dominant_info['timestamp']:
                is_dominant = True
            
            if is_dominant:
                dominant_info = msg
        
        # 如果主导消息不是自己的，则更新局部视图
        changed_by_message = False
        
        if dominant_info['sender_id'] != robot_id:
            self.local_partitions[robot_id] = dominant_info['partition']
            self.partition_versions[robot_id] = dominant_info['version']
            self.partition_timestamps[robot_id] = dominant_info['timestamp']
            
            # 找到自己在新联盟结构中的位置
            found_self = False
            
            for task_id, members in self.local_partitions[robot_id].items():
                if robot_id in members:
                    self.current_coalitions[robot_id] = task_id
                    found_self = True
                    break
            
            # 如果在新联盟结构中没有找到自己，则分配到虚拟任务t_0
            if not found_self:
                self.current_coalitions[robot_id] = 0
                
                if 0 not in self.local_partitions[robot_id]:
                    self.local_partitions[robot_id][0] = []
                
                if robot_id not in self.local_partitions[robot_id][0]:
                    self.local_partitions[robot_id][0].append(robot_id)
            
            changed_by_message = True
            self.needs_update[robot_id] = True
        
        return changed_by_message
    
    def run_iteration(self, tasks, robots):
        """
        运行一次迭代
        
        参数:
            tasks: 任务字典
            robots: 航天器字典
            
        返回:
            stable: 是否达到稳定
        """
        # 记录上一次迭代的联盟结构
        prev_partition = self.get_assignment_dict()
        
        # 记录多少航天器改变了联盟
        num_changed = 0
        
        # 所有航天器执行贪婪选择并发送消息
        for robot_id in range(self.num_robots):
            if self.greedy_selection(robot_id, tasks, robots):
                num_changed += 1
                self.send_message(robot_id)
        
        # 所有航天器处理接收到的消息
        for robot_id in range(self.num_robots):
            if self.process_messages(robot_id):
                num_changed += 1
        
        # 记录当前联盟结构
        curr_partition = self.get_assignment_dict()
        self.coalition_history.append(curr_partition)
        
        # 计算当前全局效用
        global_utility = self.calculate_global_utility(tasks, robots)
        self.utility_history.append(global_utility)
        
        # 检查是否达到稳定
        stable = (num_changed == 0)
        
        return stable
    
    def run(self, tasks, robots):
        """
        执行KS-COAL算法
        
        参数:
            tasks: 任务字典
            robots: 航天器字典
            
        返回:
            final_partition: 最终联盟结构
            global_utility: 全局效用
            iterations: 迭代次数
        """
        start_time = time.time()
        
        stable_iterations = 0
        iterations = 0
        
        print("开始执行KS-COAL算法...")
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # 执行一次迭代
            stable = self.run_iteration(tasks, robots)
            
            # 如果达到稳定，增加稳定计数
            if stable:
                stable_iterations += 1
                print(f"迭代 {iterations}: 联盟结构稳定，稳定计数 {stable_iterations}/{self.convergence_threshold}")
            else:
                stable_iterations = 0
                print(f"迭代 {iterations}: 联盟结构变化，稳定计数重置为0")
            
            # 如果连续稳定次数达到阈值，认为收敛
            if stable_iterations >= self.convergence_threshold:
                print(f"KS-COAL算法在{iterations}次迭代后收敛")
                break
        
        if iterations >= self.max_iterations:
            print(f"KS-COAL算法达到最大迭代次数 ({self.max_iterations})，未收敛")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 获取最终联盟结构和全局效用
        final_partition = self.get_assignment_dict()
        global_utility = self.calculate_global_utility(tasks, robots)
        
        print(f"KS-COAL执行时间: {execution_time:.2f}秒")
        print(f"最终全局效用: {global_utility:.2f}")
        
        return final_partition, global_utility, iterations
    
    def get_assignment_dict(self):
        """获取当前的任务分配字典"""
        assignment = {}
        
        for robot_id in range(self.num_robots):
            assignment[robot_id] = self.current_coalitions[robot_id]
        
        return assignment
    
    def calculate_global_utility(self, tasks, robots):
        """
        计算全局效用
        
        参数:
            tasks: 任务字典
            robots: 航天器字典
            
        返回:
            global_utility: 全局效用
        """
        assignment = self.get_assignment_dict()
        global_utility = 0.0
        
        for robot_id, task_id in assignment.items():
            robot = robots.get(robot_id)
            
            if robot is not None:
                # 计算航天器的实际效用贡献
                utility = robot.calculate_actual_utility(task_id, assignment, tasks)
                global_utility += utility
        
        return global_utility