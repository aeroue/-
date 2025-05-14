#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow
import matplotlib.font_manager as fm
import matplotlib
import os

# 设置全局字体支持中文
def setup_chinese_font():
    """设置支持中文的字体"""
    # 检测是否为Windows系统
    if os.name == 'nt':
        # Windows常见的中文字体
        font_names = ['SimHei', 'Microsoft YaHei', 'SimSun']
    else:
        # Linux/Mac常见的中文字体
        font_names = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Heiti SC', 'STHeiti']
    
    # 尝试设置中文字体
    font_found = False
    for font_name in font_names:
        try:
            matplotlib.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"成功设置中文字体: {font_name}")
            font_found = True
            break
        except:
            continue
    
    if not font_found:
        print("警告: 未能找到支持中文的字体，中文可能无法正确显示")

def calculate_distance(pos1, pos2):
    """
    计算两点之间的欧几里得距离
    
    参数:
        pos1: 第一个位置 [x, y]
        pos2: 第二个位置 [x, y]
        
    返回:
        distance: 距离
    """
    if pos1 is None or pos2 is None:
        return float('inf')
    
    try:
        # 确保坐标是有效的数值
        x1, y1 = pos1[:2]  # 取前两个元素，适应3D坐标
        x2, y2 = pos2[:2]
        
        if not all(isinstance(coord, (int, float)) and not np.isnan(coord) and not np.isinf(coord) 
                  for coord in [x1, y1, x2, y2]):
            return float('inf')
        
        # 计算距离平方
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
        
        if dist_sq < 0:
            return float('inf')
        
        return np.sqrt(dist_sq)
    
    except (TypeError, IndexError, ValueError) as e:
        return float('inf')

def get_neighbors(positions, robot_id, comm_range, num_robots):
    """
    获取航天器的邻居列表
    
    参数:
        positions: 所有航天器的位置
        robot_id: 航天器ID
        comm_range: 通信范围
        num_robots: 航天器总数
        
    返回:
        neighbors: 邻居列表
    """
    neighbors = []
    robot_pos = positions[robot_id]
    
    for other_id in range(num_robots):
        if other_id == robot_id:
            continue
        
        other_pos = positions[other_id]
        dist = calculate_distance(robot_pos, other_pos)
        
        if dist <= comm_range:
            neighbors.append(other_id)
    
    return neighbors

def generate_spread_out_goals(num_goals, env_size, min_distance):
    """
    生成分散的任务位置
    
    参数:
        num_goals: 任务数量
        env_size: 环境大小
        min_distance: 最小距离
        
    返回:
        goals: 任务位置列表
    """
    goals = []
    attempts = 0
    max_attempts = 1000
    
    while len(goals) < num_goals and attempts < max_attempts:
        new_goal = np.random.random(2) * env_size
        
        # 检查与已有任务的距离
        valid = True
        for goal in goals:
            if calculate_distance(new_goal, goal) < min_distance:
                valid = False
                break
        
        if valid:
            goals.append(new_goal)
        
        attempts += 1
    
    if len(goals) < num_goals:
        print(f"警告: 只能生成 {len(goals)}/{num_goals} 个分散的任务")
    
    return np.array(goals)

def identify_rendezvous_configs(num_goals, num_robots):
    """
    识别集合点配置
    
    参数:
        num_goals: 任务数量
        num_robots: 航天器数量
        
    返回:
        configs: 有效配置列表
    """
    import itertools
    
    # 简单情况：每个航天器分配一个任务
    if num_robots <= num_goals:
        return list(itertools.permutations(range(num_goals), num_robots))
    
    # 复杂情况：多个航天器可能分配到同一任务
    configs = []
    
    # 枚举所有可能的分配
    for config in itertools.product(range(num_goals), repeat=num_robots):
        # 确保每个任务至少有一个航天器
        if set(config) == set(range(num_goals)):
            configs.append(config)
    
    return configs

def check_convergence(positions, goals, convergence_type, max_distance=1.0):
    """
    检查航天器是否收敛到目标
    
    参数:
        positions: 航天器位置
        goals: 任务位置
        convergence_type: 收敛类型 ('converge' 或 'exclusive')
        max_distance: 最大距离阈值
        
    返回:
        check: 是否收敛
        selected_goals: 选择的任务
    """
    # 计算每个航天器到每个任务的距离
    distances_to_goals = [np.linalg.norm(goals - pos[:2], axis=1) for pos in positions]
    
    # 每个航天器选择的最近任务
    distances_to_selected_goal = [np.min(distances) for distances in distances_to_goals]
    selected_goals = [np.argmin(distances) for distances in distances_to_goals]
    
    if convergence_type == 'converge':
        # 检查是否所有航天器都收敛到同一个任务
        all_same_goal = [selected_goals[0] == which_goal for which_goal in selected_goals]
        check = (np.array(distances_to_selected_goal) < max_distance).all() and all(all_same_goal)
        return check, selected_goals
    elif convergence_type == 'exclusive':
        # 检查是否所有航天器都收敛到不同的任务
        all_different_goals = len(selected_goals) == len(set(selected_goals))
        check = (np.array(distances_to_selected_goal) < max_distance).all() and all_different_goals
        return check, selected_goals
    else:
        raise ValueError(f"未知的收敛类型: {convergence_type}")