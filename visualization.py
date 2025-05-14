#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import os
from utils import setup_chinese_font

class Visualizer:
    """可视化模块"""
    
    def __init__(self, env_size=100, padding=5, dpi=100, support_chinese=True):
        """
        初始化可视化模块
        
        参数:
            env_size: 环境大小
            padding: 边距
            dpi: 分辨率
            support_chinese: 是否支持中文
        """
        self.env_size = env_size
        self.padding = padding
        self.dpi = dpi
        
        # 设置中文支持
        if support_chinese:
            setup_chinese_font()
    
    def plot_comm_graph(self, positions, comm_graph, title="通信拓扑"):
        """
        绘制通信拓扑
        
        参数:
            positions: 航天器位置
            comm_graph: 通信图
            title: 图表标题
        """
        plt.figure(figsize=(10, 10), dpi=self.dpi)
        plt.xlim(-self.padding, self.env_size + self.padding)
        plt.ylim(-self.padding, self.env_size + self.padding)
        
        # 绘制节点
        pos = {i: positions[i][:2] for i in range(len(positions))}
        nx.draw(comm_graph, pos, with_labels=True, node_color='skyblue', 
                node_size=500, font_size=12, font_weight='bold')
        
        plt.title(title, fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_coalition_formation(self, initial_positions, final_positions, 
                                 tasks, initial_comm_graph, final_partition, 
                                 title="联盟形成结果"):
        """
        绘制联盟形成结果
        
        参数:
            initial_positions: 初始位置
            final_positions: 最终位置
            tasks: 任务字典
            initial_comm_graph: 初始通信图
            final_partition: 最终联盟分配
            title: 图表标题
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=self.dpi)
        
        # 设置坐标轴范围
        ax1.set_xlim(-self.padding, self.env_size + self.padding)
        ax1.set_ylim(-self.padding, self.env_size + self.padding)
        ax2.set_xlim(-self.padding, self.env_size + self.padding)
        ax2.set_ylim(-self.padding, self.env_size + self.padding)
        
        # 绘制初始状态
        ax1.set_title("初始状态与通信拓扑", fontsize=16)
        
        # 绘制航天器
        for i, pos in enumerate(initial_positions):
            ax1.plot(pos[0], pos[1], 'o', markersize=10, color='blue', alpha=0.7)
            ax1.text(pos[0] + 2, pos[1] + 2, f"SC{i}", fontsize=10)
        
        # 绘制任务
        for j, task in tasks.items():
            if j == 0:  # 跳过虚拟任务
                continue
            
            pos = task.position
            ax1.plot(pos[0], pos[1], '*', markersize=15, color='red')
            ax1.text(pos[0] + 2, pos[1] + 2, f"T{j}", fontsize=12)
        
        # 绘制通信边
        for edge in initial_comm_graph.edges():
            i, j = edge
            ax1.plot([initial_positions[i][0], initial_positions[j][0]], 
                     [initial_positions[i][1], initial_positions[j][1]], 
                     '--', color='grey', alpha=0.5)
        
        # 绘制最终状态
        ax2.set_title("最终联盟形成结果", fontsize=16)
        
        # 为不同联盟分配不同颜色
        coalition_colors = {0: 'grey'}  # 虚拟任务为灰色
        cmap = plt.cm.get_cmap('tab10', len(tasks))
        
        for j in range(1, len(tasks) + 1):
            if j in tasks:
                coalition_colors[j] = cmap(j - 1)
        
        # 绘制任务
        for j, task in tasks.items():
            if j == 0:  # 跳过虚拟任务
                continue
            
            pos = task.position
            ax2.plot(pos[0], pos[1], '*', markersize=15, color=coalition_colors[j])
            ax2.text(pos[0] + 2, pos[1] + 2, f"T{j}", fontsize=12)
        
        # 绘制航天器并连接到分配的任务
        for i, pos in enumerate(final_positions):
            task_id = final_partition.get(i, 0)
            ax2.plot(pos[0], pos[1], 'o', markersize=10, color=coalition_colors[task_id], alpha=0.7)
            ax2.text(pos[0] + 2, pos[1] + 2, f"SC{i}", fontsize=10)
            
            # 如果分配了实际任务，绘制连接线
            if task_id > 0 and task_id in tasks:
                task_pos = tasks[task_id].position
                ax2.plot([pos[0], task_pos[0]], [pos[1], task_pos[1]], 
                         '-', color=coalition_colors[task_id], alpha=0.5)
        
        # 添加图例
        ax2.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=coalition_colors[j], 
                       markersize=10, label=f"联盟 T{j}")
            for j in sorted(coalition_colors.keys()) if j > 0
        ], loc='upper right')
        
        # 显示网格
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(title, fontsize=18)
        plt.tight_layout()
        
        return fig
    
    def plot_belief_evolution(self, belief_history, num_tasks, num_types, true_types, 
                              task_type_map, title="信念演化"):
        """
        绘制信念演化
        
        参数:
            belief_history: 信念历史
            num_tasks: 任务数量
            num_types: 任务类型数量
            true_types: 真实任务类型
            task_type_map: 任务类型映射
            title: 图表标题
        """
        # 确定子图数量
        n_cols = min(3, num_tasks)
        n_rows = (num_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), 
                                dpi=self.dpi, squeeze=False)
        axes = axes.flatten()
        
        # 颜色映射
        cmap = plt.cm.get_cmap('tab10', num_types)
        colors = [cmap(i) for i in range(num_types)]
        
        # 绘制每个任务的信念演化
        for task_id in range(1, num_tasks + 1):
            ax_idx = task_id - 1
            ax = axes[ax_idx]
            
            if task_id not in belief_history or not belief_history[task_id]:
                ax.set_title(f"任务 T{task_id}: 无信念历史")
                continue
            
            belief_array = np.array(belief_history[task_id])
            if belief_array.shape[0] == 0:
                continue
            
            # 获取迭代次数
            n_iterations = belief_array.shape[0]
            iterations = range(n_iterations)
            
            # 绘制每种类型的信念概率
            for type_k in range(num_types):
                type_name = task_type_map.get(type_k, f"未知类型{type_k}")
                ax.plot(iterations, belief_array[:, type_k], marker='.', markersize=3, 
                        linestyle='-', label=f"{type_name}", color=colors[type_k])
            
            # 标记真实类型
            true_type = true_types[task_id - 1] if task_id - 1 < len(true_types) else None
            if true_type is not None:
                true_type_name = task_type_map.get(true_type, f"未知类型{true_type}")
                ax.set_title(f"任务 T{task_id} 信念演化 (真实: {true_type_name})")
            else:
                ax.set_title(f"任务 T{task_id} 信念演化")
            
            # 设置辅助线
            ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5)
            ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.5)
            
            # 设置坐标轴范围
            ax.set_ylim(-0.1, 1.1)
            
            # 设置标签
            if ax_idx // n_cols == n_rows - 1:
                ax.set_xlabel('迭代次数')
            if ax_idx % n_cols == 0:
                ax.set_ylabel('信念概率')
            
            # 添加图例
            ax.legend(fontsize=8)
            
            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 隐藏多余的子图
        for i in range(num_tasks, n_rows * n_cols):
            if i < len(axes):
                fig.delaxes(axes[i])
        
        plt.suptitle(title, fontsize=18)
        plt.tight_layout()
        
        return fig
    
    def plot_utility_evolution(self, utility_history, title="全局效用演化"):
        """
        绘制全局效用演化
        
        参数:
            utility_history: 效用历史
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # 获取迭代次数
        iterations = range(len(utility_history))
        
        # 绘制效用曲线
        ax.plot(iterations, utility_history, 'o-', color='blue', markersize=4)
        
        # 标记最大效用
        max_utility = max(utility_history)
        max_idx = utility_history.index(max_utility)
        ax.plot(max_idx, max_utility, 'ro', markersize=8)
        ax.text(max_idx + 0.5, max_utility, f"最大: {max_utility:.2f}", fontsize=12)
        
        # 设置标签
        ax.set_xlabel('迭代次数', fontsize=14)
        ax.set_ylabel('全局效用', fontsize=14)
        
        # 设置标题
        ax.set_title(title, fontsize=16)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig
    
    def plot_gantt_chart(self, assignment_history, num_robots, num_tasks, 
                         title="任务分配甘特图"):
        """
        绘制任务分配甘特图
        
        参数:
            assignment_history: 分配历史
            num_robots: 航天器数量
            num_tasks: 任务数量
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(max(10, len(assignment_history) * 0.4), 
                                        max(6, num_robots * 0.5)), 
                              dpi=self.dpi)
        
        # 设置坐标轴
        ax.set_xlabel('迭代次数', fontsize=14)
        ax.set_ylabel('航天器ID', fontsize=14)
        
        # Y轴刻度
        ax.set_yticks(range(num_robots))
        ax.set_yticklabels([f"SC{i}" for i in range(num_robots)])
        ax.invert_yaxis()  # 让SC0在顶部
        
        # X轴刻度
        ax.set_xticks(range(len(assignment_history)))
        ax.set_xlim(-0.5, len(assignment_history) - 0.5)
        
        # 为不同任务分配不同颜色
        cmap = plt.cm.get_cmap('tab10', num_tasks + 1)
        task_colors = {0: 'lightgrey'}  # 虚拟任务为灰色
        
        for j in range(1, num_tasks + 1):
            task_colors[j] = cmap(j)
        
        # 绘制甘特图
        for round_idx, assignments in enumerate(assignment_history):
            for sc_id, task_id in assignments.items():
                if sc_id >= num_robots:
                    continue
                
                color = task_colors.get(task_id, 'white')
                rect = ax.barh(y=sc_id, width=1, left=round_idx - 0.5, height=0.6, 
                              color=color, edgecolor='black', linewidth=0.5)
                
                # 在条形中添加任务标签
                if task_id != 0:
                    # 计算文本颜色（深色背景用白色文本，浅色背景用黑色文本）
                    rgb_color = plt.cm.colors.to_rgb(color)
                    text_color = 'white' if sum(rgb_color) / 3 < 0.5 else 'black'
                    
                    ax.text(round_idx, sc_id, f"T{task_id}", 
                           ha='center', va='center', fontsize=8, color=text_color)
        
        # 添加图例
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color=task_colors[j], 
                         label=f"任务 T{j}" if j > 0 else "虚拟任务")
            for j in sorted(task_colors.keys())
        ], loc='upper right')
        
        # 添加网格
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 设置标题
        ax.set_title(title, fontsize=16)
        
        plt.tight_layout()
        
        return fig
    
    def animate_trajectory(self, positions_history, goals, robot_types=None, 
                          final_assignment=None, interval=200, 
                          title="航天器轨迹动画"):
        """
        创建航天器轨迹动画
        
        参数:
            positions_history: 位置历史
            goals: 任务位置
            robot_types: 航天器类型
            final_assignment: 最终分配
            interval: 动画间隔
            title: 动画标题
            
        返回:
            ani: 动画对象
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)
        
        # 设置坐标轴范围
        ax.set_xlim(-self.padding, self.env_size + self.padding)
        ax.set_ylim(-self.padding, self.env_size + self.padding)
        
        # 获取航天器和步数信息
        num_robots = len(positions_history[0])
        num_steps = len(positions_history)
        
        # 颜色映射
        if robot_types is None:
            robot_types = ['A'] * num_robots
        
        type_colors = {'A': 'blue', 'B': 'green', 'S': 'orange'}
        robot_colors = [type_colors.get(robot_type, 'gray') for robot_type in robot_types]
        
        # 任务颜色
        num_goals = len(goals)
        goal_cmap = plt.cm.get_cmap('tab10', num_goals)
        goal_colors = [goal_cmap(i) for i in range(num_goals)]
        
        # 初始化轨迹线和标记
        robot_markers = []
        robot_trails = []
        robot_labels = []
        goal_markers = []
        goal_labels = []
        
        # 绘制目标位置
        for i, goal in enumerate(goals):
            marker = ax.plot(goal[0], goal[1], '*', markersize=15, color=goal_colors[i])[0]
            label = ax.text(goal[0] + 2, goal[1] + 2, f"目标{i}", fontsize=10)
            goal_markers.append(marker)
            goal_labels.append(label)
        
        # 绘制航天器初始位置
        for i in range(num_robots):
            # 轨迹线
            line, = ax.plot([], [], '-', alpha=0.5, color=robot_colors[i], linewidth=1)
            robot_trails.append(line)
            
            # 航天器标记
            marker, = ax.plot([], [], 'o', markersize=8, color=robot_colors[i])
            robot_markers.append(marker)
            
            # 航天器标签
            label = ax.text(0, 0, f"SC{i}", fontsize=8)
            robot_labels.append(label)
        
        # 添加时间标签
        time_label = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          verticalalignment='top')
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置标题
        ax.set_title(title)
        
        # 更新函数
        def update(frame):
            for i in range(num_robots):
                # 提取历史位置数据
                x_history = [positions_history[j][i][0] for j in range(frame + 1)]
                y_history = [positions_history[j][i][1] for j in range(frame + 1)]
                
                # 更新轨迹线
                robot_trails[i].set_data(x_history, y_history)
                
                # 更新航天器位置
                x_current = positions_history[frame][i][0]
                y_current = positions_history[frame][i][1]
                robot_markers[i].set_data(x_current, y_current)
                
                # 更新标签位置
                robot_labels[i].set_position((x_current + 1, y_current + 1))
            
            # 更新时间标签
            time_label.set_text(f'步骤: {frame}/{num_steps-1}')
            
            # 返回更新的对象
            return robot_trails + robot_markers + robot_labels + [time_label]
        
        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=num_steps,
                                   interval=interval, blit=True)
        
        return ani
    
    def plot_trajectories(self, positions_history, goals, robot_types=None, 
                         final_assignment=None, title="航天器轨迹"):
        """
        绘制航天器轨迹
        
        参数:
            positions_history: 位置历史
            goals: 任务位置
            robot_types: 航天器类型
            final_assignment: 最终分配
            title: 图表标题
            
        返回:
            fig: 图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)
        
        # 设置坐标轴范围
        ax.set_xlim(-self.padding, self.env_size + self.padding)
        ax.set_ylim(-self.padding, self.env_size + self.padding)
        
        # 获取航天器和步数信息
        num_robots = len(positions_history[0])
        num_steps = len(positions_history)
        
        # 颜色映射
        if robot_types is None:
            robot_types = ['A'] * num_robots
        
        type_colors = {'A': 'blue', 'B': 'green', 'S': 'orange'}
        robot_colors = [type_colors.get(robot_type, 'gray') for robot_type in robot_types]
        
        # 任务颜色
        num_goals = len(goals)
        goal_cmap = plt.cm.get_cmap('tab10', num_goals)
        goal_colors = [goal_cmap(i) for i in range(num_goals)]
        
        # 绘制目标位置
        for i, goal in enumerate(goals):
            ax.plot(goal[0], goal[1], '*', markersize=15, color=goal_colors[i])
            ax.text(goal[0] + 2, goal[1] + 2, f"目标{i}", fontsize=10)
        
        # 绘制航天器轨迹
        for i in range(num_robots):
            # 提取历史位置数据
            x_history = [positions_history[j][i][0] for j in range(num_steps)]
            y_history = [positions_history[j][i][1] for j in range(num_steps)]
            
            # 绘制轨迹线
            ax.plot(x_history, y_history, '-', alpha=0.5, color=robot_colors[i], linewidth=1)
            
            # 绘制起点
            ax.plot(x_history[0], y_history[0], 'o', markersize=8, color=robot_colors[i])
            ax.text(x_history[0] + 1, y_history[0] + 1, f"SC{i}起点", fontsize=8)
            
            # 绘制终点
            ax.plot(x_history[-1], y_history[-1], 's', markersize=8, color=robot_colors[i])
            ax.text(x_history[-1] + 1, y_history[-1] + 1, f"SC{i}终点", fontsize=8)
            
            # 如果有最终分配，绘制航天器到目标的连接线
            if final_assignment is not None and i in final_assignment:
                goal_id = final_assignment[i]
                if goal_id < len(goals) and goal_id >= 0:
                    ax.plot([x_history[-1], goals[goal_id][0]], 
                           [y_history[-1], goals[goal_id][1]], 
                           '--', color=robot_colors[i], alpha=0.7)
        
        # 添加图例
        robot_legend = [plt.Line2D([0], [0], color=type_colors[t], marker='o', 
                                  linestyle='-', markersize=8, 
                                  label=f"{t}型航天器")
                       for t in set(robot_types)]
        
        goal_legend = [plt.Line2D([0], [0], color=goal_colors[i], marker='*', 
                                 linestyle='', markersize=10, 
                                 label=f"目标{i}")
                      for i in range(num_goals)]
        
        ax.legend(handles=robot_legend + goal_legend, loc='upper right')
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置标题
        ax.set_title(title, fontsize=16)
        
        plt.tight_layout()
        
        return fig
    
    def plot_free_energy(self, free_energy_history, title="自由能变化"):
        """
        绘制自由能变化
        
        参数:
            free_energy_history: 自由能历史
            title: 图表标题
            
        返回:
            fig: 图表对象
        """
        # 分离航天器ID和自由能值
        robot_ids = [entry[0] for entry in free_energy_history]
        energy_values = [entry[1] for entry in free_energy_history]
        
        # 获取不同的航天器ID
        unique_ids = sorted(set(robot_ids))
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        # 为每个航天器分组数据
        for robot_id in unique_ids:
            indices = [i for i, id in enumerate(robot_ids) if id == robot_id]
            steps = [i // len(unique_ids) for i in indices]
            values = [energy_values[i] for i in indices]
            
            ax.plot(steps, values, 'o-', markersize=4, label=f"航天器{robot_id}")
        
        # 设置标签和标题
        ax.set_xlabel('决策步骤', fontsize=14)
        ax.set_ylabel('期望自由能', fontsize=14)
        ax.set_title(title, fontsize=16)
        
        # 添加网格和图例
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        return fig
    
    def plot_comparison(self, metrics, methods, metric_name="全局效用", 
                      title="不同方法比较"):
        """
        绘制不同方法的比较
        
        参数:
            metrics: 度量值列表
            methods: 方法名称列表
            metric_name: 度量名称
            title: 图表标题
            
        返回:
            fig: 图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # 绘制条形图
        bars = ax.bar(methods, metrics, width=0.6)
        
        # 为条形图添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=12)
        
        # 设置标签和标题
        ax.set_xlabel('方法', fontsize=14)
        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_title(title, fontsize=16)
        
        # 添加网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 调整Y轴范围，使数据更加明显
        y_min = min(metrics) * 0.9 if min(metrics) > 0 else min(metrics) * 1.1
        y_max = max(metrics) * 1.1
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        return fig