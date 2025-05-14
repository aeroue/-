#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

from dynamics import CWDynamics
from mpc_controller import SpectralMPC
from ks_coal import KSCOAL
from active_inference import ActiveInference
from belief_update import BayesianBeliefUpdate
from utils import setup_chinese_font, generate_spread_out_goals
from visualization import Visualizer
from simulation import Simulation, Task, Spacecraft

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='异构航天器动态联盟组建与协同决策系统')
    
    parser.add_argument('--scenario', type=str, default='strong_comm',
                      choices=['strong_comm', 'weak_comm', 'combined'],
                      help='仿真场景类型')
    
    parser.add_argument('--num_robots', type=int, default=5,
                      help='航天器数量')
    
    parser.add_argument('--num_tasks', type=int, default=3,
                      help='任务数量')
    
    parser.add_argument('--num_types', type=int, default=3,
                      help='任务类型数量')
    
    parser.add_argument('--env_size', type=float, default=100.0,
                      help='环境大小')
    
    parser.add_argument('--comm_range', type=float, default=150.0,
                      help='通信范围')
    
    parser.add_argument('--dt', type=float, default=5.0,
                      help='时间步长')
    
    parser.add_argument('--max_iterations', type=int, default=30,
                      help='最大迭代次数')
    
    parser.add_argument('--save_dir', type=str, default='results',
                      help='结果保存目录')
    
    parser.add_argument('--seed', type=int, default=None,
                      help='随机种子')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建结果保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 初始化CW动力学
    dynamics = CWDynamics(dt=args.dt)
    
    # 初始化仿真环境
    sim = Simulation(env_size=args.env_size, comm_range=args.comm_range, dt=args.dt)
    
    # 初始化任务
    # 为了方便演示，使用固定的任务类型
    task_types = [2, 1, 0]  # 对应Hard、Moderate、Easy
    if args.num_tasks > len(task_types):
        task_types = task_types + [random.randint(0, 2) for _ in range(args.num_tasks - len(task_types))]
    task_types = task_types[:args.num_tasks]
    
    sim.initialize_tasks(args.num_tasks, task_types)
    
    # 初始化航天器
    # 为了演示异构性，使用不同类型的航天器
    sc_types = ['A', 'B', 'S']
    spacecraft_types = []
    for i in range(args.num_robots):
        spacecraft_types.append(sc_types[i % len(sc_types)])
    
    sim.initialize_spacecrafts(args.num_robots, args.num_tasks, args.num_types, 
                              spacecraft_types, dynamics)
    
    # 初始化信念
    sim.initialize_beliefs('uniform')
    
    # 初始化MPC控制器
    Q = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    R = np.diag([1.0, 1.0, 1.0])
    P = np.diag([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
    
    mpc = SpectralMPC(
        dynamics.A_d, dynamics.B_d, Q, R, P, 
        N_horizon=5, N_control=3, umax=0.05, dt=args.dt
    )
    
    # 初始化可视化器
    visualizer = Visualizer(env_size=args.env_size)
    
    # 运行仿真
    print(f"\n===== 开始仿真：{args.scenario} =====")
    start_time = time.time()
    
    if args.scenario == 'strong_comm':
        # 强通信条件下的KS-COAL算法
        print("使用强通信条件下的KS-COAL算法")
        
        # 初始化KS-COAL算法
        ks_coal = KSCOAL(args.num_robots, args.num_tasks, K=None, 
                        max_iterations=args.max_iterations)
        
        # 更新通信图
        sim.update_comm_graph()
        
        # 运行KS-COAL算法
        final_partition, global_utility, iterations = sim.run_ks_coal(ks_coal)
        
        # 更新信念
        aggregated_observations = sim.update_observations()
        sim.update_beliefs(aggregated_observations)
        
        # 保存结果
        result_file = os.path.join(args.save_dir, 'ks_coal_result.txt')
        with open(result_file, 'w') as f:
            f.write(f"全局效用: {global_utility:.2f}\n")
            f.write(f"迭代次数: {iterations}\n")
            f.write(f"最终联盟结构: {final_partition}\n")
        
        # 可视化结果
        # 1. 绘制通信拓扑
        comm_graph_fig = visualizer.plot_comm_graph(
            [sc.position for sc in sim.spacecrafts.values()],
            ks_coal.comm_graph,
            title="通信拓扑"
        )
        comm_graph_fig.savefig(os.path.join(args.save_dir, 'comm_graph.png'))
        
        # 2. 绘制联盟形成结果
        coalition_fig = visualizer.plot_coalition_formation(
            [sc.position for sc in sim.spacecrafts.values()],
            [sc.position for sc in sim.spacecrafts.values()],
            sim.tasks,
            ks_coal.comm_graph,
            final_partition,
            title="联盟形成结果"
        )
        coalition_fig.savefig(os.path.join(args.save_dir, 'coalition_formation.png'))
        
        # 3. 绘制效用演化
        utility_fig = visualizer.plot_utility_evolution(
            sim.utility_history,
            title="全局效用演化"
        )
        utility_fig.savefig(os.path.join(args.save_dir, 'utility_evolution.png'))
        
        # 4. 绘制信念演化 (以航天器0为例)
        belief_fig = visualizer.plot_belief_evolution(
            sim.belief_history[0],
            args.num_tasks,
            args.num_types,
            task_types,
            {0: "Easy", 1: "Moderate", 2: "Hard"},
            title="航天器0的信念演化"
        )
        belief_fig.savefig(os.path.join(args.save_dir, 'belief_evolution.png'))
        
    elif args.scenario == 'weak_comm':
        # 弱通信条件下的主动推理
        print("使用弱通信条件下的主动推理")
        
        # 初始化主动推理算法
        active_inference = ActiveInference(
            args.num_robots, args.num_tasks, 
            observation_error_std=2.0,
            alpha=0.4, beta=0.6,
            horizon=5, max_reasoning_level=3,
            dt=args.dt
        )
        
        # 运行主动推理
        positions_history, belief_history, converged, steps = sim.run_active_inference(
            active_inference, max_steps=args.max_iterations
        )
        
        # 保存结果
        result_file = os.path.join(args.save_dir, 'active_inference_result.txt')
        with open(result_file, 'w') as f:
            f.write(f"收敛: {converged}\n")
            f.write(f"步数: {steps}\n")
            f.write(f"最终分配: {sim.get_assignment_dict()}\n")
            f.write(f"全局效用: {sim.calculate_global_utility():.2f}\n")
        
        # 可视化结果
        # 1. 绘制轨迹
        trajectory_fig = visualizer.plot_trajectories(
            positions_history,
            [task.position for task in sim.tasks.values()],
            [sc.type for sc in sim.spacecrafts.values()],
            sim.get_assignment_dict(),
            title="航天器轨迹"
        )
        trajectory_fig.savefig(os.path.join(args.save_dir, 'trajectories.png'))
        
        # 2. 创建动画
        animation = visualizer.animate_trajectory(
            positions_history,
            [task.position for task in sim.tasks.values()],
            [sc.type for sc in sim.spacecrafts.values()],
            sim.get_assignment_dict(),
            interval=200,
            title="航天器轨迹动画"
        )
        animation.save(os.path.join(args.save_dir, 'trajectory_animation.mp4'))
        
        # 3. 绘制自由能变化
        energy_fig = visualizer.plot_free_energy(
            active_inference.free_energy_history,
            title="期望自由能变化"
        )
        energy_fig.savefig(os.path.join(args.save_dir, 'free_energy.png'))
        
    elif args.scenario == 'combined':
        # 组合方法
        print("使用组合方法")
        
        # 初始化KS-COAL算法
        ks_coal = KSCOAL(args.num_robots, args.num_tasks, K=None, 
                        max_iterations=20)
        
        # 初始化主动推理算法
        active_inference = ActiveInference(
            args.num_robots, args.num_tasks, 
            observation_error_std=2.0,
            alpha=0.4, beta=0.6,
            horizon=5, max_reasoning_level=3,
            dt=args.dt
        )
        
        # 运行组合方法
        utility_history, assignment_history = sim.run_combined_approach(
            ks_coal, active_inference, comm_threshold=0.7, 
            max_rounds=10, steps_per_round=5
        )
        
        # 保存结果
        result_file = os.path.join(args.save_dir, 'combined_result.txt')
        with open(result_file, 'w') as f:
            f.write(f"最终效用: {utility_history[-1]:.2f}\n")
            f.write(f"最终分配: {assignment_history[-1]}\n")
        
        # 可视化结果
        # 1.