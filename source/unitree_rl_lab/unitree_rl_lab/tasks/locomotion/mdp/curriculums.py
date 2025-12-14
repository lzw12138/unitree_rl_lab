from __future__ import annotations

"""
课程学习（Curriculum Learning）函数
用于在训练过程中逐步增加任务难度，帮助智能体更好地学习
"""

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """
    根据性能动态调整线性速度命令范围
    
    这是一种课程学习策略：当智能体在当前的线性速度跟踪任务上表现良好时，
    逐步增加速度命令的范围，提高任务难度
    
    参数:
        env: 强化学习环境实例
        env_ids: 环境ID序列
        reward_term_name: 奖励项名称，默认为 "track_lin_vel_xy"
    
    返回:
        torch.Tensor: 当前线性速度 x 的最大值（用于监控）
    """
    # 获取速度命令管理器
    command_term = env.command_manager.get_term("base_velocity")
    # 获取当前的速度范围配置
    ranges = command_term.cfg.ranges
    # 获取速度范围的限制（最大允许范围）
    limit_ranges = command_term.cfg.limit_ranges

    # 获取奖励项配置
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # 计算平均奖励（归一化到每个时间步）
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 在每个回合结束时检查是否需要增加难度
    if env.common_step_counter % env.max_episode_length == 0:
        # 如果平均奖励超过目标奖励的 80%，则增加速度范围
        if reward > reward_term.weight * 0.8:
            # 定义速度范围的增量（负值和正值，用于扩大范围）
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            # 更新 x 方向速度范围，并限制在最大允许范围内
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            # 更新 y 方向速度范围，并限制在最大允许范围内
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    # 返回当前 x 方向速度的最大值（用于监控课程进度）
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """
    根据性能动态调整角速度命令范围
    
    这是一种课程学习策略：当智能体在当前的角速度跟踪任务上表现良好时，
    逐步增加角速度命令的范围，提高任务难度
    
    参数:
        env: 强化学习环境实例
        env_ids: 环境ID序列
        reward_term_name: 奖励项名称，默认为 "track_ang_vel_z"
    
    返回:
        torch.Tensor: 当前角速度 z 的最大值（用于监控）
    """
    # 获取速度命令管理器
    command_term = env.command_manager.get_term("base_velocity")
    # 获取当前的速度范围配置
    ranges = command_term.cfg.ranges
    # 获取速度范围的限制（最大允许范围）
    limit_ranges = command_term.cfg.limit_ranges

    # 获取奖励项配置
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # 计算平均奖励（归一化到每个时间步）
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 在每个回合结束时检查是否需要增加难度
    if env.common_step_counter % env.max_episode_length == 0:
        # 如果平均奖励超过目标奖励的 80%，则增加速度范围
        if reward > reward_term.weight * 0.8:
            # 定义速度范围的增量（负值和正值，用于扩大范围）
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            # 更新 z 方向角速度范围，并限制在最大允许范围内
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    # 返回当前 z 方向角速度的最大值（用于监控课程进度）
    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
