from __future__ import annotations

"""
步态相位观察函数
用于计算和返回机器人的步态相位信息，帮助智能体理解当前步态周期中的位置
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """
    计算步态相位观察值

    步态相位用于表示机器人在步态周期中的位置，这对于四足机器人的步态控制非常重要。
    函数返回相位的正弦和余弦值，这样可以提供连续的相位信息，避免相位跳跃。

    参数:
        env: 强化学习环境实例
        period: 步态周期（秒），例如 0.8 表示一个完整的步态周期为 0.8 秒

    返回:
        torch.Tensor: 形状为 (num_envs, 2) 的张量
            - phase[:, 0]: 相位的正弦值 sin(2π * phase)
            - phase[:, 1]: 相位的余弦值 cos(2π * phase)
    """
    # 如果环境还没有存储当前回合的步数，则初始化为零
    # episode_length_buf 记录每个环境当前回合已经执行的步数
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long)

    # 计算全局相位：将当前时间转换为 [0, 1] 范围内的相位值
    # (env.episode_length_buf * env.step_dt) 是当前回合已经过的时间
    # % period 取模运算确保相位在 [0, period] 范围内
    # / period 归一化到 [0, 1] 范围
    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    # 创建相位张量，形状为 (num_envs, 2)
    # 第一列是正弦值，第二列是余弦值
    phase = torch.zeros(env.num_envs, 2, device=env.device)

    # 计算相位的正弦值：sin(2π * phase)
    # 这样可以提供周期性的相位信息
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)

    # 计算相位的余弦值：cos(2π * phase)
    # 正弦和余弦的组合可以唯一确定相位值，避免相位歧义
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    return phase
