from __future__ import annotations

"""
速度命令配置
定义了用于课程学习的速度命令配置类，支持动态调整速度范围
"""

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """
    均匀层级速度命令配置类
    
    继承自 UniformVelocityCommandCfg，增加了 limit_ranges 属性
    用于课程学习：在训练过程中逐步增加速度命令的范围
    
    属性:
        limit_ranges: 速度范围的最大限制，课程学习不能超过此范围
            包含 lin_vel_x, lin_vel_y, ang_vel_z 的最大和最小值
    """
    # 速度范围的最大限制（用于课程学习）
    # MISSING 表示这是一个必需字段，必须在实例化时提供
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
