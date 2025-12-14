# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO (Proximal Policy Optimization) 强化学习算法的配置文件
用于配置 RSL-RL 库中的 PPO 训练器参数
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    基础 PPO 训练器配置类
    继承自 RslRlOnPolicyRunnerCfg，定义了 PPO 算法的所有超参数
    """
    
    # 每个环境每次收集的步数（rollout length）
    # 这个值决定了每次更新前需要收集多少经验数据
    num_steps_per_env = 24
    
    # 最大训练迭代次数
    # 训练将在此迭代次数后停止
    max_iterations = 50000
    
    # 模型保存间隔（以迭代次数为单位）
    # 每训练这么多次迭代就保存一次模型检查点
    save_interval = 100
    
    # 实验名称，通常与任务名称相同
    # 用于区分不同的训练实验
    experiment_name = ""  # same as task name
    
    # 是否使用经验归一化
    # False 表示不使用经验归一化，直接使用原始观察值
    empirical_normalization = False
    
    # 策略网络（Actor）和值函数网络（Critic）的配置
    policy = RslRlPpoActorCriticCfg(
        # 初始动作噪声标准差
        # 用于在训练初期增加探索，帮助智能体探索动作空间
        init_noise_std=1.0,
        
        # Actor 网络隐藏层维度
        # 定义了策略网络的层数和每层的神经元数量 [512, 256, 128] 表示三层网络
        actor_hidden_dims=[512, 256, 128],
        
        # Critic 网络隐藏层维度
        # 定义了值函数网络的层数和每层的神经元数量
        critic_hidden_dims=[512, 256, 128],
        
        # 激活函数类型
        # "elu" 表示使用指数线性单元激活函数
        activation="elu",
    )
    
    # PPO 算法超参数配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 值函数损失系数
        # 控制值函数损失在总损失中的权重
        value_loss_coef=1.0,
        
        # 是否使用裁剪的值函数损失
        # True 表示对值函数损失进行裁剪，防止值函数更新过大
        use_clipped_value_loss=True,
        
        # PPO 裁剪参数（epsilon）
        # 限制策略更新的幅度，防止策略更新过大导致性能下降
        clip_param=0.2,
        
        # 熵系数
        # 鼓励策略探索，防止过早收敛到次优策略
        entropy_coef=0.01,
        
        # 每次更新时的学习轮数
        # 对同一批经验数据进行多次梯度更新
        num_learning_epochs=5,
        
        # 小批量（mini-batch）的数量
        # 将收集的经验数据分成多个小批次进行训练
        num_mini_batches=4,
        
        # 学习率
        # 控制参数更新的步长
        learning_rate=1.0e-3,
        
        # 学习率调度策略
        # "adaptive" 表示自适应调整学习率
        schedule="adaptive",
        
        # 折扣因子（gamma）
        # 用于计算未来奖励的现值，范围 [0, 1]
        gamma=0.99,
        
        # GAE (Generalized Advantage Estimation) 的 lambda 参数
        # 用于平衡偏差和方差，范围 [0, 1]
        lam=0.95,
        
        # 期望的 KL 散度
        # 用于自适应调整学习率，保持策略更新的稳定性
        desired_kl=0.01,
        
        # 梯度裁剪的最大范数
        # 防止梯度爆炸，稳定训练过程
        max_grad_norm=1.0,
    )
