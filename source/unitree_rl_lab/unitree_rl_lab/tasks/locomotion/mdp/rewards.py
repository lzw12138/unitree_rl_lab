from __future__ import annotations

"""
奖励函数模块
定义了强化学习训练中使用的各种奖励项，用于引导智能体学习期望的行为
包括关节惩罚、机器人姿态奖励、足部奖励、步态奖励等
"""

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
关节惩罚相关函数
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚机器人关节消耗的能量

    能量消耗 = Σ|关节速度| × |关节力矩|
    这个奖励项鼓励智能体使用更少的能量，提高能效

    参数:
        env: 强化学习环境实例
        asset_cfg: 场景实体配置，指定要计算能量的机器人关节

    返回:
        torch.Tensor: 每个环境的能量消耗惩罚值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取指定关节的角速度
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # 获取指定关节的施加力矩
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # 计算能量：速度的绝对值 × 力矩的绝对值，然后对所有关节求和
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    奖励机器人保持静止状态

    当速度命令很小时（接近零），鼓励机器人保持默认姿态（站立姿态）
    这个奖励项帮助智能体学习在不需要移动时保持稳定

    参数:
        env: 强化学习环境实例
        command_name: 命令名称，默认为 "base_velocity"
        asset_cfg: 场景实体配置

    返回:
        torch.Tensor: 每个环境的奖励值，当命令很小时返回关节位置偏差，否则返回0
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算关节位置与默认位置的偏差（L1距离）
    reward = torch.sum(torch.abs(asset.data.joint_pos -
                       asset.data.default_joint_pos), dim=1)
    # 获取速度命令的范数（大小）
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    # 只有当命令很小时（< 0.1）才给予奖励，否则返回0
    return reward * (cmd_norm < 0.1)


"""
机器人姿态相关奖励函数
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    奖励智能体将重力方向与期望重力方向对齐（使用 L2 平方核）

    通过计算投影重力和期望重力的余弦距离，鼓励机器人保持正确的姿态

    参数:
        env: 强化学习环境实例
        desired_gravity: 期望的重力方向向量（在机器人本体坐标系中）
        asset_cfg: 场景实体配置

    返回:
        torch.Tensor: 每个环境的奖励值，范围 [0, 1]，1 表示完全对齐
    """
    # 提取使用的量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]

    # 将期望重力向量转换为张量
    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    # 计算余弦距离（点积）：投影重力与期望重力的内积
    # 值越接近1表示对齐越好，值越接近-1表示对齐越差
    cos_dist = torch.sum(asset.data.projected_gravity_b *
                         desired_gravity, dim=-1)  # cosine distance
    # 将余弦距离从 [-1, 1] 映射到 [0, 1]
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    # 返回平方值，使奖励更平滑
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    奖励机器人保持向上姿态（使用 L2 平方核）
    
    通过惩罚 z 轴方向的重力投影偏差，鼓励机器人保持直立
    
    参数:
        env: 强化学习环境实例
        asset_cfg: 场景实体配置
    
    返回:
        torch.Tensor: 每个环境的奖励值，当机器人直立时（projected_gravity_b[:, 2] = -1）奖励最大
    """
    # 提取使用的量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算奖励：当 projected_gravity_b[:, 2] = -1（完全向上）时，reward = 0（最大）
    # 当 projected_gravity_b[:, 2] 偏离 -1 时，reward 增大（惩罚增加）
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """
    惩罚关节位置与默认位置的偏差
    
    当机器人静止或移动速度很小时，使用更大的惩罚系数，鼓励保持默认姿态
    当机器人正在移动时，允许更大的关节位置偏差
    
    参数:
        env: 强化学习环境实例
        asset_cfg: 场景实体配置
        stand_still_scale: 静止时的惩罚缩放系数（通常 > 1，使惩罚更大）
        velocity_threshold: 速度阈值，低于此值视为静止
    
    返回:
        torch.Tensor: 每个环境的关节位置偏差惩罚值
    """
    # 提取使用的量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算速度命令的范数
    cmd = torch.linalg.norm(
        env.command_manager.get_command("base_velocity"), dim=1)
    # 计算机器人本体的线速度（xy平面）范数
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    # 计算关节位置与默认位置的偏差（L2范数）
    reward = torch.linalg.norm(
        (asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    # 如果命令不为零或本体速度超过阈值，使用正常惩罚；否则使用放大后的惩罚
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
足部相关奖励函数
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    惩罚足部撞击垂直表面（绊倒）
    
    检测足部是否撞击到垂直表面（如墙壁），这种情况会导致机器人绊倒
    通过比较水平力和垂直力的大小来判断
    
    参数:
        env: 强化学习环境实例
        sensor_cfg: 接触传感器配置
    
    返回:
        torch.Tensor: 每个环境的惩罚值，1 表示检测到绊倒，0 表示正常
    """
    # 提取使用的量（用于类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取 z 轴方向（垂直方向）的接触力绝对值
    forces_z = torch.abs(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    # 获取 xy 平面（水平方向）的接触力范数
    forces_xy = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # 惩罚足部撞击垂直表面：如果水平力 > 4 × 垂直力，则认为撞击了垂直表面
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """
    奖励摆动足部达到指定高度（在机器人本体坐标系中）
    
    鼓励机器人在摆动阶段将足部抬起到目标高度，避免拖地
    奖励与足部高度误差和水平速度相关
    
    参数:
        env: 强化学习环境实例
        command_name: 命令名称
        asset_cfg: 场景实体配置，指定要检查的足部
        target_height: 目标高度（在机器人本体坐标系中）
        tanh_mult: tanh 函数的倍数，用于缩放速度项
    
    返回:
        torch.Tensor: 每个环境的奖励值
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:,
                                                   asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(
        env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(
        env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(
        footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(
        env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"]
                          .data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """
    奖励摆动足部达到指定高度（在世界坐标系中）
    
    使用指数函数将高度误差转换为奖励，误差越小奖励越大
    奖励还与足部水平速度相关，速度越大权重越大
    
    参数:
        env: 强化学习环境实例
        asset_cfg: 场景实体配置，指定要检查的足部
        target_height: 目标高度（在世界坐标系中，相对于地面）
        std: 标准差，用于控制奖励的衰减速度
        tanh_mult: tanh 函数的倍数，用于缩放速度项
    
    返回:
        torch.Tensor: 每个环境的奖励值，范围 [0, 1]，1 表示完全达到目标
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚足部距离过近
    
    防止机器人两足距离过近导致的不稳定步态
    
    参数:
        env: 强化学习环境实例
        threshold: 最小距离阈值，低于此值将受到惩罚
        asset_cfg: 场景实体配置，指定要检查的足部
    
    返回:
        torch.Tensor: 每个环境的惩罚值，距离越近惩罚越大
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取足部在世界坐标系中的位置
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # 计算两足之间的距离（假设只有两个足部）
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    # 如果距离小于阈值，返回惩罚值（阈值 - 距离），否则返回0
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    奖励在无命令时足部与地面接触
    
    当速度命令为零时，鼓励机器人保持足部与地面接触，保持稳定站立
    
    参数:
        env: 强化学习环境实例
        sensor_cfg: 接触传感器配置
        command_name: 命令名称，默认为 "base_velocity"
    
    返回:
        torch.Tensor: 每个环境的奖励值，当命令很小时返回接触足部数量，否则返回0
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:,
                                                          sensor_cfg.body_ids] > 0

    command_norm = torch.norm(
        env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    惩罚各足部在空中/地面时间的不一致性
    
    鼓励各足部的步态周期保持一致，避免步态不协调
    
    参数:
        env: 强化学习环境实例
        sensor_cfg: 接触传感器配置
    
    返回:
        torch.Tensor: 每个环境的惩罚值，方差越大惩罚越大
    
    注意:
        需要启用 ContactSensor 的 track_air_time 功能
    """
    # 提取使用的量（用于类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检查是否启用了空中时间跟踪
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # 计算奖励
    # 获取每个足部最后一次在空中的时间
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # 获取每个足部最后一次在地面的时间
    last_contact_time = contact_sensor.data.last_contact_time[:,
                                                              sensor_cfg.body_ids]
    # 计算各足部空中时间的方差和地面时间的方差，并求和
    # 使用 clip 限制最大值，避免异常值影响
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
足部步态相关奖励函数
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """
    奖励符合期望步态的足部接触模式
    
    根据步态周期和相位偏移，检查足部是否在正确的时刻与地面接触
    用于训练特定的步态模式（如对角步态、小跑步态等）
    
    参数:
        env: 强化学习环境实例
        period: 步态周期（秒）
        offset: 各足部的相位偏移列表，例如 [0.0, 0.5] 表示两足相位差180度
        sensor_cfg: 接触传感器配置
        threshold: 相位阈值，低于此值视为支撑相（stance phase）
        command_name: 命令名称，如果提供，只在有命令时给予奖励
    
    返回:
        torch.Tensor: 每个环境的奖励值，符合步态时奖励更大
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检查哪些足部正在与地面接触
    is_contact = contact_sensor.data.current_contact_time[:,
                                                          sensor_cfg.body_ids] > 0

    # 计算全局相位：将当前时间转换为 [0, 1] 范围内的相位值
    global_phase = ((env.episode_length_buf * env.step_dt) %
                    period / period).unsqueeze(1)
    # 为每个足部计算相位（加上偏移量）
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    # 将所有足部的相位拼接在一起
    leg_phase = torch.cat(phases, dim=-1)

    # 初始化奖励为零
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    # 对每个足部检查是否符合步态
    for i in range(len(sensor_cfg.body_ids)):
        # 判断当前相位是否处于支撑相
        is_stance = leg_phase[:, i] < threshold
        # 使用异或运算检查：如果 (支撑相 AND 接触) 或 (摆动相 AND 不接触)，则符合步态
        # ~(is_stance ^ is_contact) 表示两者状态一致时返回 True
        reward += ~(is_stance ^ is_contact[:, i])

    # 如果提供了命令名称，只在有命令时给予奖励
    if command_name is not None:
        cmd_norm = torch.norm(
            env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
其他奖励函数
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """
    惩罚镜像关节位置的不对称性
    
    鼓励左右对称的关节保持对称位置，这对于双足或四足机器人的稳定步态很重要
    
    参数:
        env: 强化学习环境实例
        asset_cfg: 场景实体配置
        mirror_joints: 镜像关节对列表，每个元素是一个包含两个关节名称的列表
            例如 [["left_hip", "right_hip"], ["left_knee", "right_knee"]]
    
    返回:
        torch.Tensor: 每个环境的惩罚值，不对称性越大惩罚越大
    """
    # 提取使用的量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 缓存关节索引（避免重复查找）
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # 为所有关节对缓存关节位置索引
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # 遍历所有关节对
    for joint_pair in env.joint_mirror_joints_cache:
        # 计算每对关节的位置差值的平方，并累加到总奖励中
        reward += torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    # 平均化：除以关节对的数量
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
