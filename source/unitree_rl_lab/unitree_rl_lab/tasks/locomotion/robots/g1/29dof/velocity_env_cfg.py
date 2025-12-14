"""
Unitree G1 29自由度机器人速度跟踪环境配置
定义了用于训练 G1 机器人速度跟踪任务的完整环境配置，包括场景、观察、动作、奖励等
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# 鹅卵石道路地形生成器配置
# 用于生成训练地形，包括平坦地面和不同难度的地形
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 每个地形块的大小（米）
    border_width=20.0,  # 边界宽度（米）
    num_rows=9,  # 地形行数（用于课程学习，从简单到困难）
    num_cols=21,  # 地形列数
    horizontal_scale=0.1,  # 水平方向的地形细节缩放
    vertical_scale=0.005,  # 垂直方向的地形高度缩放
    slope_threshold=0.75,  # 坡度阈值，超过此值视为陡坡
    difficulty_range=(0.0, 1.0),  # 难度范围 [0, 1]
    use_cache=False,  # 是否使用缓存
    sub_terrains={
        # 平坦地形，占 50% 的比例
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """
    事件配置
    定义了在训练过程中发生的各种事件，用于域随机化和提高鲁棒性
    包括启动时的事件、重置时的事件和间隔事件
    """

    # 启动时的事件（在环境初始化时执行一次）
    # 随机化物理材质：为机器人各部件设置随机的摩擦系数
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",  # 启动模式：在环境启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 应用到所有机器人部件
            "static_friction_range": (0.3, 1.0),  # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 1.0),  # 动摩擦系数范围
            "restitution_range": (0.0, 0.0),  # 弹性系数范围（0表示无弹性）
            "num_buckets": 64,  # 随机化分桶数量（用于离散化随机值）
        },
    )

    # 随机化基础质量：为躯干添加随机质量
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",  # 启动模式
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 应用到躯干
            "mass_distribution_params": (-1.0, 3.0),  # 质量分布参数（相对于原始质量）
            "operation": "add",  # 操作类型：添加（而非替换）
        },
    )

    # 重置时的事件（在每个回合开始时执行）
    # 施加外部力和力矩：在重置时对机器人施加外部干扰（当前设置为0，即无干扰）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",  # 重置模式：在每个回合开始时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 应用到躯干
            "force_range": (0.0, 0.0),  # 力范围（当前为0，表示不施加力）
            "torque_range": (-0.0, 0.0),  # 力矩范围（当前为0，表示不施加力矩）
        },
    )

    # 重置机器人基座状态：随机设置机器人的初始位置和姿态
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",  # 重置模式
        params={
            # 位置范围：在 xy 平面内随机分布，偏航角随机
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # 速度范围：所有速度初始化为0
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # 重置机器人关节：随机设置关节位置和速度
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",  # 重置模式
        params={
            "position_range": (1.0, 1.0),  # 关节位置范围（相对于默认位置，1.0表示使用默认位置）
            "velocity_range": (-1.0, 1.0),  # 关节速度范围（归一化值）
        },
    )

    # 间隔事件（在训练过程中定期执行）
    # 推动机器人：定期对机器人施加水平方向的推动
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",  # 间隔模式：定期执行
        interval_range_s=(5.0, 5.0),  # 执行间隔：每5秒执行一次
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # 推动速度范围（xy方向）
    )


@configclass
class CommandsCfg:
    """
    命令配置
    定义了智能体需要跟踪的目标命令（速度命令）
    """

    # 基础速度命令：机器人需要跟踪的线速度和角速度
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",  # 目标资产名称
        resampling_time_range=(10.0, 10.0),  # 命令重新采样时间范围（秒），每10秒生成新命令
        rel_standing_envs=0.02,  # 保持静止的环境比例（2%的环境保持零速度命令）
        rel_heading_envs=1.0,  # 使用航向命令的环境比例（100%）
        heading_command=False,  # 是否使用航向命令（False表示不使用）
        debug_vis=True,  # 是否显示调试可视化
        # 初始速度范围（用于课程学习，从简单开始）
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),  # x方向线速度范围（米/秒）
            lin_vel_y=(-0.1, 0.1),  # y方向线速度范围（米/秒）
            ang_vel_z=(-0.1, 0.1)  # z方向角速度范围（弧度/秒）
        ),
        # 最大速度范围限制（课程学习的上限）
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),  # x方向最大速度范围（可后退0.5，前进1.0米/秒）
            lin_vel_y=(-0.3, 0.3),  # y方向最大速度范围（左右各0.3米/秒）
            ang_vel_z=(-0.2, 0.2)  # z方向最大角速度范围（左右各0.2弧度/秒）
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """
    观察配置
    定义了智能体可以观察到的状态信息
    包括策略网络（Policy）和值函数网络（Critic）的观察
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """
        策略网络观察配置
        定义了策略网络（Actor）可以观察到的状态信息
        这些观察值会输入到策略网络，用于决策
        """

        # 观察项（顺序会被保留）
        # 基础角速度：机器人本体的角速度
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,  # 缩放因子
            noise=Unoise(n_min=-0.2, n_max=0.2)  # 添加均匀噪声（用于域随机化）
        )
        # 投影重力：重力在机器人本体坐标系中的投影（用于感知姿态）
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)  # 添加噪声
        )
        # 速度命令：当前需要跟踪的目标速度
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        # 关节相对位置：关节位置相对于默认位置的偏差
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)  # 添加噪声
        )
        # 关节相对速度：关节角速度（归一化）
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,  # 缩放因子
            noise=Unoise(n_min=-1.5, n_max=1.5)  # 添加噪声
        )
        # 上一个动作：上一次执行的动作（用于动作平滑）
        last_action = ObsTerm(func=mdp.last_action)
        # 步态相位（已注释）：用于步态控制的相位信息
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5  # 历史长度：保留过去5个时间步的观察
            self.enable_corruption = True  # 启用观察损坏（用于域随机化）
            self.concatenate_terms = True  # 将所有观察项连接成一个向量

    # 观察组
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """
        值函数网络观察配置
        定义了值函数网络（Critic）可以观察到的状态信息
        Critic 通常可以使用更多信息（特权信息）来更好地估计价值
        """

        # 基础线速度：机器人本体的线速度（Policy 没有此信息）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # 基础角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        # 投影重力
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 速度命令
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        # 关节相对位置
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 关节相对速度
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        # 上一个动作
        last_action = ObsTerm(func=mdp.last_action)
        # 步态相位（已注释）
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # 高度扫描（已注释）：地面高度信息
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5  # 历史长度

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """
    奖励配置
    定义了强化学习训练中使用的所有奖励项
    奖励项分为几类：任务奖励、基础奖励、机器人姿态奖励、足部奖励等
    每个奖励项都有权重（weight），用于平衡不同目标的重要性
    """

    # -- 任务相关奖励（主要目标）
    # 跟踪线性速度（xy方向）：奖励机器人跟踪目标线速度
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,  # 使用指数函数计算奖励
        weight=1.0,  # 奖励权重
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},  # 标准差参数
    )
    # 跟踪角速度（z方向）：奖励机器人跟踪目标角速度
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,  # 权重较小，因为角速度跟踪相对容易
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # 存活奖励：鼓励机器人保持站立，不摔倒
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-
                          10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={
                           "minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """
    机器人环境配置
    整合了所有配置类，定义了完整的强化学习环境
    这是训练和评估时使用的主配置类
    """

    # 场景设置
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)  # 4096个并行环境，间距2.5米
    # 基础设置
    observations: ObservationsCfg = ObservationsCfg()  # 观察配置
    actions: ActionsCfg = ActionsCfg()  # 动作配置
    commands: CommandsCfg = CommandsCfg()  # 命令配置
    # MDP（马尔可夫决策过程）设置
    rewards: RewardsCfg = RewardsCfg()  # 奖励配置
    terminations: TerminationsCfg = TerminationsCfg()  # 终止条件配置
    events: EventCfg = EventCfg()  # 事件配置
    curriculum: CurriculumCfg = CurriculumCfg()  # 课程学习配置

    def __post_init__(self):
        """
        后初始化
        在配置类实例化后自动调用，用于设置一些依赖其他配置的参数
        """
        # 通用设置
        self.decimation = 4  # 降采样因子：每4个物理步执行一次策略更新
        self.episode_length_s = 20.0  # 每个回合的长度（秒）

        # 仿真设置
        self.sim.dt = 0.005  # 物理仿真时间步长（秒）
        self.sim.render_interval = self.decimation  # 渲染间隔（与策略更新频率一致）
        self.sim.physics_material = self.scene.terrain.physics_material  # 使用地形的物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # PhysX GPU最大刚体补丁数

        # 更新传感器更新周期
        # 所有传感器基于最小的更新周期（物理更新周期）进行更新
        self.scene.contact_forces.update_period = self.sim.dt  # 接触力传感器：每个物理步更新
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 高度扫描器：每个策略步更新

        # 检查是否启用了地形等级课程学习
        # 如果启用，地形生成器会生成难度递增的地形，这对训练很有用
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """
    机器人演示环境配置
    用于演示和评估训练好的策略，与训练环境相比：
    - 环境数量更少（32个，便于可视化）
    - 地形更简单（2行10列）
    - 速度命令范围使用最大值（测试完整性能）
    """
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32  # 演示时使用更少的环境
        self.scene.terrain.terrain_generator.num_rows = 2  # 减少地形行数
        self.scene.terrain.terrain_generator.num_cols = 10  # 减少地形列数
        # 使用最大速度范围（测试完整性能）
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
