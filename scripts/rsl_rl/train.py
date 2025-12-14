# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 RSL-RL 训练强化学习智能体的脚本。
该脚本用于训练基于策略的强化学习算法（如PPO），支持单GPU和多GPU分布式训练。
"""

"""重要提示：请先启动 Isaac Sim 模拟器，然后再运行此训练脚本。"""


import pathlib
import gymnasium as gym
import sys
import argparse
import argcomplete
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
import importlib.metadata as metadata
import platform
from packaging import version
import inspect
import os
import shutil
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner  # TODO: 考虑在终端中打印实验名称，方便用户识别当前训练任务。
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# 临时添加父目录到系统路径，以便导入环境列表模块
sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401  # 导入包以注册所有可用的环境任务

# 恢复系统路径，移除临时添加的路径
sys.path.pop(0)

# 从gym注册表中收集所有Unitree相关的任务（排除Isaac任务）
tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)


# 本地导入（已在上面完成）

# 添加命令行参数解析器，用于接收用户输入的训练配置参数
parser = argparse.ArgumentParser(description="使用 RSL-RL 训练强化学习智能体。")
parser.add_argument("--video", action="store_true",
                    default=False, help="在训练过程中录制视频，用于观察智能体的行为表现。")
parser.add_argument("--video_length", type=int, default=200,
                    help="录制的视频长度，单位为环境步数（steps）。")
parser.add_argument("--video_interval", type=int, default=2000,
                    help="视频录制的间隔，单位为训练步数（steps），每隔指定步数录制一次视频。")
parser.add_argument("--num_envs", type=int, default=None,
                    help="并行仿真的环境数量。更多的环境可以加快训练速度，但会消耗更多内存和计算资源。")
parser.add_argument("--task", type=str, default=None,
                    choices=tasks, help="要训练的任务名称，从可用的Unitree任务列表中选择。")
parser.add_argument("--seed", type=int, default=None,
                    help="随机种子值，用于确保实验的可重复性。相同的种子会产生相同的随机序列。")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="强化学习策略训练的最大迭代次数。每个迭代包含多个环境步数和一次策略更新。")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="启用多GPU或多节点的分布式训练模式，可以显著加速训练过程。"
)
# 添加 RSL-RL 库特定的命令行参数（如学习率、批次大小等）
cli_args.add_rsl_rl_args(parser)
# 添加 AppLauncher 的命令行参数（如设备选择、窗口设置等）
AppLauncher.add_app_launcher_args(parser)
# 启用命令行自动补全功能，提升用户体验
argcomplete.autocomplete(parser)
# 解析命令行参数，分离出Hydra配置参数和普通CLI参数
args_cli, hydra_args = parser.parse_known_args()

# 如果启用了视频录制功能，则自动启用相机，因为录制视频需要相机支持
if args_cli.video:
    args_cli.enable_cameras = True

# 清理sys.argv，只保留脚本名称和Hydra参数，以便Hydra能够正确解析配置
sys.argv = [sys.argv[0]] + hydra_args

# 启动Omniverse应用（Isaac Sim），这是运行仿真的基础环境
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""检查 RSL-RL 库的最低支持版本。
分布式训练需要特定版本的RSL-RL库支持，如果版本不匹配会提示用户升级。"""


# 对于分布式训练，检查已安装的RSL-RL库版本是否满足最低要求
RSL_RL_VERSION = "2.3.1"  # 分布式训练所需的最低RSL-RL版本
installed_version = metadata.version("rsl-rl-lib")  # 获取当前安装的RSL-RL版本
# 如果启用了分布式训练但版本过低，则提示用户升级并退出程序
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    # 根据操作系统类型生成相应的安装命令
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip",
               "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip",
               "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"请安装正确版本的 RSL-RL 库。\n当前安装的版本是: '{installed_version}'"
        f"，所需的最低版本是: '{RSL_RL_VERSION}'。\n要安装正确版本，请运行以下命令:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""以下为训练脚本的主要逻辑部分。"""


# 导入Isaac Lab任务模块，用于注册所有可用的Isaac任务环境
import isaaclab_tasks  # noqa: F401

# 导入Unitree RL Lab任务模块，用于注册所有可用的Unitree任务环境
import unitree_rl_lab.tasks  # noqa: F401

# 配置PyTorch的CUDA和cuDNN后端设置，以优化训练性能
# 启用TF32精度（TensorFloat-32），可以在保持性能的同时使用更快的计算模式
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 禁用确定性模式，允许cuDNN使用非确定性算法以获得更好的性能
torch.backends.cudnn.deterministic = False
# 禁用基准测试模式，避免在每次运行时重新测试最佳算法（可以提高启动速度）
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """使用 RSL-RL 智能体进行训练的主函数。

    该函数负责：
    1. 配置环境和智能体参数
    2. 创建训练环境
    3. 初始化训练器
    4. 执行训练过程
    5. 保存训练结果和配置
    """
    # 使用非Hydra的命令行参数覆盖配置文件中的设置
    # 这样用户可以通过命令行快速修改关键参数，而不需要修改配置文件
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # 如果命令行指定了环境数量，则使用命令行参数；否则使用配置文件中的默认值
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # 如果命令行指定了最大迭代次数，则使用命令行参数；否则使用配置文件中的默认值
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境的随机种子，确保实验的可重复性
    # 注意：某些随机化操作发生在环境初始化阶段，因此需要在这里设置种子
    env_cfg.seed = agent_cfg.seed
    # 如果命令行指定了计算设备（如cuda:0），则使用命令行参数；否则使用配置文件中的默认设备
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 多GPU分布式训练的配置
    # 在分布式训练模式下，每个进程使用不同的GPU和不同的随机种子
    if args_cli.distributed:
        # 根据当前进程的本地排名（local_rank）分配对应的GPU设备
        # 例如：进程0使用cuda:0，进程1使用cuda:1，以此类推
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 为不同的训练进程设置不同的随机种子，以增加训练数据的多样性
        # 每个进程的种子 = 基础种子 + 进程排名，这样可以确保不同进程使用不同的随机序列
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 指定实验日志的根目录路径
    # 日志目录结构：logs/rsl_rl/{实验名称}/
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)  # 转换为绝对路径，避免相对路径问题
    print(f"[INFO] 实验日志将保存在目录: {log_root_path}")
    # 为每次训练运行创建唯一的子目录，格式为：{时间戳}_{运行名称}
    # 时间戳格式：年-月-日_时-分-秒，例如：2025-01-15_14-30-25
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 这种命名方式使得Ray Tune等超参数调优工具可以轻松提取实验名称
    print(f"从命令行请求的确切实验名称: {log_dir}")
    # 如果配置中指定了运行名称，则追加到目录名后面
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)  # 组合完整的日志目录路径

    # 创建Isaac环境实例
    # 如果启用了视频录制，则设置渲染模式为rgb_array以生成视频帧；否则不渲染以节省计算资源
    env = gym.make(args_cli.task, cfg=env_cfg,render_mode="rgb_array" if args_cli.video else None)

    # 如果环境是多智能体环境（DirectMARLEnv），但RL算法需要单智能体接口
    # 则将多智能体环境转换为单智能体环境，以便与RSL-RL训练器兼容
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 在创建新的日志目录之前，先保存恢复训练的检查点路径
    # 如果启用了恢复训练（resume）或使用蒸馏算法（Distillation），需要加载之前的模型检查点
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        # 根据日志根目录、运行名称和检查点名称获取完整的检查点文件路径
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 如果启用了视频录制功能，则使用gym的RecordVideo包装器包装环境
    if args_cli.video:
        # 配置视频录制参数
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),  # 视频保存目录
            "step_trigger": lambda step: step % args_cli.video_interval == 0,  # 触发录制的条件：每隔指定步数录制一次
            "video_length": args_cli.video_length,  # 每个视频的长度（环境步数）
            "disable_logger": True,  # 禁用gym的默认日志记录器，避免重复日志
        }
        print("[INFO] 训练过程中将录制视频。")
        print_dict(video_kwargs, nesting=4)  # 打印视频录制配置信息，方便用户查看
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 用RecordVideo包装器包装环境

    # 使用RSL-RL的向量化环境包装器包装环境
    # clip_actions参数控制是否对动作进行裁剪，确保动作值在合法范围内
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建RSL-RL的在线策略训练器（OnPolicyRunner）
    # 该训练器负责管理训练循环、策略更新、日志记录等核心训练功能
    runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                            log_dir=log_dir, device=agent_cfg.device)
    # 将当前代码仓库的Git状态写入日志目录，方便追踪训练时使用的代码版本
    runner.add_git_repo_to_log(__file__)
    # 如果需要恢复训练或使用蒸馏算法，则加载之前保存的模型检查点
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: 正在从以下路径加载模型检查点: {resume_path}")
        # 加载之前训练好的模型权重和训练状态，以便继续训练或进行知识蒸馏
        runner.load(resume_path)

    # 将配置信息保存到日志目录中，方便后续查看和复现实验
    # 保存环境配置为YAML格式，包含所有环境相关的参数设置
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # 保存智能体配置为YAML格式，包含所有训练算法相关的参数设置
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # 导出部署配置，用于后续的模型部署和推理
    export_deploy_cfg(env.unwrapped, log_dir)
    # 将环境配置类的源文件复制到日志目录，确保完整保存配置定义
    # 这样即使配置文件发生变化，也能知道训练时使用的确切配置代码
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(
            inspect.getfile(env_cfg.__class__))),
    )

    # 开始执行训练过程
    # num_learning_iterations: 训练的总迭代次数
    # init_at_random_ep_len: 是否在随机长度的episode处初始化，这有助于提高训练的稳定性
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)

    # 训练完成后，关闭仿真环境，释放资源
    env.close()


if __name__ == "__main__":
    # 运行主训练函数，执行完整的训练流程
    main()
    # 关闭Isaac Sim应用，清理所有相关资源
    simulation_app.close()
