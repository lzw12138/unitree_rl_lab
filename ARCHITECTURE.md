# Unitree RL Lab 项目架构文档

## 📋 目录

1. [项目概述](#项目概述)
2. [整体架构](#整体架构)
3. [训练系统架构](#训练系统架构)
4. [部署系统架构](#部署系统架构)
5. [数据流](#数据流)
6. [关键组件](#关键组件)
7. [技术栈](#技术栈)

---

## 项目概述

**Unitree RL Lab** 是一个基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 构建的强化学习训练和部署框架，专门用于训练和控制宇树（Unitree）机器人。

### 核心功能

- 🤖 **多机器人支持**: Go2（四足）、H1（人形）、G1-29dof（人形）
- 🎯 **多任务支持**: Locomotion（运动控制）、Mimic（动作模仿）
- 🚀 **完整流程**: 训练 → Sim2Sim验证 → Sim2Real部署
- 🔧 **模块化设计**: 易于扩展和定制

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Unitree RL Lab                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐         ┌──────────────────────┐      │
│  │   训练系统 (Python)   │         │  部署系统 (C++)      │      │
│  │                      │         │                      │      │
│  │  ┌────────────────┐  │         │  ┌────────────────┐  │      │
│  │  │  IsaacLab      │  │         │  │  FSM控制器     │  │      │
│  │  │  仿真环境      │  │         │  │                │  │      │
│  │  └────────────────┘  │         │  └────────────────┘  │      │
│  │         │            │         │         │            │      │
│  │  ┌────────────────┐  │         │  ┌────────────────┐  │      │
│  │  │  RSL-RL        │  │         │  │  ONNX推理     │  │      │
│  │  │  PPO训练       │  │         │  │                │  │      │
│  │  └────────────────┘  │         │  └────────────────┘  │      │
│  │         │            │         │         │            │      │
│  │  ┌────────────────┐  │         │  ┌────────────────┐  │      │
│  │  │  模型导出      │──┼─────────>│  │  模型加载     │  │      │
│  │  │  (ONNX)        │  │         │  │                │  │      │
│  │  └────────────────┘  │         │  └────────────────┘  │      │
│  └──────────────────────┘         └──────────────────────┘      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 架构层次

```
应用层
  ├── 训练脚本 (scripts/rsl_rl/train.py)
  ├── 推理脚本 (scripts/rsl_rl/play.py)
  └── 部署程序 (deploy/robots/*/main.cpp)

任务层
  ├── Locomotion (运动控制任务)
  └── Mimic (动作模仿任务)

环境层
  ├── 机器人配置 (robots/*/velocity_env_cfg.py)
  ├── MDP组件 (observations, rewards, commands)
  └── 算法配置 (agents/rsl_rl_ppo_cfg.py)

基础设施层
  ├── IsaacLab (仿真引擎)
  ├── RSL-RL (强化学习库)
  └── Unitree SDK2 (机器人通信)
```

---

## 训练系统架构

### 目录结构

```
source/unitree_rl_lab/unitree_rl_lab/
├── tasks/                          # 任务定义
│   ├── locomotion/                 # 运动控制任务
│   │   ├── agents/                 # 算法配置
│   │   │   └── rsl_rl_ppo_cfg.py  # PPO超参数
│   │   ├── mdp/                    # MDP组件
│   │   │   ├── observations.py    # 观察空间
│   │   │   ├── rewards.py         # 奖励函数
│   │   │   ├── commands/          # 命令生成
│   │   │   └── curriculums.py     # 课程学习
│   │   └── robots/                 # 机器人配置
│   │       ├── go2/
│   │       ├── g1/29dof/
│   │       └── h1/
│   └── mimic/                      # 动作模仿任务
│       ├── agents/
│       ├── mdp/
│       └── robots/
├── assets/                         # 资源文件
│   └── robots/
│       ├── unitree.py              # 机器人模型加载
│       └── unitree_actuators.py   # 执行器配置
└── utils/                          # 工具函数
    ├── export_deploy_cfg.py       # 配置导出
    └── parser_cfg.py              # 配置解析
```

### 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                      训练流程                                 │
└─────────────────────────────────────────────────────────────┘

1. 初始化阶段
   ├── 加载环境配置 (velocity_env_cfg.py)
   ├── 创建仿真场景 (RobotSceneCfg)
   │   ├── 地形生成器
   │   ├── 机器人模型
   │   └── 传感器配置
   ├── 配置MDP组件
   │   ├── 观察空间 (ObservationsCfg)
   │   ├── 动作空间 (ActionsCfg)
   │   ├── 命令生成 (CommandsCfg)
   │   ├── 奖励函数 (RewardsCfg)
   │   └── 终止条件 (TerminationsCfg)
   └── 初始化PPO算法 (rsl_rl_ppo_cfg.py)

2. 训练循环 (每个iteration)
   │
   ├── 数据收集阶段 (Rollout)
   │   ├── 环境重置
   │   │   ├── 事件处理 (EventCfg)
   │   │   │   ├── 物理参数随机化
   │   │   │   ├── 机器人状态重置
   │   │   │   └── 外部干扰
   │   │   └── 生成新命令
   │   │
   │   ├── 并行环境步进 (num_envs次)
   │   │   ├── 计算观察 (observations)
   │   │   ├── 策略网络推理 → 动作
   │   │   ├── 执行动作 → 环境更新
   │   │   ├── 计算奖励 (rewards)
   │   │   └── 检查终止条件
   │   │
   │   └── 课程学习更新 (CurriculumCfg)
   │
   └── 策略更新阶段 (Update)
       ├── 计算优势函数
       ├── PPO算法更新
       │   ├── Actor网络更新
       │   └── Critic网络更新
       └── 保存检查点

3. 模型导出
   └── 转换为ONNX格式 → deploy/robots/*/config/policy/
```

### MDP组件详解

#### 1. 观察空间 (Observations)

```python
Policy观察组 (策略网络):
  ├── 基础状态
  │   ├── 角速度 (base_ang_vel)
  │   ├── 重力投影 (gravity_proj)
  │   └── 速度命令 (commands)
  ├── 关节状态
  │   ├── 相对位置 (joint_pos_rel)
  │   └── 速度 (joint_vel)
  └── 历史动作 (actions_history)

Critic观察组 (价值网络):
  ├── Policy观察的所有内容
  └── 特权信息 (真实速度等)
```

#### 2. 奖励函数 (Rewards)

```python
任务奖励:
  └── 速度跟踪奖励 (tracking_lin_vel, tracking_ang_vel)

基础奖励:
  ├── 姿态奖励 (orientation, upward)
  ├── 能量惩罚 (energy)
  └── 平滑性奖励 (action_rate)

足部奖励 (仅四足/人形):
  ├── 步态奖励 (feet_gait)
  ├── 接触奖励 (feet_contact)
  └── 抬升奖励 (feet_height)

惩罚项:
  ├── 非法接触惩罚
  └── 关节限制惩罚
```

#### 3. 命令生成 (Commands)

```python
速度命令:
  ├── 线性速度 (x, y)
  └── 角速度 (z)

课程学习:
  ├── 逐步增加速度范围
  └── 根据性能自适应调整
```

#### 4. 事件处理 (Events)

```python
启动时 (startup):
  ├── 随机化物理材质 (摩擦、恢复系数)
  └── 随机化机器人质量

重置时 (reset):
  ├── 重置机器人位置和姿态
  ├── 重置关节状态
  └── 应用外部力/力矩

间隔 (interval):
  └── 随机推动机器人
```

---

## 部署系统架构

### 目录结构

```
deploy/
├── include/                        # 头文件
│   ├── FSM/                        # 有限状态机
│   │   ├── BaseState.h            # 状态基类
│   │   ├── CtrlFSM.h              # FSM控制器
│   │   ├── State_Passive.h        # 被动状态
│   │   ├── State_FixStand.h       # 固定站立状态
│   │   └── State_RLBase.h         # RL控制状态
│   ├── isaaclab/                   # IsaacLab接口
│   │   ├── algorithms/            # 算法接口
│   │   ├── envs/                   # 环境接口
│   │   └── manager/                # 管理器接口
│   └── unitree_articulation.h     # 机器人接口
│
└── robots/                         # 机器人特定实现
    ├── g1_29dof/
    │   ├── CMakeLists.txt
    │   ├── config/
    │   │   ├── config.yaml         # FSM配置
    │   │   └── policy/             # ONNX模型
    │   ├── include/
    │   │   └── Types.h             # 类型定义
    │   ├── main.cpp                # 主程序
    │   └── src/
    │       └── State_RLBase.cpp   # RL状态实现
    ├── go2/
    ├── h1/
    └── ...
```

### 部署流程

```
┌─────────────────────────────────────────────────────────────┐
│                      部署流程                                 │
└─────────────────────────────────────────────────────────────┘

1. Sim2Sim (Mujoco仿真验证)
   ├── 启动Mujoco仿真器
   ├── 加载ONNX模型
   ├── 初始化FSM控制器
   │   ├── State_Passive (被动状态)
   │   ├── State_FixStand (站立状态)
   │   └── State_RLBase (RL控制状态)
   └── 运行控制循环
       ├── 读取传感器数据
       ├── ONNX模型推理
       └── 发送关节命令

2. Sim2Real (真实机器人部署)
   ├── 连接机器人 (Unitree SDK2)
   ├── 加载ONNX模型
   ├── 初始化FSM控制器
   └── 运行控制循环 (1kHz)
       ├── 读取IMU和关节编码器
       ├── ONNX模型推理
       └── 发送关节命令
```

### FSM状态机设计

```
┌─────────────────────────────────────────────────────────────┐
│                    FSM状态转换                                │
└─────────────────────────────────────────────────────────────┘

State_Passive (被动状态)
    │
    │ [L2 + Up] 按键
    ▼
State_FixStand (固定站立)
    │
    │ 稳定后自动转换
    ▼
State_RLBase (RL控制状态)
    │
    │ [R1 + X] 开始运行策略
    │
    ├──> 正常运行
    │
    └──> 异常检测 → State_Passive
```

### State_RLBase 实现

```cpp
class State_RLBase : public BaseState {
    // 核心组件
    ├── ONNX推理引擎 (Ort::Session)
    ├── 观察管理器 (ObservationManager)
    ├── 动作管理器 (ActionManager)
    └── 终止条件检查 (TerminationManager)
    
    // 执行流程
    run() {
        1. 读取传感器数据
        2. 构建观察向量
        3. ONNX模型推理
        4. 后处理动作
        5. 发送关节命令
        6. 检查终止条件
    }
}
```

---

## 数据流

### 训练阶段数据流

```
┌─────────────┐
│  仿真环境    │
│  (IsaacLab)  │
└──────┬───────┘
       │ 状态、奖励、终止标志
       ▼
┌─────────────┐
│  观察计算    │
│ (Observations)│
└──────┬───────┘
       │ 观察向量
       ▼
┌─────────────┐
│  策略网络    │
│  (Actor)     │
└──────┬───────┘
       │ 动作
       ▼
┌─────────────┐
│  动作执行    │
│  (Actions)   │
└──────┬───────┘
       │ 关节目标
       ▼
┌─────────────┐
│  物理仿真    │
│  (Physics)   │
└──────┬───────┘
       │
       └──> (循环)
```

### 部署阶段数据流

```
┌─────────────┐
│  传感器      │
│ (IMU/编码器) │
└──────┬───────┘
       │ 原始数据
       ▼
┌─────────────┐
│  观察构建    │
│ (ObservationManager)│
└──────┬───────┘
       │ 观察向量
       ▼
┌─────────────┐
│  ONNX推理    │
│  (Ort::Session)│
└──────┬───────┘
       │ 动作
       ▼
┌─────────────┐
│  动作后处理  │
│ (ActionManager)│
└──────┬───────┘
       │ 关节命令
       ▼
┌─────────────┐
│  机器人执行  │
│  (Unitree SDK2)│
└─────────────┘
```

---

## 关键组件

### 1. 训练系统组件

| 组件 | 文件路径 | 功能 |
|------|---------|------|
| 训练脚本 | `scripts/rsl_rl/train.py` | 主训练入口，管理训练循环 |
| 推理脚本 | `scripts/rsl_rl/play.py` | 模型评估和可视化 |
| 环境配置 | `tasks/locomotion/robots/*/velocity_env_cfg.py` | 机器人特定环境配置 |
| PPO算法 | `tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | PPO超参数配置 |
| 观察空间 | `tasks/locomotion/mdp/observations.py` | 观察计算函数 |
| 奖励函数 | `tasks/locomotion/mdp/rewards.py` | 奖励计算函数 |
| 命令生成 | `tasks/locomotion/mdp/commands/velocity_command.py` | 速度命令生成 |
| 课程学习 | `tasks/locomotion/mdp/curriculums.py` | 自适应难度调整 |
| 配置导出 | `utils/export_deploy_cfg.py` | 导出部署配置和ONNX模型 |

### 2. 部署系统组件

| 组件 | 文件路径 | 功能 |
|------|---------|------|
| FSM控制器 | `deploy/include/FSM/CtrlFSM.h` | 状态机主控制器 |
| 状态基类 | `deploy/include/FSM/BaseState.h` | 状态接口定义 |
| RL状态 | `deploy/robots/*/src/State_RLBase.cpp` | RL控制状态实现 |
| 观察管理 | `deploy/include/isaaclab/manager/observation_manager.h` | 观察向量构建 |
| 动作管理 | `deploy/include/isaaclab/manager/action_manager.h` | 动作后处理 |
| 主程序 | `deploy/robots/*/main.cpp` | 部署程序入口 |

### 3. 机器人模型

| 组件 | 文件路径 | 功能 |
|------|---------|------|
| 模型加载 | `assets/robots/unitree.py` | USD/URDF模型加载 |
| 执行器配置 | `assets/robots/unitree_actuators.py` | 关节执行器参数 |

---

## 技术栈

### 训练系统

- **仿真引擎**: Isaac Sim 5.1.0 + Isaac Lab 2.3.0
- **强化学习**: RSL-RL (PPO算法)
- **深度学习**: PyTorch
- **配置管理**: Hydra + OmegaConf
- **Python版本**: 3.10+

### 部署系统

- **推理引擎**: ONNX Runtime 1.22.0
- **机器人SDK**: Unitree SDK2
- **状态机**: 自定义FSM框架
- **配置解析**: YAML-CPP
- **日志**: spdlog
- **C++标准**: C++17+

### 工具链

- **构建系统**: CMake
- **模型格式**: ONNX
- **配置文件**: YAML
- **容器化**: Docker

---

## 扩展指南

### 添加新机器人

1. **训练配置**:
   - 在 `tasks/locomotion/robots/` 创建新目录
   - 创建 `velocity_env_cfg.py` 配置文件
   - 调整观察、奖励、动作空间

2. **部署配置**:
   - 在 `deploy/robots/` 创建新目录
   - 实现 `State_RLBase.cpp`
   - 配置 `config.yaml` FSM设置

### 添加新任务

1. 在 `tasks/` 创建新任务目录
2. 实现MDP组件 (observations, rewards, commands)
3. 创建机器人配置
4. 注册到Gym环境注册表

### 添加新奖励项

1. 在 `mdp/rewards.py` 定义奖励函数
2. 在机器人配置的 `RewardsCfg` 中添加 `RewTerm`
3. 调整权重参数

---

## 总结

Unitree RL Lab 采用**模块化、可扩展**的架构设计：

- **训练系统**: 基于IsaacLab的高性能并行仿真，支持课程学习和域随机化
- **部署系统**: 基于FSM的实时控制框架，支持Sim2Sim和Sim2Real
- **数据流**: 清晰的训练→验证→部署流程
- **可扩展性**: 易于添加新机器人、新任务和新组件

整个框架实现了从仿真训练到真实部署的完整闭环，为机器人强化学习提供了完整的解决方案。

