# Locomotion 任务程序流程与文件说明

## 📋 目录结构

```
locomotion/
├── agents/              # 强化学习算法配置
│   └── rsl_rl_ppo_cfg.py
├── mdp/                 # 马尔可夫决策过程组件
│   ├── commands/        # 命令生成
│   ├── curriculums.py   # 课程学习
│   ├── observations.py  # 观察空间
│   └── rewards.py       # 奖励函数
└── robots/              # 不同机器人的环境配置
    ├── go2/            # Go2 机器人配置
    ├── g1/             # G1 机器人配置
    └── h1/             # H1 机器人配置
```

---

## 🔄 程序执行流程

### 1. 初始化阶段

```
训练脚本 (train.py)
    ↓
加载环境配置 (robots/*/velocity_env_cfg.py)
    ↓
创建仿真环境 (RobotEnvCfg)
    ├── 场景设置 (RobotSceneCfg)
    │   ├── 地形生成 (terrain)
    │   ├── 机器人模型 (robot)
    │   └── 传感器配置 (height_scanner, contact_forces)
    ├── MDP 组件配置
    │   ├── 观察空间 (ObservationsCfg)
    │   ├── 动作空间 (ActionsCfg)
    │   ├── 命令生成 (CommandsCfg)
    │   ├── 奖励函数 (RewardsCfg)
    │   ├── 终止条件 (TerminationsCfg)
    │   ├── 事件处理 (EventCfg)
    │   └── 课程学习 (CurriculumCfg)
    └── 算法配置 (agents/rsl_rl_ppo_cfg.py)
```

### 2. 训练循环

```
每个训练迭代 (iteration):
    ├── 环境重置 (reset)
    │   ├── 事件处理 (EventCfg)
    │   │   ├── 随机化物理材质
    │   │   ├── 随机化机器人质量
    │   │   ├── 重置机器人位置和姿态
    │   │   └── 重置关节状态
    │   └── 生成新命令 (CommandsCfg)
    │
    ├── 数据收集阶段 (rollout)
    │   ├── 观察空间计算 (ObservationsCfg)
    │   │   ├── Policy 观察组 (给策略网络)
    │   │   └── Critic 观察组 (给价值网络)
    │   ├── 策略网络推理 → 动作
    │   ├── 执行动作 → 环境步进
    │   ├── 计算奖励 (RewardsCfg)
    │   ├── 检查终止条件 (TerminationsCfg)
    │   └── 课程学习更新 (CurriculumCfg)
    │
    └── 策略更新阶段 (update)
        ├── 计算优势函数
        ├── PPO 算法更新
        └── 保存检查点
```

### 3. 单步执行流程

```
环境步进 (step):
    ├── 1. 获取观察 (observations)
    │   ├── 基础状态: 角速度、重力投影、关节位置/速度
    │   ├── 命令: 目标速度命令
    │   └── 历史动作
    │
    ├── 2. 策略网络输出动作
    │   └── 关节位置目标
    │
    ├── 3. 物理仿真步进
    │   ├── 应用动作到关节
    │   ├── 更新传感器数据
    │   └── 计算接触力
    │
    ├── 4. 计算奖励
    │   ├── 任务奖励: 速度跟踪
    │   ├── 姿态奖励: 保持平衡
    │   ├── 能量惩罚: 关节力矩
    │   └── 步态奖励: 足部接触模式
    │
    ├── 5. 检查终止条件
    │   ├── 超时
    │   ├── 摔倒 (高度/姿态)
    │   └── 非法接触
    │
    └── 6. 课程学习更新
        ├── 地形难度递增
        └── 命令范围扩展
```

---

## 📁 文件详细说明

### 🤖 agents/ - 强化学习算法配置

#### `rsl_rl_ppo_cfg.py`
- **作用**: 定义 PPO (Proximal Policy Optimization) 算法的超参数配置
- **主要内容**:
  - 网络结构: Actor-Critic 网络层数和激活函数
  - 训练参数: 学习率、批次大小、更新次数
  - PPO 特定参数: clip 范围、熵系数、价值损失系数
  - 训练设置: 最大迭代次数、保存间隔

```python
# 关键配置项:
- num_steps_per_env: 每个环境收集的步数
- max_iterations: 最大训练迭代次数
- actor_hidden_dims: Actor 网络隐藏层维度
- critic_hidden_dims: Critic 网络隐藏层维度
- learning_rate: 学习率
- clip_param: PPO clip 参数
- gamma: 折扣因子
```

---

### 🎯 mdp/ - 马尔可夫决策过程组件

#### `observations.py`
- **作用**: 定义观察空间的计算函数
- **主要函数**:
  - `gait_phase()`: 计算步态相位（用于步态同步）
- **观察项**:
  - 基础角速度、重力投影
  - 速度命令
  - 关节相对位置和速度
  - 历史动作

#### `rewards.py`
- **作用**: 定义各种奖励和惩罚函数
- **奖励类别**:

  **1. 关节相关**:
  - `energy()`: 能量消耗惩罚（关节速度 × 力矩）
  - `stand_still()`: 静止站立奖励
  - `joint_position_penalty()`: 关节位置偏差惩罚

  **2. 机器人姿态**:
  - `orientation_l2()`: 姿态对齐奖励
  - `upward()`: 保持直立奖励

  **3. 足部相关**:
  - `feet_stumble()`: 足部碰撞惩罚
  - `feet_height_body()`: 足部抬升高度奖励
  - `foot_clearance_reward()`: 足部离地间隙奖励
  - `feet_too_near()`: 足部距离过近惩罚
  - `feet_contact_without_cmd()`: 无命令时足部接触奖励
  - `air_time_variance_penalty()`: 空中时间方差惩罚
  - `feet_gait()`: 步态模式奖励

  **4. 其他**:
  - `joint_mirror()`: 关节对称性奖励

#### `commands/velocity_command.py`
- **作用**: 定义速度命令生成配置
- **功能**:
  - 生成随机速度命令（线性速度 x/y，角速度 z）
  - 支持课程学习，逐步增加命令难度
  - 命令重采样时间间隔控制

#### `curriculums.py`
- **作用**: 实现课程学习策略
- **主要函数**:
  - `lin_vel_cmd_levels()`: 根据性能动态调整线性速度命令范围
  - `ang_vel_cmd_levels()`: 根据性能动态调整角速度命令范围
- **工作原理**:
  - 监控奖励性能
  - 当性能达到阈值时，扩大命令范围
  - 逐步增加任务难度

---

### 🤖 robots/ - 机器人环境配置

每个机器人目录包含 `velocity_env_cfg.py`，定义该机器人的完整训练环境配置。

#### 通用配置结构:

**1. RobotSceneCfg (场景配置)**
- 地形生成器配置
- 机器人模型加载
- 传感器配置:
  - `height_scanner`: 高度扫描器（用于感知地形）
  - `contact_forces`: 接触力传感器（用于检测足部接触）

**2. EventCfg (事件配置)**
- **启动时 (startup)**:
  - 随机化物理材质（摩擦系数、恢复系数）
  - 随机化机器人质量
- **重置时 (reset)**:
  - 重置机器人位置和姿态
  - 重置关节状态
  - 应用外部力/力矩
- **间隔 (interval)**:
  - 随机推动机器人（增加鲁棒性）

**3. CommandsCfg (命令配置)**
- 速度命令生成范围
- 命令重采样时间
- 课程学习限制范围

**4. ActionsCfg (动作配置)**
- 关节位置控制
- 动作缩放和限制

**5. ObservationsCfg (观察配置)**
- **PolicyCfg**: 策略网络观察（可能包含噪声）
  - 基础状态、命令、关节状态、历史动作
- **CriticCfg**: 价值网络观察（特权信息）
  - 更完整的状态信息（包括真实速度）

**6. RewardsCfg (奖励配置)**
- **任务奖励**: 速度跟踪奖励
- **基础奖励**: 姿态、能量、平滑性
- **足部奖励**: 步态、接触、抬升
- **惩罚项**: 非法接触、关节限制

**7. TerminationsCfg (终止配置)**
- 超时终止
- 摔倒终止（高度/姿态）
- 非法接触终止

**8. CurriculumCfg (课程学习配置)**
- 地形难度递增
- 命令范围扩展

#### 机器人特定配置:

**go2/velocity_env_cfg.py**
- 四足机器人 Go2 的配置
- 相对简单的步态控制

**g1/29dof/velocity_env_cfg.py**
- 人形机器人 G1 (29自由度) 的配置
- 包含手臂和腿部的完整控制
- 更复杂的步态和平衡控制

**h1/velocity_env_cfg.py**
- 人形机器人 H1 的配置
- 包含步态相位观察
- 针对人形机器人的特殊奖励设计

---

## 🔧 关键设计模式

### 1. 模块化设计
- 每个组件（观察、奖励、命令等）独立定义
- 便于修改和扩展

### 2. 配置类 (ConfigClass)
- 使用 `@configclass` 装饰器
- 支持 Hydra 配置管理
- 类型安全的配置

### 3. 课程学习
- 从简单任务开始（低速、平地）
- 逐步增加难度（高速、复杂地形）
- 自适应调整命令范围

### 4. 域随机化
- 物理参数随机化（摩擦、质量）
- 初始状态随机化
- 外部干扰（推动）

### 5. 特权学习
- Policy 使用带噪声的观察（模拟真实传感器）
- Critic 使用完整状态（加速训练）

---

## 🚀 使用流程

### 训练新模型

```bash
# 1. 选择机器人配置
# 例如: go2, g1/29dof, h1

# 2. 运行训练脚本
python scripts/rsl_rl/train.py \
    task=Unitree-Go2-Velocity-v0 \
    agent=rsl_rl_ppo_cfg \
    num_envs=4096 \
    max_iterations=50000
```

### 配置文件选择

训练脚本会根据任务名称自动加载对应的环境配置:
- `Unitree-Go2-Velocity-v0` → `robots/go2/velocity_env_cfg.py`
- `Unitree-G1-29Dof-Velocity-v0` → `robots/g1/29dof/velocity_env_cfg.py`
- `Unitree-H1-Velocity-v0` → `robots/h1/velocity_env_cfg.py`

---

## 📊 训练监控

训练过程中会记录:
- 奖励曲线
- 策略损失
- 价值损失
- 环境性能指标
- 检查点保存

日志保存在: `logs/rsl_rl/{experiment_name}/{timestamp}/`

---

## 🔍 调试建议

1. **观察空间**: 检查观察值是否在合理范围内
2. **奖励函数**: 监控各项奖励的贡献，调整权重
3. **课程学习**: 观察命令范围和地形难度的变化
4. **终止条件**: 检查过早终止的原因
5. **域随机化**: 调整随机化范围以平衡鲁棒性和训练效率

---

## 📝 扩展指南

### 添加新奖励项
1. 在 `mdp/rewards.py` 中定义新函数
2. 在对应机器人的 `RewardsCfg` 中添加 `RewTerm`

### 添加新观察项
1. 在 `mdp/observations.py` 中定义新函数
2. 在 `ObservationsCfg` 中添加 `ObsTerm`

### 添加新机器人
1. 在 `robots/` 下创建新目录
2. 复制现有配置并修改机器人模型
3. 调整奖励和观察以适应新机器人

---

## 🎓 总结

Locomotion 任务是一个完整的强化学习训练框架，通过模块化设计实现了:
- **灵活的配置系统**: 易于修改和扩展
- **课程学习**: 从简单到复杂的渐进式训练
- **域随机化**: 提高策略的鲁棒性
- **多机器人支持**: 统一的框架支持不同机器人

核心思想是通过奖励函数引导机器人学习稳定的步态和速度跟踪能力，同时通过课程学习和域随机化提高策略的泛化能力。

