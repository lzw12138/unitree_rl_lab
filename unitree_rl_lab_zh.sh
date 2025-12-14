#!/usr/bin/env bash

# 设置 UNITREE_RL_LAB_PATH 环境变量，指向脚本所在目录的绝对路径
export UNITREE_RL_LAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 检查是否在 Conda 环境中
if ! [[ -z "${CONDA_PREFIX}" ]]; then
    # 如果处于 Conda 环境中，则使用 Conda 环境中的 Python 解释器
    python_exe=${CONDA_PREFIX}/bin/python
else
    # 如果未检测到 Conda 环境，报错提示用户先激活 Conda 环境
    echo "[Error] No conda environment activated. Please activate the conda environment first."
    # exit 1
    # 注：这里注释掉了退出命令，允许脚本继续执行，用户可以选择修复环境后继续
fi


# 任务环境名称自动补全函数
# 此函数配合 argcomplete 实现命令行参数的自动补全功能
_ut_rl_lab_python_argcomplete_wrapper() {
    local IFS=$'\013'  # 设置内部字段分隔符为垂直制表符
    local SUPPRESS_SPACE=0
    # 检查当前是否设置了 nospace 选项
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    # 通过 _ARGCOMPLETE 环境变量触发 argcomplete 自动补全
    # 将补全结果保存到 COMPREPLY 数组
    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}
# 为脚本注册自动补全功能
# -o nospace: 补全后不自动添加空格
# -F: 指定补全函数
complete -o nospace -F _ut_rl_lab_python_argcomplete_wrapper "./unitree_rl_lab.sh"


# 设置 Conda 环境的函数
# 在 Conda 环境激活时自动加载必要的环境变量
_ut_setup_conda_env() {
    # 复制自 isaaclab/_isaac_sim/setup_conda_env.sh
    # 在 conda activate.d 目录下创建 setenv.sh 脚本
    # 这个脚本会在每次激活 Conda 环境时自动执行
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# show icon if not running headless' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' \
        '# for unitree_rl_lab' \
        'source '${UNITREE_RL_LAB_PATH}'/unitree_rl_lab.sh' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    # 检查是否存在 _isaac_sim 目录，如果存在说明 Isaac Sim 二进制文件已安装
    # 需要设置 Conda 环境变量以加载这些二进制文件
    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # 将 Isaac Sim 的环境设置追加到 setenv.sh 文件中
        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi
}

# 根据传入的命令行参数执行不同的操作
case "$1" in
    -i|--install)
        # 安装模式: 安装 git lfs 和 unitree_rl_lab 包
        git lfs install  # 确保 git lfs 已安装
        # 以可编辑模式安装 unitree_rl_lab
        pip install -e ${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab/
        # 设置 Conda 环境
        _ut_setup_conda_env
        # 激活全局 Python argcomplete
        activate-global-python-argcomplete
        ;;
    -l|--list)
        # 列表模式: 列出可用的训练环境
        shift  # 移除已处理的选项
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/list_envs.py "$@"
        ;;
    -p|--play)
        # 游戏模式: 运行训练好的模型进行演示
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/play.py "$@"
        ;;
    -t|--train)
        # 训练模式: 启动强化学习训练（无头模式）
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py --headless "$@"
        ;;
    *) 
        # 未知选项或没有选项: 执行默认操作（可能是显示帮助信息）
        # 这里没有默认操作，可能是显示用法说明
        ;;
esac