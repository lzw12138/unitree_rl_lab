#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include <iomanip>
#include <unistd.h>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";
    std::cout << "\n=== 手柄数据监控 (每秒更新) ===\n";
    std::cout << "按 Ctrl+C 退出\n\n";

    while (true)
    {
        // // 更新手柄数据
        // FSMState::lowstate->update();
        // auto& joy = FSMState::lowstate->joystick;
        
        // // 检查手柄是否超时（无数据）
        // if (FSMState::lowstate->isTimeout()) {
        //     std::cout << "[超时] 手柄未检测到数据！\n";
        // } else {
        //     std::cout << "[正常] 手柄连接正常\n";
            
        //     // 打印按键状态
        //     std::cout << "按键: A=" << (joy.A.pressed ? "1" : "0") 
        //               << " B=" << (joy.B.pressed ? "1" : "0")
        //               << " X=" << (joy.X.pressed ? "1" : "0")
        //               << " Y=" << (joy.Y.pressed ? "1" : "0")
        //               << " LB=" << (joy.LB.pressed ? "1" : "0")
        //               << " RB=" << (joy.RB.pressed ? "1" : "0")
        //               << " LT=" << (joy.LT.pressed ? "1" : "0")
        //               << " RT=" << (joy.RT.pressed ? "1" : "0") << "\n";
            
        //     // 打印方向键
        //     std::cout << "方向: 上=" << (joy.up.pressed ? "1" : "0")
        //               << " 下=" << (joy.down.pressed ? "1" : "0")
        //               << " 左=" << (joy.left.pressed ? "1" : "0")
        //               << " 右=" << (joy.right.pressed ? "1" : "0") << "\n";
            
        //     // 打印摇杆值
        //     std::cout << "摇杆: LX=" << std::fixed << std::setprecision(3) << joy.lx()
        //               << " LY=" << joy.ly()
        //               << " RX=" << joy.rx()
        //               << " RY=" << joy.ry() << "\n";
            
        //     // 打印扳机值
        //     std::cout << "扳机: LT=" << joy.LT() << " RT=" << joy.RT() << "\n";
        // }
        
        // std::cout << "---\n";
        sleep(1); // 每秒更新一次
    }
    
    return 0;
}

