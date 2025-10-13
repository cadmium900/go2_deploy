#pragma once

#include <array>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <deque>
#include <sstream>
#include <cmath>

#include <ATen/Context.h>  // for at::globalContext()
#include "torch/script.h"
#include "robot_interface.hpp"
#include "gamepad.hpp"
#include "cfg.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{
    class BasicUserController
    {
    public:
        BasicUserController() {}

        virtual void LoadParam(fs::path &param_folder) = 0;
        virtual void reset(BasicRobotInterface &robot_interface, Gamepad &gamepad) = 0;
        virtual void GetInput(BasicRobotInterface &robot_interface, Gamepad &gamepad) = 0;
        virtual void Calculate() = 0;

        virtual std::vector<float> GetLog() = 0;

        float dt;
        std::array<float, 12> start_pos;
        std::array<float, 12> stand_pos;
        std::array<float, 12> jpos_des;
    };

class RLController : public BasicUserController
{
    public:
        RLController():stand_kp(0.0), stand_kd(0.0), ctrl_kp(0.0), ctrl_kd(0.0)
         {
            // observation init to 0
            base_ang_vel.fill(0.0);
            projected_gravity.fill(0.);
            projected_gravity.at(2) = -1.0;
            cmd.fill(0.0);
            jpos_processed.fill(0.0);
            jvel.fill(0.0);
            actions.fill(0.0);
            clock_input.fill(0.0);
            gait_period = 0.0;
            frame_stack = 5;
            num_single_obs = 61; // length of single step observation
            lin_vel_scale = 2.0;
            ang_vel_scale = 0.25;
            dof_vel_scale = 0.05;
            num_gaits = 1;
        }

        /**
         * @brief 从json文件中读取参数
         * @param param_folder 参数文件夹路径
         * @note  读取参数文件rl_params.json, 包含以下参数:
         * @note  - dt: 控制周期
         * @note  - stand_kp: 站立状态 kp
         * @note  - stand_kd: 站立状态 kd
         * @note  - ctrl_kp: 控制状态 kp
         * @note  - ctrl_kd: 控制状态 kd
         * @note  - action_scale: 动作缩放系数
         * @note  - lin_vel_scale: 线速度缩放系数
         * @note  - ang_vel_scale: 角速度缩放系数
         * @note  - dof_vel_scale: 关节速度缩放系数
         * @note  - gait_time: 步态周期
         * @note  - stand_pos: 站立状态关节位置
         * @note  - swing_phase_ratio: 摆动相占比
        */
        void LoadParam(fs::path &param_folder)
        {
            // load param file
            std::ifstream cfg_file(param_folder / "rl_params.json");
            std::cout << "Read params from: " << (param_folder / "rl_params.json") << std::endl;
            std::stringstream ss;
            ss << cfg_file.rdbuf();
            FromJsonString(ss.str(), cfg);

            // get data from json
            device = cfg.device;
            dt = cfg.dt;
            stand_kp = cfg.stand_kp;
            stand_kd = cfg.stand_kd;
            action_scale = cfg.action_scale;
            lin_vel_scale = cfg.lin_vel_scale;
            ang_vel_scale = cfg.ang_vel_scale;
            dof_vel_scale = cfg.dof_vel_scale;
            policy_name = cfg.policy_name;
            ctrl_kp = cfg.ctrl_kp;
            ctrl_kd = cfg.ctrl_kd;
            frame_stack = cfg.frame_stack;
            num_single_obs = cfg.num_single_obs;
            num_gaits = cfg.num_gaits;
            for (int i = 0; i < 12; ++i)
            {
                stand_pos.at(i) = cfg.stand_pos.at(i);
                sit_pos.at(i) = cfg.sit_pos.at(i);
            }
            // Read behavior param range
            for (int i = 0; i < 2; ++i)
            {
                gait_period_range.at(i) = cfg.gait_period_range.at(i);
                base_height_target_range.at(i) = cfg.base_height_target_range.at(i);
                foot_clearance_target_range.at(i) = cfg.foot_clearance_target_range.at(i);
            }
            gait_period = gait_period_range.at(1);
            base_height_target = base_height_target_range.at(1);
            foot_clearance_target = foot_clearance_target_range.at(1);

            // initialize history observation buffer
            // prepare history buffers
            single_step_obs.resize(num_single_obs);
            single_step_obs.clear();
            for(int i = 0; i < num_single_obs; ++i)
            {
                single_step_obs.push_back(0.0);
            }
            history_obs.resize(frame_stack);
            for(int i = 0; i < frame_stack; ++i)
            {
                history_obs.at(i).resize(num_single_obs);
                history_obs.at(i) = single_step_obs;
            }

            arm_left_shoulder_lateral_target = 0.0;
            arm_left_shoulder_front_target   = 0.0;
            arm_left_elbow_target            = 0.0;
            arm_left_gripper_target          = 0.0;

            arm_right_shoulder_lateral_target = 0.0;
            arm_right_shoulder_front_target   = 0.0;
            arm_right_elbow_target            = 0.0;
            arm_right_gripper_target          = 0.0;
        }

        void loadPolicy()
        {
            fs::path model_path = fs::current_path() / "../models";

            // Prefer a persistent device member so all code uses the same device.
            if (device == "cpu")
                inference_device = torch::Device(torch::kCPU, 0);
            else
                inference_device = torch::Device(torch::kCUDA, 0);

            // Enable TF32 (Tensor Cores for FP32 matmul/conv) and let cuDNN pick fast kernels.
            at::globalContext().setAllowTF32CuBLAS(true);
            at::globalContext().setAllowTF32CuDNN(true);
            at::globalContext().setBenchmarkCuDNN(true);

            // Load directly onto CUDA in FP32
            policy = torch::jit::load((model_path / policy_name).string(), inference_device);
            policy.eval();

            std::cout << "Load policy from: " << (model_path / policy_name) << std::endl;

            // Warmup with a CUDA FP32 tensor to verify device/dtype path.
            c10::InferenceMode guard;
            auto policy_input_tensor = torch::zeros(
                {1, frame_stack * num_single_obs},
                torch::TensorOptions().dtype(torch::kFloat32).device(inference_device));

            auto out = policy.forward({policy_input_tensor}).toTensor();
            std::cout << "Forward OK. Output shape: " << out.sizes() << std::endl;
        }

        void GetInput(BasicRobotInterface &robot_interface, Gamepad &gamepad)
        {
            // save necessary data from input
            std::copy(robot_interface.gyro.begin(), robot_interface.gyro.end(), base_ang_vel.begin());
            std::copy(robot_interface.projected_gravity.begin(), robot_interface.projected_gravity.end(), projected_gravity.begin());

            // Control arms when select is pressed
            if (gamepad.select.pressed)
            {
                // Left Arm
                arm_left_shoulder_lateral_target -= gamepad.lx * 0.1f;
                arm_left_shoulder_front_target   -= gamepad.ly * 0.1f;
                if (gamepad.L2.pressed)
                    arm_left_elbow_target  += 0.1f;
                else if (gamepad.L1.pressed)
                    arm_left_elbow_target  -= 0.1f;

                arm_left_gripper_target          = 0.0f; //*** TODO

                // Limits
                arm_left_shoulder_lateral_target = std::min(std::max(arm_left_shoulder_lateral_target, cfg.fl_hip_limit[0]), cfg.fl_hip_limit[1]);
                arm_left_shoulder_front_target   = std::min(std::max(arm_left_shoulder_front_target,   cfg.fl_thigh_limit[0]), cfg.fl_thigh_limit[1]);
                arm_left_elbow_target            = std::min(std::max(arm_left_elbow_target,            cfg.fl_calf_limit[0]),  cfg.fl_calf_limit[1]);

                // Right Arm
                arm_right_shoulder_lateral_target -= gamepad.rx * 0.1f;
                arm_right_shoulder_front_target   -= gamepad.ry * 0.1f;
                if (gamepad.R2.pressed)
                    arm_right_elbow_target  += 0.1f;
                else if (gamepad.R1.pressed)
                    arm_right_elbow_target  -= 0.1f;
                arm_right_gripper_target          = 0.0f; //*** TODO
                // Limits
                arm_right_shoulder_lateral_target = std::min(std::max(arm_right_shoulder_lateral_target, cfg.fr_hip_limit[0]), cfg.fr_hip_limit[1]);
                arm_right_shoulder_front_target   = std::min(std::max(arm_right_shoulder_front_target,   cfg.fr_thigh_limit[0]), cfg.fr_thigh_limit[1]);
                arm_right_elbow_target            = std::min(std::max(arm_right_elbow_target,            cfg.fr_calf_limit[0]),  cfg.fr_calf_limit[1]);
            }
            else // Motion control
            {
                if (gamepad.left.pressed)
                {
                    gait_period += 0.005;
                    gait_period = std::min(gait_period, gait_period_range.at(1));
                    printf("Gait_Period to %f\n", gait_period);
                }
                else if (gamepad.right.pressed)
                {
                    gait_period -= 0.005;
                    gait_period = std::max(gait_period, gait_period_range.at(0));
                    printf("Gait_Period to %f\n", gait_period);
                }
                // Up and Down for base height target
                if (gamepad.up.pressed)
                {
                    base_height_target += 0.005;
                    base_height_target = std::min(base_height_target, base_height_target_range.at(1));
                    printf("Base_Height to %f\n", base_height_target);
                }
                else if (gamepad.down.pressed)
                {
                    base_height_target -= 0.005;
                    base_height_target = std::max(base_height_target, base_height_target_range.at(0));
                    printf("Base_Height to %f\n", base_height_target);
                }
                // L1 and L2 for foot clearance target
                if (gamepad.L1.pressed)
                {
                    foot_clearance_target += 0.005;
                    foot_clearance_target = std::min(foot_clearance_target, foot_clearance_target_range.at(1));
                    printf("Foot_Clearance to %f\n", foot_clearance_target);
                }
                else if (gamepad.L2.pressed)
                {
                    foot_clearance_target -= 0.005;
                    foot_clearance_target = std::max(foot_clearance_target, foot_clearance_target_range.at(0));
                    printf("Foot_Clearance to %f\n", foot_clearance_target);
                }

                // record command
                cmd.at(0) = gamepad.ly; // linear_x: [-1,1]
                cmd.at(1) = -gamepad.lx; // linear_y; [-1,1]
                cmd.at(2) = -gamepad.rx; // angular_z: [-1,1]
            }

            // record robot state
            for (int i = 0; i < 12; ++i)
            {
                jpos_processed.at(i) = robot_interface.jpos.at(i) - stand_pos.at(i);
                jvel.at(i) = robot_interface.jvel.at(i);
                tau.at(i) = robot_interface.tau.at(i);
            }
        }

        void reset(BasicRobotInterface &robot_interface, Gamepad &gamepad)
        {
            gait_time = 0.0;
            phi = 0.0;
            gait_period = gait_period_range.at(1);
            base_height_target = base_height_target_range.at(1);
            foot_clearance_target = foot_clearance_target_range.at(1);
            // reset history observation buffer
            for(int i = 0; i < num_single_obs; ++i)
            {
                single_step_obs.at(i) = 0.0;
            }
            for(int i = 0; i < frame_stack; ++i)
            {
                history_obs.at(i).resize(num_single_obs);
                history_obs.at(i) = single_step_obs;
            }
        }

        void DummyCalculate()
        {
            // warmup the neural network
            for(int i = 0; i < num_single_obs; ++i)
            {
                single_step_obs.at(i) = 0.0;
            }
            history_obs.pop_front();
            history_obs.push_back(single_step_obs);

            // Build FP32 input on CPU and move to CUDA (same device as policy)
            auto policy_input_tensor = build_policy_input_tensor();

            c10::InferenceMode guard;
            auto policy_output_tensor = policy.forward({policy_input_tensor}).toTensor();
            (void)policy_output_tensor;
        }

        void Calculate()
        {
            pre_process();
            // put current observation into history buffer
            fill_single_step_obs();
            history_obs.pop_front();
            history_obs.push_back(single_step_obs);

            // Build FP32 input on CPU and move to CUDA (same device as policy)
            auto policy_input_tensor = build_policy_input_tensor();

            c10::InferenceMode guard;
            auto policy_output_tensor = policy.forward({policy_input_tensor}).toTensor();

            std::array<float, 12> actions_scaled;
            for(int i = 0; i < 12; ++i)
            {
                actions.at(i) = policy_output_tensor[0][i].item<float>();
                actions_scaled.at(i) = actions.at(i) * action_scale;
                jpos_des.at(i) = stand_pos.at(i) + actions_scaled.at(i);
            }
        }

        std::vector<float> GetLog()
        {
            // record input, output and other info into a vector
            std::vector<float> log;
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(cmd.at(i));
            }
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(projected_gravity.at(i));
            }
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(base_ang_vel.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jpos_processed.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jvel.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(actions.at(i));
            }
            for (int i = 0; i < 4; i++)
            {
                log.push_back(clock_input.at(i));
            }
            log.push_back(gait_period);
            return log;
        }

        void save_jpos(BasicRobotInterface &robot_interface)
        {
            for (int i = 0; i < 12; ++i)
            {
                start_pos.at(i) = robot_interface.jpos.at(i);
            }
        }

        // cfg
        RLCfg cfg;

        std::array<float, 12> tau;
        std::array<float, 12> sit_pos; // sit状态最终位置

        // observation
        std::array<float, 3> base_ang_vel;
        std::array<float, 3> projected_gravity;
        std::array<float, 3> cmd;
        std::array<float, 12> jpos_processed;       //joint sequence: sim
        std::array<float, 12> jvel;
        std::array<float, 12> actions;
        std::array<float, 4> clock_input;

        float arm_left_shoulder_lateral_target;
        float arm_left_shoulder_front_target;
        float arm_left_elbow_target;
        float arm_left_gripper_target;

        float arm_right_shoulder_lateral_target;
        float arm_right_shoulder_front_target;
        float arm_right_elbow_target;
        float arm_right_gripper_target;

        float gait_period;
        float base_height_target;
        float foot_clearance_target;
        int frame_stack;
        int num_single_obs; // length of single step observation
        std::vector<float> single_step_obs; // 用于记录单步观测数据
        std::deque<std::vector<float>> history_obs; // 用于记录历史观测数据

        // normalization parameters
        float lin_vel_scale;
        float ang_vel_scale;
        float dof_vel_scale;
        // params
        float stand_kp;
        float stand_kd;
        float ctrl_kp;
        float ctrl_kd;
        float action_scale;
        // gait
        int num_gaits;
        std::array<float, 2> gait_period_range;
        std::array<float, 2> base_height_target_range;
        std::array<float, 2> foot_clearance_target_range;
        float gait_time;
        float phi;
        // NN model
        std::string policy_name;
        std::string device;
        torch::jit::script::Module policy;

    private:
        // All inference uses this device (kept consistent across calls).
        torch::Device inference_device{torch::kCUDA, 0};

        // Build a {1, frame_stack*num_single_obs} FP32 tensor on CPU from history,
        // then move to CUDA on the same device as the policy.
        torch::Tensor build_policy_input_tensor()
        {
            std::vector<float> flat(frame_stack * num_single_obs);
            for (int i = 0; i < frame_stack; ++i)
            {
                const auto &row = history_obs.at(i);
                std::copy(row.begin(), row.end(), flat.begin() + i * num_single_obs);
            }
            auto cpu = torch::from_blob(
                           flat.data(),
                           {1, frame_stack * num_single_obs},
                           torch::TensorOptions().dtype(torch::kFloat32))
                           .clone(); // own the memory
            return cpu.to(inference_device);
        }

        void calc_periodic_obs()
        {
            for (int i = 0; i < 4; i++)
            {
                //*** eventually just have 1 clock_input
                clock_input.at(i) = sin(2 * M_PI * (phi));
            }
        }

        void pre_process()
        {
            // gait time step
            gait_time += dt;
            if(gait_time > (gait_period - dt/2))
            {
                gait_time = 0.0;
            }
            phi = gait_time / gait_period;
            // calculate clock input
            calc_periodic_obs();
        }

        void fill_single_step_obs()
        {
            single_step_obs.at(0) = cmd.at(0) * lin_vel_scale;
            single_step_obs.at(1) = cmd.at(1) * lin_vel_scale;
            single_step_obs.at(2) = cmd.at(2) * ang_vel_scale;
            for(int i = 0; i < 3; ++i)
            {
                single_step_obs.at(i+3) = projected_gravity.at(i);
                single_step_obs.at(i+6) = base_ang_vel.at(i) * ang_vel_scale;
            }
            for(int i = 0; i < 12; ++i)
            {
                single_step_obs.at(i+9) = jpos_processed.at(i);
                single_step_obs.at(i+21) = jvel.at(i) * dof_vel_scale;
                single_step_obs.at(i+33) = actions.at(i);
            }
            for(int i = 0; i < 4; ++i)
            {
                single_step_obs.at(i+45) = clock_input.at(i);
            }
            single_step_obs.at(49) = gait_period;
            single_step_obs.at(50) = base_height_target;
            single_step_obs.at(51) = foot_clearance_target;
            single_step_obs.at(52) = 1.0; //pitch_target;

            single_step_obs.at(53) = arm_left_shoulder_lateral_target;
            single_step_obs.at(54) = arm_left_shoulder_front_target;
            single_step_obs.at(55) = arm_left_elbow_target;
            single_step_obs.at(56) = arm_left_gripper_target;

            single_step_obs.at(57) = arm_right_shoulder_lateral_target;
            single_step_obs.at(58) = arm_right_shoulder_front_target;
            single_step_obs.at(59) = arm_right_elbow_target;
            single_step_obs.at(60) = arm_right_gripper_target;
        }
    };
} // namespace unitree::common
