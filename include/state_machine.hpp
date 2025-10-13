#pragma once

#include "user_controller.hpp"
#include <algorithm>

namespace unitree::common
{
    enum class STATES
    {
        DAMPING = 0,
        SIT = 1,
        STAND = 2,
        CTRL = 3
    };

    class SimpleStateMachine
    {
    public:
        SimpleStateMachine() : state(STATES::DAMPING) {}

        bool Stop()
        {
            state = STATES::DAMPING;
            return true;
        }

        bool Stand() // stand state, only
        {
            if (state == STATES::DAMPING)
            {
                state = STATES::STAND;
                standing_count = 0; // 进入站立状态后重置计数
                return true;
            }
            else // if state is not DAMPING
            {
                return false;
            }
        }

        bool Ctrl() // control state, only in stand state can you enter this state
        {
            if (state == STATES::STAND)
            {
                state = STATES::CTRL;
                return true;
            }
            else
            {
                return false;
            }
        }

        void Standing(RLController& ctrl)
        {
            standing_percent = (float)standing_count / standing_duration;
            for (int i = 0; i < 12; i++)
            {
                ctrl.jpos_des.at(i) = (1 - standing_percent)*ctrl.start_pos.at(i) + standing_percent*ctrl.stand_pos.at(i);
            }
            standing_count++;
            standing_count = standing_count > standing_duration? standing_duration : standing_count;
        }

        STATES state;
    protected:
        int   standing_count    = 0;
        float standing_percent  = 0.0;
        int   standing_duration = 50;

    };

    class RLStateMachine: public SimpleStateMachine
    {
    public:
        bool Sit()
        {
            if(state == STATES::DAMPING)
            {
                state = STATES::SIT;
                sitting_count = 0;
                return true;
            }
            else
            {
                return false;
            }
        }

        // Stand state, only in sit state can you enter this state
        bool Stand()
        {
            if (state == STATES::SIT)
            {
                state = STATES::STAND;
                standing_count = 0;
                return true;
            }
            else // if state is not SIT
            {
                return false;
            }
        }

        void Sitting(RLController& ctrl)
        {
            sitting_percent = (float)sitting_count / sitting_duration;
            for (int i = 0; i < 12; i++)
            {
                ctrl.jpos_des.at(i) = (1 - sitting_percent)*ctrl.start_pos.at(i) + sitting_percent*ctrl.sit_pos.at(i);
            }
            sitting_count++;
            sitting_count = sitting_count > sitting_duration? sitting_duration : sitting_count;
        }

    protected:
        int   sitting_count    = 0;
        float sitting_percent  = 0.0;
        int   sitting_duration = 50;
    };
} // namespace unitree
