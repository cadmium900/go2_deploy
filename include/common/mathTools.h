/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef MATHTOOLS_H
#define MATHTOOLS_H

#include <array>
#include <cmath>
#include <stdio.h>
#include <iostream>

/**
 * @brief wrap the angle to [-pi, pi]
 * @param angle the input angle
 * @return the wrapped angle
 */
float wrap_to_pi(float angle)
{
    while(angle > M_PI) angle -= 2*M_PI;
    while(angle < -M_PI) angle += 2*M_PI;
    return angle;
}

/**
 * @brief clip the value to the range [low_limit, high_limit]
 * @param value the input value
 * @param low_limit the lower limit
 * @param high_limit the upper limit
 * @return the clipped value
 */
float clip(float value, float low_limit, float high_limit)
{
    if(value <= low_limit) return low_limit;
    if(value >= high_limit) return high_limit;
    return value;
}

template<typename T1, typename T2>
inline T1 max(const T1 a, const T2 b){
	return (a > b ? a : b);
}

template<typename T1, typename T2>
inline T1 min(const T1 a, const T2 b){
	return (a < b ? a : b);
}

template<size_t N>
float max_abs(const std::array<float, N> &arr)
{
    float max_abs_val = 0.0f;
    for (size_t i = 0; i < N; i++)
    {
        const float v = std::fabs(arr[i]);
        if (v > max_abs_val)
        {
            max_abs_val = v;
        }
    }
    return max_abs_val;
}

#endif  // MATHTOOLS_H
