#!/usr/bin/env python3
"""
Debug script to check the exact state immediately after custom reset
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunar_lander import custom_reset, custom_info
import gymnasium as gym

def debug_reset_state():
    """Check the exact state immediately after custom reset"""
    env = gym.make("LunarLander-v3")
    
    print("=== PYTHON DEBUG: After custom reset ===")
    obs, info, env = custom_reset(env)
    print(f"Custom reset obs: {obs}")
    reset_info = custom_info(env, {})
    print(f"After custom reset raw state:")
    print(f"  pos_x: {reset_info.get('raw_lander_pos_x', 'N/A')}")
    print(f"  pos_y: {reset_info.get('raw_lander_pos_y', 'N/A')}")
    print(f"  vel_x: {reset_info.get('raw_lander_vel_x', 'N/A')}")
    print(f"  vel_y: {reset_info.get('raw_lander_vel_y', 'N/A')}")
    print(f"  angle: {reset_info.get('raw_lander_angle', 'N/A')}")
    print(f"  angular_vel: {reset_info.get('raw_lander_angular_vel', 'N/A')}")
    
    print("\n=== EXPECTED VALUES ===")
    VIEWPORT_W = 600.0
    VIEWPORT_H = 400.0  
    SCALE = 30.0
    expected = {
        'pos_x': VIEWPORT_W / SCALE / 2.0,  # 10.0
        'pos_y': VIEWPORT_H / SCALE * 0.8,  # 10.666667
        'vel_x': 0.0,
        'vel_y': -1.0,
        'angle': 0.0,
        'angular_vel': 0.0
    }
    for key, val in expected.items():
        print(f"  {key}: {val}")
    
    print("\n=== PYTHON DEBUG: After one step (action=0) ===")
    obs, reward, done, truncated, info = env.step(0)  # No-op action
    print(f"After step obs: {obs}")
    step_info = custom_info(env, {})
    print(f"After step raw state:")
    print(f"  pos_x: {step_info.get('raw_lander_pos_x', 'N/A')}")
    print(f"  pos_y: {step_info.get('raw_lander_pos_y', 'N/A')}")
    print(f"  vel_x: {step_info.get('raw_lander_vel_x', 'N/A')}")
    print(f"  vel_y: {step_info.get('raw_lander_vel_y', 'N/A')}")
    print(f"  angle: {step_info.get('raw_lander_angle', 'N/A')}")
    print(f"  angular_vel: {step_info.get('raw_lander_angular_vel', 'N/A')}")
    print(f"  reward: {reward}")
    
    env.close()

if __name__ == "__main__":
    debug_reset_state()