#!/usr/bin/env python3
"""
Test script to check if the Python LunarLander custom reset is deterministic.
This runs the same test multiple times and compares the results.
"""

import sys
import os
import numpy as np
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunar_lander import custom_reset
import gymnasium as gym

def run_test():
    """Run a single test and return the first few step results."""
    env = gym.make("LunarLander-v3")
    obs, info, env = custom_reset(env)
    
    results = []
    for i in range(5):  # Just test first 5 steps
        action = 0  # Always use action 0 (no-op) for consistency
        obs, reward, done, truncated, info = env.step(action)
        results.append({
            "step": i,
            "observation": obs.tolist(),
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated)
        })
        if done:
            break
    
    env.close()
    return results

def main():
    print("Testing Python LunarLander determinism...")
    
    # Run the test multiple times
    num_runs = 3
    all_results = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        results = run_test()
        all_results.append(results)
    
    # Compare results
    print("\nComparing results:")
    
    if len(all_results) < 2:
        print("Need at least 2 runs to compare")
        return
    
    first_run = all_results[0]
    all_identical = True
    
    for run_idx in range(1, len(all_results)):
        current_run = all_results[run_idx]
        
        if len(first_run) != len(current_run):
            print(f"Run {run_idx + 1} has different number of steps: {len(current_run)} vs {len(first_run)}")
            all_identical = False
            continue
            
        for step_idx in range(len(first_run)):
            first_step = first_run[step_idx]
            current_step = current_run[step_idx]
            
            # Compare observations with tolerance
            obs1 = np.array(first_step["observation"])
            obs2 = np.array(current_step["observation"])
            
            if not np.allclose(obs1, obs2, rtol=1e-10, atol=1e-10):
                print(f"Step {step_idx} observation differs between run 1 and run {run_idx + 1}")
                print(f"  Run 1: {obs1}")
                print(f"  Run {run_idx + 1}: {obs2}")
                print(f"  Max diff: {np.max(np.abs(obs1 - obs2))}")
                all_identical = False
            
            # Compare rewards
            if abs(first_step["reward"] - current_step["reward"]) > 1e-10:
                print(f"Step {step_idx} reward differs: {first_step['reward']} vs {current_step['reward']}")
                all_identical = False
    
    if all_identical:
        print("✅ All runs produced identical results - Python implementation is deterministic!")
    else:
        print("❌ Results differ between runs - Python implementation is not fully deterministic")
    
    # Save detailed results for inspection
    with open("determinism_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to determinism_test_results.json")

if __name__ == "__main__":
    main()