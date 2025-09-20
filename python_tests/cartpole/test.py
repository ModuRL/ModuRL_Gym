import gymnasium as gym
import json
import os
import numpy as np

env = gym.make("CartPole-v1")
obs = env.reset(seed=123, options={"low": 0.0, "high": 0.0})

# Lists to store inputs and outputs
inputs = []
outputs = []

for _ in range(100):
    action = env.action_space.sample()

    inputs.append(int(action))
    
    obs, reward, done, truncated, info = env.step(action)
    outputs.append({
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated)
    })
    
    if done:
        obs = env.reset(seed=123, options={"low": 0.0, "high": 0.0})

env.close()

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save to JSON files in script directory
with open(os.path.join(script_dir, 'inputs.json'), 'w') as f:
    json.dump(inputs, f, indent=2)

with open(os.path.join(script_dir, 'output.json'), 'w') as f:
    json.dump(outputs, f, indent=2)