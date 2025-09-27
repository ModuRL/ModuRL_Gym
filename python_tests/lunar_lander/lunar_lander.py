import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_gym import test_gym

# Constants from Rust implementation
VIEWPORT_W = 600.0
VIEWPORT_H = 400.0  
SCALE = 30.0

# Deterministic initial state matching the Rust implementation
INITIAL = np.array([
    VIEWPORT_W / SCALE / 2.0,  # pos_x: center horizontally (10.0)
    VIEWPORT_H / SCALE * 0.8,  # pos_y: high up for landing (10.666...)
    0.0,                       # vel_x: no horizontal velocity
    -1.0,                      # vel_y: slight downward velocity
    0.0,                       # angle: upright
    0.0,                       # angular_vel: not rotating
    0.0,                       # leg1_contact: not touching
    0.0,                       # leg2_contact: not touching
])

def custom_reset(env):
    """
    Custom reset function that sets the lunar lander to a deterministic initial state
    matching the Rust implementation's reset_deterministic function.
    """
    # First do a normal reset to initialize the world structure
    obs, info = env.reset(seed=42, options={})
    
    unwrapped = env.unwrapped
    
    # Override terrain to match Rust deterministic terrain
    # In Rust, we set all terrain heights to h/8.0 = (VIEWPORT_H/SCALE)/8.0
    h = VIEWPORT_H / SCALE
    fixed_terrain_height = h / 8.0
    
    # Try to override the terrain heights if accessible
    if hasattr(unwrapped, 'terrain') and hasattr(unwrapped, 'helipad_y'):
        # Set helipad_y to match Rust: h/4.0
        unwrapped.helipad_y = h / 4.0
        
        # If we can access the terrain generation, override it
        if hasattr(unwrapped, '_generate_terrain'):
            print("Found terrain generation method - attempting to override")
    
    # Override the physics state to match our deterministic values
    
    # Set lander state
    if hasattr(unwrapped, 'lander') and unwrapped.lander is not None:
        lander = unwrapped.lander
        
        # Set position and angle (raw physics coordinates)
        lander.position = (INITIAL[0], INITIAL[1])
        lander.angle = INITIAL[4]
        
        # Set velocities
        lander.linearVelocity = (INITIAL[2], INITIAL[3])
        lander.angularVelocity = INITIAL[5]
        
        # Reset any applied forces/impulses
        lander.linearDamping = 0.0
        lander.angularDamping = 0.0
        
        # Try to clear any forces that might have been applied during reset
        if hasattr(lander, 'ApplyForceToCenter'):
            # Reset the force accumulator if possible
            pass
        
        # Wake up the body to ensure changes take effect
        if hasattr(lander, 'SetAwake'):
            lander.SetAwake(True)
        elif hasattr(lander, 'awake'):
            lander.awake = True
    
    # Set leg states if accessible
    if hasattr(unwrapped, 'legs') and unwrapped.legs:
        LEG_AWAY = 20.0 / SCALE
        LEG_DOWN = 18.0 / SCALE
        
        for i, leg in enumerate(unwrapped.legs):
            if leg is not None:
                # Position legs relative to lander (matching Rust implementation)
                i_f = -1.0 if i == 0 else 1.0
                leg_x = INITIAL[0] - i_f * LEG_AWAY
                leg_y = INITIAL[1] - LEG_DOWN
                
                leg.position = (leg_x, leg_y)
                leg.angle = INITIAL[4] + i_f * 0.05
                leg.linearVelocity = (INITIAL[2], INITIAL[3])
                leg.angularVelocity = INITIAL[5]
    
    # Reset any wind state if present
    if hasattr(unwrapped, 'wind_idx'):
        unwrapped.wind_idx = 0
    if hasattr(unwrapped, 'torque_idx'):
        unwrapped.torque_idx = 0
    
    # Reset game state
    if hasattr(unwrapped, 'game_over'):
        unwrapped.game_over = False
    if hasattr(unwrapped, 'prev_shaping'):
        unwrapped.prev_shaping = None
    
    # Get the updated observation
    try:
        if hasattr(unwrapped, '_get_obs'):
            obs = unwrapped._get_obs()
        elif hasattr(unwrapped, 'get_observation'):
            obs = unwrapped.get_observation()
        else:
            # If we can't get updated obs, construct it manually
            # This matches the Rust normalization logic
            helipad_y = getattr(unwrapped, 'helipad_y', VIEWPORT_H / SCALE / 4.0)
            leg_down = 18.0 / SCALE
            
            obs = np.array([
                (INITIAL[0] - VIEWPORT_W / SCALE / 2.0) / (VIEWPORT_W / SCALE / 2.0),
                (INITIAL[1] - (helipad_y + leg_down)) / (VIEWPORT_H / SCALE / 2.0),
                INITIAL[2] * (VIEWPORT_W / SCALE / 2.0) / 50.0,  # FPS = 50
                INITIAL[3] * (VIEWPORT_H / SCALE / 2.0) / 50.0,
                INITIAL[4],
                20.0 * INITIAL[5] / 50.0,
                INITIAL[6],
                INITIAL[7]
            ], dtype=np.float32)
    except Exception as e:
        print(f"Warning: Could not get updated observation: {e}")
    
    return obs, info, env

def custom_info(env, info, obs=None):
    """
    Custom info function that extracts raw physics state from the lunar lander environment.
    This helps us verify that the underlying physics state matches our expectations.
    """
    unwrapped = env.unwrapped
    custom_info_dict = {}
    
    # Get lander raw physics state
    if hasattr(unwrapped, 'lander') and unwrapped.lander is not None:
        lander = unwrapped.lander
        
        # Raw position and angle
        pos = lander.position
        custom_info_dict['raw_lander_pos_x'] = float(pos[0])
        custom_info_dict['raw_lander_pos_y'] = float(pos[1])
        custom_info_dict['raw_lander_angle'] = float(lander.angle)
        
        # Raw velocities
        vel = lander.linearVelocity
        custom_info_dict['raw_lander_vel_x'] = float(vel[0])
        custom_info_dict['raw_lander_vel_y'] = float(vel[1])
        custom_info_dict['raw_lander_angular_vel'] = float(lander.angularVelocity)
        
        # Lander awake status
        custom_info_dict['lander_awake'] = bool(lander.awake) if hasattr(lander, 'awake') else True
    
    # Get leg states
    if hasattr(unwrapped, 'legs') and unwrapped.legs:
        for i, leg in enumerate(unwrapped.legs):
            if leg is not None:
                pos = leg.position
                vel = leg.linearVelocity
                custom_info_dict[f'raw_leg{i}_pos_x'] = float(pos[0])
                custom_info_dict[f'raw_leg{i}_pos_y'] = float(pos[1])
                custom_info_dict[f'raw_leg{i}_angle'] = float(leg.angle)
                custom_info_dict[f'raw_leg{i}_vel_x'] = float(vel[0])
                custom_info_dict[f'raw_leg{i}_vel_y'] = float(vel[1])
                custom_info_dict[f'raw_leg{i}_angular_vel'] = float(leg.angularVelocity)
    
    # Extract leg contact state from the observation passed in or try to get current observation
    if obs is not None and len(obs) >= 8:
        # Use the observation that was passed in (most reliable)
        leg0_contact = float(obs[6])
        leg1_contact = float(obs[7])
        custom_info_dict['leg0_contact'] = leg0_contact
        custom_info_dict['leg1_contact'] = leg1_contact
    else:
        # Fallback: try to get observation from environment
        try:
            current_obs = None
            # Try multiple ways to get the current observation
            if hasattr(unwrapped, '_get_obs'):
                current_obs = unwrapped._get_obs()
            elif hasattr(unwrapped, 'get_observation'):
                current_obs = unwrapped.get_observation()
            
            if current_obs is not None and len(current_obs) >= 8:
                leg0_contact = float(current_obs[6])
                leg1_contact = float(current_obs[7])
                custom_info_dict['leg0_contact'] = leg0_contact
                custom_info_dict['leg1_contact'] = leg1_contact
                
                if leg0_contact > 0.0 or leg1_contact > 0.0:
                    print(f"DEBUG: Contact from _get_obs! leg0={leg0_contact}, leg1={leg1_contact}")
            else:
                # Final fallback: direct leg contact detection
                if hasattr(unwrapped, 'legs') and unwrapped.legs:
                    leg0_contact = 1.0 if hasattr(unwrapped.legs[0], 'ground_contact') and unwrapped.legs[0].ground_contact else 0.0
                    leg1_contact = 1.0 if hasattr(unwrapped.legs[1], 'ground_contact') and unwrapped.legs[1].ground_contact else 0.0
                    custom_info_dict['leg0_contact'] = leg0_contact
                    custom_info_dict['leg1_contact'] = leg1_contact
                    
                    if leg0_contact > 0.0 or leg1_contact > 0.0:
                        print(f"DEBUG: Contact from direct leg check! leg0={leg0_contact}, leg1={leg1_contact}")
                else:
                    custom_info_dict['leg0_contact'] = 0.0
                    custom_info_dict['leg1_contact'] = 0.0
        except Exception as e:
            print(f"DEBUG: Exception getting contact info: {e}")
            custom_info_dict['leg0_contact'] = 0.0
            custom_info_dict['leg1_contact'] = 0.0
    
    # Get environment state
    if hasattr(unwrapped, 'helipad_y'):
        custom_info_dict['helipad_y'] = float(unwrapped.helipad_y)
    if hasattr(unwrapped, 'helipad_x1'):
        custom_info_dict['helipad_x1'] = float(unwrapped.helipad_x1)
    if hasattr(unwrapped, 'helipad_x2'):
        custom_info_dict['helipad_x2'] = float(unwrapped.helipad_x2)
    
    # Get wind state if present
    if hasattr(unwrapped, 'wind_idx'):
        custom_info_dict['wind_idx'] = int(unwrapped.wind_idx)
    if hasattr(unwrapped, 'torque_idx'):
        custom_info_dict['torque_idx'] = int(unwrapped.torque_idx)
    
    # Get game state
    if hasattr(unwrapped, 'game_over'):
        custom_info_dict['game_over'] = bool(unwrapped.game_over)
    if hasattr(unwrapped, 'prev_shaping'):
        custom_info_dict['prev_shaping'] = float(unwrapped.prev_shaping) if unwrapped.prev_shaping is not None else None
    
    # Add normalization constants for reference
    custom_info_dict['VIEWPORT_W'] = VIEWPORT_W
    custom_info_dict['VIEWPORT_H'] = VIEWPORT_H
    custom_info_dict['SCALE'] = SCALE
    
    # Merge with original info
    if isinstance(info, dict):
        info.update(custom_info_dict)
        return info
    else:
        return custom_info_dict

if __name__ == "__main__":
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_gym("LunarLander-v3", output_dir=current_dir, custom_reset=custom_reset, custom_info=custom_info)
