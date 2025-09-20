#![cfg(test)]

use candle_core::{Device, Tensor};
use modurl::gym::Gym;

#[derive(serde::Deserialize)]
struct ExpectedOutput {
    observation: Vec<f32>,
    reward: f32,
    done: bool,
    truncated: bool,
}

pub(crate) trait Testable {
    fn reset_deterministic(&mut self) -> Result<Tensor, candle_core::Error>;
    fn set_state(&mut self, state: Tensor);
}

pub(crate) fn test_gym_against_python<T, E>(folder: &str, mut env: T)
where
    T: Gym<Error = E> + Testable,
    E: std::fmt::Debug,
{
    // Read the JSON files
    let inputs_path = format!(
        "{}/python_tests/{}/inputs.json",
        env!("CARGO_MANIFEST_DIR"),
        folder
    );
    let outputs_path = format!(
        "{}/python_tests/{}/output.json",
        env!("CARGO_MANIFEST_DIR"),
        folder
    );

    let inputs_json = std::fs::read_to_string(inputs_path).expect("Failed to read inputs.json");
    let outputs_json = std::fs::read_to_string(outputs_path).expect("Failed to read output.json");

    let inputs: Vec<u32> = serde_json::from_str(&inputs_json).expect("Failed to parse inputs.json");
    let expected_outputs: Vec<ExpectedOutput> =
        serde_json::from_str(&outputs_json).expect("Failed to parse output.json");

    env.reset_deterministic()
        .expect("Failed to reset environment");

    for i in 0..inputs.len() {
        let action = inputs[i];
        let action_tensor = Tensor::from_vec(vec![action], vec![], &Device::Cpu)
            .expect("Failed to create action tensor");

        if i == 0 || expected_outputs[i - 1].done {
            env.reset_deterministic()
                .expect("Failed to reset environment");
        } else {
            let state_dim = expected_outputs[i - 1].observation.len();
            env.set_state(
                Tensor::from_vec(
                    expected_outputs[i - 1].observation.clone(),
                    vec![state_dim],
                    &Device::Cpu,
                )
                .expect("Failed to set state from previous output"),
            );
        }

        let step_info = env.step(action_tensor).expect("Failed to step environment");

        let expected = &expected_outputs[i];

        // Get the actual observation as a vector
        let actual_obs = step_info
            .state
            .to_vec1::<f32>()
            .expect("Failed to convert state to vector");

        if step_info.reward != expected.reward {
            panic!(
                "Mismatch at step {}: expected reward {}, got {}, expected obs {:?}, got {:?}",
                i, expected.reward, step_info.reward, expected.observation, actual_obs
            );
        }

        if step_info.done {
            assert!(expected.done, "Expected done to be true at step {}", i);
            // reset isn't done here we set it at the top of the loop
        } else {
            assert!(!expected.done, "Expected done to be false at step {}", i);
        }

        if step_info.truncated {
            assert!(
                expected.truncated,
                "Expected truncated to be true at step {}",
                i
            );
        } else {
            assert!(
                !expected.truncated,
                "Expected truncated to be false at step {}",
                i
            );
        }

        // verify observation matches expected (within a tolerance)
        let obs_len = expected.observation.len();
        for j in 0..obs_len {
            assert!(
                (actual_obs[j] - expected.observation[j]).abs() < 1e-4,
                "Mismatch at step {}, observation index {}: expected {}, got {}",
                i,
                j,
                expected.observation[j],
                actual_obs[j]
            );
        }
    }

    assert!(!inputs.is_empty(), "Inputs should not be empty");
    assert!(
        !expected_outputs.is_empty(),
        "Expected outputs should not be empty"
    );
    assert_eq!(
        inputs.len(),
        expected_outputs.len(),
        "Input and output lengths should match"
    );
}
