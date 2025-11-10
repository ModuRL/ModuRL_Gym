use std::io;

use ale::Ale;
use bon::{bon, builder};
use candle_core::Tensor;
use modurl::{
    gym::{Gym, StepInfo},
    spaces::{BoxSpace, Discrete},
};

pub use ale::BundledRom;

#[derive(Clone, Copy)]
pub enum AtariObsType {
    RAM,
    RGBScreen,
    GrayscaleScreen,
}

pub struct AtariGym {
    ale: Ale,
    obs_type: AtariObsType,
    device: candle_core::Device,
    observation_space: BoxSpace,
    action_space: Discrete,
    frame_skip: usize,
    lives: u32,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::rendering::Renderer>,
}

pub enum AtariRom {
    Bundled(BundledRom),
    Path(String),
}

#[derive(Debug)]
pub enum AtariGymError {
    IoError(io::Error),
    CandleError(candle_core::Error),
}

#[bon]
impl AtariGym {
    #[builder]
    pub fn new(
        rom: AtariRom,
        obs_type: AtariObsType,
        device: candle_core::Device,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, AtariGymError> {
        let mut ale = Ale::new();
        match rom {
            AtariRom::Bundled(bundled_rom) => {
                ale.load_rom(bundled_rom).map_err(AtariGymError::IoError)?
            }
            AtariRom::Path(path) => {
                let c_path = std::ffi::CString::new(path).unwrap();
                ale.load_rom_file(&c_path)
            }
        };

        let observation_space = Self::get_observation_space_initial(&mut ale, obs_type, &device)
            .map_err(AtariGymError::CandleError)?;
        let action_space = Self::get_action_space_initial(&mut ale);

        Ok(Self {
            lives: ale.lives() as u32,
            #[cfg(feature = "rendering")]
            renderer: if render {
                Some(crate::rendering::Renderer::new(
                    ale.screen_width(),
                    ale.screen_height(),
                    "Atari Gym Renderer",
                ))
            } else {
                None
            },
            ale,
            obs_type,
            device,
            observation_space,
            action_space,
            frame_skip: 4,
        })
    }
}

impl AtariGym {
    pub fn set_frame_skip(&mut self, frame_skip: usize) {
        assert!(frame_skip >= 1, "frame_skip must be at least 1");
        self.frame_skip = frame_skip;
    }

    fn get_action_space_initial(ale: &mut Ale) -> Discrete {
        Discrete::new(ale.minimal_action_set().len() as usize)
    }

    fn get_observation_space_initial(
        ale: &mut Ale,
        obs_type: AtariObsType,
        device: &candle_core::Device,
    ) -> Result<BoxSpace, candle_core::Error> {
        Ok(match obs_type {
            AtariObsType::RAM => BoxSpace::new(
                Tensor::full(0.0f32, &[ale.ram_size()], &device)?,
                Tensor::full(1.0f32, &[ale.ram_size()], &device)?,
            ),
            AtariObsType::RGBScreen => {
                let (width, height) = (ale.screen_width(), ale.screen_height());
                modurl::spaces::BoxSpace::new(
                    Tensor::full(0.0f32, &[height as usize, width as usize, 3], &device)?,
                    Tensor::full(1.0f32, &[height as usize, width as usize, 3], &device)?,
                )
            }
            AtariObsType::GrayscaleScreen => {
                let (width, height) = (ale.screen_width(), ale.screen_height());
                modurl::spaces::BoxSpace::new(
                    Tensor::full(0.0f32, &[height as usize, width as usize, 1], &device)?,
                    Tensor::full(1.0f32, &[height as usize, width as usize, 1], &device)?,
                )
            }
        })
    }

    fn get_state(&mut self) -> Result<Tensor, candle_core::Error> {
        match self.obs_type {
            AtariObsType::RAM => {
                let mut ram_vec = vec![0u8; self.ale.ram_size()];
                self.ale.get_ram(ram_vec.as_mut_slice());
                // normalize to 0..1
                let ram_bytes: Vec<f32> = ram_vec.iter().map(|&x| x as f32 / 255.0).collect();
                Tensor::from_slice(&ram_bytes, &[ram_bytes.len()], &self.device)
            }
            AtariObsType::RGBScreen => {
                let mut screen_vec =
                    vec![0u8; (self.ale.screen_width() * self.ale.screen_height() * 3) as usize];
                self.ale.get_screen_rgb(screen_vec.as_mut_slice());
                let (width, height) = (self.ale.screen_width(), self.ale.screen_height());
                // convert to f32 0..1
                let screen: Vec<f32> = screen_vec.iter().map(|&x| x as f32 / 255.0).collect();
                Tensor::from_slice(&screen, &[height as usize, width as usize, 3], &self.device)
            }
            AtariObsType::GrayscaleScreen => {
                let mut screen_vec =
                    vec![0u8; (self.ale.screen_width() * self.ale.screen_height()) as usize];
                self.ale.get_screen_grayscale(screen_vec.as_mut_slice());
                let (width, height) = (self.ale.screen_width(), self.ale.screen_height());
                // convert to f32 0..1
                let screen: Vec<f32> = screen_vec.iter().map(|&x| x as f32 / 255.0).collect();
                Tensor::from_slice(&screen, &[height as usize, width as usize], &self.device)
            }
        }
    }

    fn step_usize(&mut self, action: usize) -> Result<StepInfo, candle_core::Error> {
        let StepInfo {
            state: _,
            mut reward,
            done,
            truncated,
        };

        assert!(action < self.get_action_space().get_possible_values());
        let mapped_action = self.ale.minimal_action_set()[action];

        let mut state = None;
        reward = 0.0f32;
        for i in 0..self.frame_skip {
            reward += self.ale.act(mapped_action) as f32;

            // if this is one of the last 2 frames, get the state
            if i as i32 >= (self.frame_skip as i32) - 2 {
                match state {
                    None => {
                        state = Some(self.get_state()?);
                    }
                    Some(ref mut s) => {
                        let new_state = self.get_state()?;
                        *s = Tensor::maximum(s, &new_state)?;
                    }
                }
            }

            if self.ale.is_game_over() {
                break;
            }
        }
        done = self.ale.is_game_over();
        truncated = false;
        self.lives = self.ale.lives() as u32;

        if state.is_none() {
            state = Some(self.get_state()?);
        }

        #[cfg(feature = "rendering")]
        self.render();

        Ok(StepInfo {
            state: state.unwrap(),
            reward,
            done,
            truncated,
        })
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            let mut screen_vec =
                vec![0u8; (self.ale.screen_width() * self.ale.screen_height() * 3) as usize];
            self.ale.get_screen_rgb(screen_vec.as_mut_slice());
            renderer.set_buffer(
                &screen_vec
                    .chunks(3)
                    .map(|chunk| {
                        ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32)
                    })
                    .collect::<Vec<u32>>(),
            );
            renderer.present();
        }
    }

    fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    fn get_observation_space(&self) -> &BoxSpace {
        &self.observation_space
    }

    pub fn get_lives(&self) -> u32 {
        self.lives
    }
}

impl Gym for AtariGym {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.ale.reset_game();
        self.get_state()
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        Box::new(self.get_action_space().clone())
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        Box::new(self.get_observation_space().clone())
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let action = action.to_scalar::<u32>()? as usize;
        self.step_usize(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ale::BundledRom;
    use candle_core::Device;

    #[test]
    fn test_atari_gym() {
        let device = Device::Cpu;
        let mut gym = AtariGym::builder()
            .rom(AtariRom::Bundled(BundledRom::RiverRaid))
            .obs_type(AtariObsType::RAM)
            .device(device.clone())
            .build()
            .unwrap();

        let obs = gym.reset().unwrap();
        println!("Initial observation shape: {:?}", obs.shape());

        let action_space = gym.action_space();

        for _ in 0..10 {
            let action = action_space.sample(&device).unwrap();
            let step_info = gym.step(action).unwrap();
            println!(
                "Step info - Reward: {}, Done: {}",
                step_info.reward, step_info.done
            );
            let observation_space = gym.observation_space();
            println!(
                "Observation shape: {:?} vs expected: {:?}",
                step_info.state.shape(),
                observation_space.shape()
            );
            assert!(gym.observation_space().contains(&step_info.state));
            if step_info.done {
                break;
            }
        }
    }
}
