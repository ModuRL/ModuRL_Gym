use std::{io, sync::Mutex};

use ale::{Ale, BundledRom};
use bon::bon;
use candle_core::Tensor;
use modurl::{
    gym::{Gym, StepInfo},
    spaces::Discrete,
};

pub enum AtariObsType {
    RAM,
    RGBScreen,
    GrayscaleScreen,
}

pub struct AtariGymInternal {
    ale: Ale,
    obs_type: AtariObsType,
    device: candle_core::Device,
}

pub enum AtariRom {
    Bundled(BundledRom),
    Path(String),
}

impl AtariGymInternal {
    pub fn new(
        rom: AtariRom,
        obs_type: AtariObsType,
        device: candle_core::Device,
    ) -> Result<Self, io::Error> {
        let mut ale = Ale::new();
        match rom {
            AtariRom::Bundled(bundled_rom) => ale.load_rom(bundled_rom)?,
            AtariRom::Path(path) => {
                let c_path = std::ffi::CString::new(path).unwrap();
                ale.load_rom_file(&c_path)
            }
        };

        Ok(Self {
            ale,
            obs_type,
            device,
        })
    }
}

impl AtariGymInternal {
    fn get_action_space(&mut self) -> Discrete {
        Discrete::new(self.ale.legal_action_set().len() as usize)
    }

    fn get_observation_space(
        &mut self,
    ) -> Result<Box<dyn modurl::spaces::Space<Error = candle_core::Error>>, candle_core::Error>
    {
        Ok(match self.obs_type {
            AtariObsType::RAM => Box::new(Discrete::new(self.ale.ram_size())),
            AtariObsType::RGBScreen => {
                let (width, height) = (self.ale.screen_width(), self.ale.screen_height());
                Box::new(modurl::spaces::BoxSpace::new(
                    Tensor::full(0.0, &[height as usize, width as usize, 3], &self.device)?,
                    Tensor::full(1.0, &[height as usize, width as usize, 3], &self.device)?,
                ))
            }
            AtariObsType::GrayscaleScreen => {
                let (width, height) = (self.ale.screen_width(), self.ale.screen_height());
                Box::new(modurl::spaces::BoxSpace::new(
                    Tensor::full(0.0, &[height as usize, width as usize, 1], &self.device)?,
                    Tensor::full(1.0, &[height as usize, width as usize, 1], &self.device)?,
                ))
            }
        })
    }

    fn get_state(&mut self) -> Result<Tensor, candle_core::Error> {
        match self.obs_type {
            AtariObsType::RAM => {
                let mut ram_vec = vec![0u8; self.ale.ram_size()];
                self.ale.get_ram(ram_vec.as_mut_slice());
                // we need to split each u8 into 8 bits
                let ram_bits: Vec<u8> = ram_vec
                    .iter()
                    .flat_map(|byte| {
                        (0..8)
                            .rev()
                            .map(move |i| if (byte >> i) & 1 == 1 { 1u8 } else { 0u8 })
                    })
                    .collect();
                Tensor::from_slice(&ram_bits, &[ram_bits.len()], &self.device)
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
            state,
            reward,
            done,
            truncated,
        };

        assert!(action < self.get_action_space().get_possible_values());
        let mapped_action = self.ale.legal_action_set()[action];
        reward = self.ale.act(mapped_action) as f32;
        done = self.ale.is_game_over();
        state = self.get_state()?;
        truncated = false;

        Ok(StepInfo {
            state,
            reward,
            done,
            truncated,
        })
    }
}

struct AtariGym {
    inner: Mutex<AtariGymInternal>,
}

#[bon]
impl AtariGym {
    #[builder]
    pub fn new(
        rom: AtariRom,
        obs_type: AtariObsType,
        device: candle_core::Device,
    ) -> Result<Self, io::Error> {
        let inner = AtariGymInternal::new(rom, obs_type, device)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }
}

impl Gym for AtariGym {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        let inner = &mut *self.inner.lock().unwrap();
        inner.ale.reset_game();
        inner.get_state()
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        let inner = &mut *self.inner.lock().unwrap();
        Box::new(inner.get_action_space())
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        let inner = &mut *self.inner.lock().unwrap();
        inner.get_observation_space().unwrap()
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let action = action.to_scalar::<u32>()? as usize;
        let inner = &mut *self.inner.lock().unwrap();
        inner.step_usize(action)
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
            .rom(AtariRom::Bundled(BundledRom::Pong))
            .obs_type(AtariObsType::RGBScreen)
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
            if step_info.done {
                break;
            }
        }
    }
}
