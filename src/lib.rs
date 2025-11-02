use std::marker::PhantomData;

#[cfg(feature = "atari")]
pub mod atari;

pub mod box_2d;
pub mod classic_control;
pub(crate) mod testing;

#[cfg(feature = "rendering")]
pub(crate) mod rendering;

// Rendering components are not Send or Sync
// So we use this to make sure that even without rendering enabled the types are still not Send or Sync
struct PhantonUnsendsync(PhantomData<*const ()>);
