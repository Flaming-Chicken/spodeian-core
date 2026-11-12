//! Spodeian ML Utils
//!
//! **Stability: Experimental** — API subject to change
//!
//! SafeTensors streaming loader, dynamic multi-horizon/quantile heads, RLS online adaptation,
//! dual-speed TTA, L-BFGS second-order optimization, numerical parity verification,
//! streaming time-series statistics, and universal hardware device routing.

pub mod adaptation;
pub mod device;
pub mod factors;
pub mod heads;
pub mod integrator;
pub mod loader;
pub mod logarithmic;
pub mod optimization;
pub mod parity;
pub mod quantization;
pub mod stability;
pub mod streaming;

pub use candle_core;

#[doc = "**Stability: Experimental** — API subject to change"]
pub use adaptation::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use device::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use factors::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use heads::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use integrator::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use loader::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use logarithmic::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use optimization::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use parity::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use quantization::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use stability::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use streaming::*;
