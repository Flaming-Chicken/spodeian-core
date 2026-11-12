//! Spodeian ML Runtime (`sml-runtime`)
//!
//! **Stability: Experimental** — API subject to change
//!
//! SafeTensors streaming loader, continuous flow ODE solvers (RK4, RK45, RK23, PCEC),
//! PyTorch 2 parity training suite (optimizers, schedulers, losses, GradScaler, EMA),
//! dynamic multi-horizon/quantile heads, RLS online adaptation, universal 12-precision quantization,
//! GGUF v2/v3 / legacy GGML conversion, and model surgery for Candle neural networks.

#![forbid(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::type_complexity,
    clippy::manual_div_ceil,
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::len_without_is_empty,
    clippy::field_reassign_with_default,
    clippy::derivable_impls,
    clippy::doc_lazy_continuation,
    clippy::manual_checked_ops
)]

pub mod adaptation;
pub mod ast;
pub mod conversion;
pub mod distribution;
pub mod factors;
pub mod format;
pub mod gnn;
pub mod heads;
pub mod loader;
pub mod logarithmic;
pub mod loss;
pub mod lr_scheduler;
pub mod optim;
pub mod optim_kernels;
pub mod optim_quantized;
pub mod precision;
pub mod quantization;
pub mod sampler;
pub mod solvers;
pub mod ssm;
pub mod stability;
pub mod surgery;
pub mod train;
pub mod trajectory;
pub mod umbral;

pub use candle_core;

#[doc = "**Stability: Experimental** — API subject to change"]
pub use adaptation::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use ast::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use distribution::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use factors::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use format::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use gnn::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use heads::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use loader::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use logarithmic::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use loss::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use lr_scheduler::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use optim::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use precision::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use quantization::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use sampler::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use solvers::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use ssm::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use stability::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use surgery::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use train::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use trajectory::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use umbral::*;
