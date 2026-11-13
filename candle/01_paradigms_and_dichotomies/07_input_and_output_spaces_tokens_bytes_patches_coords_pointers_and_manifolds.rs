//! Category 01 Topic 07: Input & Output Spaces: Tokens, Bytes, Patches, Coords, Pointers, and Manifolds.
//!
//! Provides native Candle implementations for:
//! 1. Continuous 6D Rotation Manifold Representation (Zhou et al. CVPR 2019):
//!    Gram-Schmidt orthogonalization mapping unconstrained $\mathbb{R}^6 \to SO(3)$.
//! 2. Pointer Mechanism (Vinyals et al. NeurIPS 2015 Pointer Networks):
//!    Content-based additive attention scoring over variable-length inputs in $\Delta_N$.
//! 3. Multi-Frequency Coordinate Positional Encoding (Mildenhall et al. ECCV 2020 NeRF):
//!    Continuous Fourier feature mapping for spatial coordinates.
//! 4. Linear Patch Embedding (Dosovitskiy et al. ICLR 2021 ViT):
//!    Linear projection of flattened grid patches.
//!
//! # Mathematical Invariants
//! - Manifold Orthogonality: For any 6D vector, the projected rotation matrix satisfies
//!   $\mathbf{R}^\top \mathbf{R} = \mathbf{I}_3$ and $\det(\mathbf{R}) = +1.0$ ($SO(3)$ Special Orthogonal group).
//! - Pointer Probability Simplex: $\sum_{j=1}^N p_j = 1.0$ and $p_j > 0$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use sml_candle::c01_paradigms_and_dichotomies::t07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds::rotation_6d_to_matrix;
//!
//! let device = Device::Cpu;
//! let d6 = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]], &device).unwrap();
//! let rot = rotation_6d_to_matrix(&d6).unwrap();
//! assert_eq!(rot.dims(), &[1, 3, 3]);
//! ```

use candle_core::{Result, Tensor};

/// Evaluates 3D cross product for batches of 3D vectors: $\mathbf{u} \times \mathbf{v}$.
pub fn cross_product_3d(u: &Tensor, v: &Tensor) -> Result<Tensor> {
    let ux = u.narrow(1, 0, 1)?;
    let uy = u.narrow(1, 1, 1)?;
    let uz = u.narrow(1, 2, 1)?;

    let vx = v.narrow(1, 0, 1)?;
    let vy = v.narrow(1, 1, 1)?;
    let vz = v.narrow(1, 2, 1)?;

    let wx = uy.mul(&vz)?.sub(&uz.mul(&vy)?)?;
    let wy = uz.mul(&vx)?.sub(&ux.mul(&vz)?)?;
    let wz = ux.mul(&vy)?.sub(&uy.mul(&vx)?)?;

    Tensor::cat(&[&wx, &wy, &wz], 1)
}

/// Continuous 6D rotation representation to 3x3 SO(3) rotation matrix via Gram-Schmidt.
pub fn rotation_6d_to_matrix(d6: &Tensor) -> Result<Tensor> {
    let a1 = d6.narrow(1, 0, 3)?;
    let a2 = d6.narrow(1, 3, 3)?;

    // 1. Normalize first vector: b1 = a1 / ||a1||_2
    let norm_a1 = (a1.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-16))?.sqrt()?;
    let b1 = a1.broadcast_div(&norm_a1)?;

    // 2. Gram-Schmidt orthogonal projection: u2 = a2 - (b1^T a2) b1
    let dot_prod = a2.mul(&b1)?.sum_keepdim(1)?;
    let proj = dot_prod.broadcast_mul(&b1)?;
    let u2 = a2.sub(&proj)?;
    let norm_u2 = (u2.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-16))?.sqrt()?;
    let b2 = u2.broadcast_div(&norm_u2)?;

    // 3. Cross product for third vector: b3 = b1 x b2
    let b3 = cross_product_3d(&b1, &b2)?;

    // Stack column-wise to form rotation matrix [B, 3, 3]
    Tensor::stack(&[&b1, &b2, &b3], 2)
}

/// Pointer Mechanism Layer.
#[derive(Debug, Clone)]
pub struct PointerNetworkLayer {
    pub w_enc: Tensor,
    pub w_dec: Tensor,
    pub b_ptr: Tensor,
    pub v_ptr: Tensor,
}

impl PointerNetworkLayer {
    pub fn new(w_enc: Tensor, w_dec: Tensor, b_ptr: Tensor, v_ptr: Tensor) -> Self {
        Self {
            w_enc,
            w_dec,
            b_ptr,
            v_ptr,
        }
    }

    /// Forward pass computing pointer logits and softmax probabilities.
    pub fn forward(
        &self,
        encoder_seq: &Tensor,
        decoder_state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, num_elem, d_model) = encoder_seq.dims3()?;
        let h_attn = self.w_enc.dim(1)?;

        // Broadcast 2D weight to 3D for batched matmul: [B, N, D] @ [B, D, H] -> [B, N, H]
        let w_enc_3d = self.w_enc.broadcast_as((batch_size, d_model, h_attn))?;
        let e_proj = encoder_seq.matmul(&w_enc_3d)?;

        // Decoder query projection: [B, D] @ [D, H] -> [B, H] -> [B, 1, H]
        let d_proj = decoder_state.matmul(&self.w_dec)?.unsqueeze(1)?;

        let scores = e_proj
            .broadcast_add(&d_proj)?
            .broadcast_add(&self.b_ptr)?
            .tanh()?;

        // Score with v_ptr: [B * N, H] @ [H, 1] -> [B * N, 1] -> [B, N]
        let scores_flat = scores.flatten(0, 1)?;
        let logits_flat = scores_flat.matmul(&self.v_ptr)?;
        let logits = logits_flat.reshape((batch_size, num_elem))?;

        let probs = candle_nn::ops::softmax(&logits, 1)?;
        Ok((logits, probs))
    }
}

/// Multi-Scale Sinusoidal Positional Encoding for Continuous Coordinates.
pub fn fourier_coordinate_encode(coords: &Tensor, num_bands: usize) -> Result<Tensor> {
    let (batch_size, d_coord) = coords.dims2()?;
    let coords_vec = coords.to_vec2::<f32>()?;

    let out_dim = d_coord * 2 * num_bands;
    let mut out = vec![0.0f32; batch_size * out_dim];

    let pi = std::f32::consts::PI;
    for i in 0..batch_size {
        let mut idx = 0;
        for d in 0..d_coord {
            let x = coords_vec[i][d];
            for k in 0..num_bands {
                let scale = (2.0f32.powi(k as i32)) * pi;
                let val = x * scale;
                out[i * out_dim + idx] = val.sin();
                out[i * out_dim + idx + 1] = val.cos();
                idx += 2;
            }
        }
    }

    Tensor::from_vec(out, (batch_size, out_dim), coords.device())
}

/// Linear Patch Embedding.
#[derive(Debug, Clone)]
pub struct PatchEmbedding {
    pub w_patch: Tensor,
    pub b_patch: Tensor,
}

impl PatchEmbedding {
    pub fn new(w_patch: Tensor, b_patch: Tensor) -> Self {
        Self { w_patch, b_patch }
    }

    /// Projects flattened patches into model dimension: [B, num_patches, patch_dim] -> [B, num_patches, d_model].
    pub fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        let (batch_size, _num_patches, patch_dim) = patches.dims3()?;
        let d_model = self.w_patch.dim(1)?;

        let w_patch_3d = self
            .w_patch
            .broadcast_as((batch_size, patch_dim, d_model))?;
        patches.matmul(&w_patch_3d)?.broadcast_add(&self.b_patch)
    }
}
