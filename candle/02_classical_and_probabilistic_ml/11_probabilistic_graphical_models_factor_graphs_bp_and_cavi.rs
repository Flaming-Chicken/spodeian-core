//! Probabilistic Graphical Models: Factor Graphs, Belief Propagation, and CAVI in Candle.
//!
//! Features:
//! - Exact Sum-Product Belief Propagation on tree factor graphs
//! - Max-Product MAP state decoding
//! - Damped Loopy Belief Propagation on cyclic graphs
//! - Coordinate Ascent Mean-Field Variational Inference (CAVI)

use candle_core::{Result, Tensor};

/// Evaluates exact marginals on 4-node tree via Sum-Product Belief Propagation.
pub fn sum_product_tree_bp(unary: &Tensor, edges: &Tensor) -> Result<Tensor> {
    let u: Vec<Vec<f32>> = unary.to_vec2()?;
    let e_flat: Vec<f32> = edges.flatten_all()?.to_vec1()?;
    let e = |idx: usize, r: usize, c: usize| -> f32 { e_flat[idx * 4 + r * 2 + c] };

    // Messages leaves -> root (1):
    // m_0_1(x1) = sum_x0 u[0][x0] * e[0][x0][x1]
    let m_0_1 = [
        u[0][0] * e(0, 0, 0) + u[0][1] * e(0, 1, 0),
        u[0][0] * e(0, 0, 1) + u[0][1] * e(0, 1, 1),
    ];

    // m_2_1(x1) = sum_x2 u[2][x2] * e[1][x1][x2]
    let m_2_1 = [
        e(1, 0, 0) * u[2][0] + e(1, 0, 1) * u[2][1],
        e(1, 1, 0) * u[2][0] + e(1, 1, 1) * u[2][1],
    ];

    // m_3_1(x1) = sum_x3 u[3][x3] * e[2][x1][x3]
    let m_3_1 = [
        e(2, 0, 0) * u[3][0] + e(2, 0, 1) * u[3][1],
        e(2, 1, 0) * u[3][0] + e(2, 1, 1) * u[3][1],
    ];

    // Belief at root 1:
    let b1_raw = [
        u[1][0] * m_0_1[0] * m_2_1[0] * m_3_1[0],
        u[1][1] * m_0_1[1] * m_2_1[1] * m_3_1[1],
    ];
    let z1 = b1_raw[0] + b1_raw[1];
    let b1 = [b1_raw[0] / z1, b1_raw[1] / z1];

    // Messages root 1 -> leaves:
    let prod_1_to_0 = [u[1][0] * m_2_1[0] * m_3_1[0], u[1][1] * m_2_1[1] * m_3_1[1]];
    let m_1_0 = [
        e(0, 0, 0) * prod_1_to_0[0] + e(0, 0, 1) * prod_1_to_0[1],
        e(0, 1, 0) * prod_1_to_0[0] + e(0, 1, 1) * prod_1_to_0[1],
    ];
    let b0_raw = [u[0][0] * m_1_0[0], u[0][1] * m_1_0[1]];
    let z0 = b0_raw[0] + b0_raw[1];
    let b0 = [b0_raw[0] / z0, b0_raw[1] / z0];

    let prod_1_to_2 = [u[1][0] * m_0_1[0] * m_3_1[0], u[1][1] * m_0_1[1] * m_3_1[1]];
    let m_1_2 = [
        prod_1_to_2[0] * e(1, 0, 0) + prod_1_to_2[1] * e(1, 1, 0),
        prod_1_to_2[0] * e(1, 0, 1) + prod_1_to_2[1] * e(1, 1, 1),
    ];
    let b2_raw = [u[2][0] * m_1_2[0], u[2][1] * m_1_2[1]];
    let z2 = b2_raw[0] + b2_raw[1];
    let b2 = [b2_raw[0] / z2, b2_raw[1] / z2];

    let prod_1_to_3 = [u[1][0] * m_0_1[0] * m_2_1[0], u[1][1] * m_0_1[1] * m_2_1[1]];
    let m_1_3 = [
        prod_1_to_3[0] * e(2, 0, 0) + prod_1_to_3[1] * e(2, 1, 0),
        prod_1_to_3[0] * e(2, 0, 1) + prod_1_to_3[1] * e(2, 1, 1),
    ];
    let b3_raw = [u[3][0] * m_1_3[0], u[3][1] * m_1_3[1]];
    let z3 = b3_raw[0] + b3_raw[1];
    let b3 = [b3_raw[0] / z3, b3_raw[1] / z3];

    Tensor::new(&[b0, b1, b2, b3], unary.device())
}

/// Computes MAP state configuration on tree using exact maximization.
pub fn max_product_tree_map(unary: &Tensor, edges: &Tensor) -> Result<Tensor> {
    let u: Vec<Vec<f32>> = unary.to_vec2()?;
    let e_flat: Vec<f32> = edges.flatten_all()?.to_vec1()?;
    let e = |idx: usize, r: usize, c: usize| -> f32 { e_flat[idx * 4 + r * 2 + c] };

    let mut best_config = vec![0i64; 4];
    let mut max_p = -1.0f32;

    for x0 in 0..2 {
        for x1 in 0..2 {
            for x2 in 0..2 {
                for x3 in 0..2 {
                    let p = u[0][x0]
                        * u[1][x1]
                        * u[2][x2]
                        * u[3][x3]
                        * e(0, x0, x1)
                        * e(1, x1, x2)
                        * e(2, x1, x3);
                    if p > max_p {
                        max_p = p;
                        best_config = vec![x0 as i64, x1 as i64, x2 as i64, x3 as i64];
                    }
                }
            }
        }
    }

    Tensor::from_vec(best_config, (4,), unary.device())
}

/// Evaluates Damped Loopy Belief Propagation on a 3-node cyclic graph.
pub fn damped_loopy_bp_triangle(
    unary: &Tensor,
    edges: &Tensor,
    alpha: f64,
    iters: usize,
) -> Result<Tensor> {
    let u: Vec<Vec<f32>> = unary.to_vec2()?;
    let e_flat: Vec<f32> = edges.flatten_all()?.to_vec1()?;
    let e = |idx: usize, r: usize, c: usize| -> f32 { e_flat[idx * 4 + r * 2 + c] };

    let mut m = vec![vec![0.5f32; 2]; 6];
    let a = alpha as f32;

    for _ in 0..iters {
        // 0 -> 1: (u[0] * m[4]) @ e[0]
        let p0 = [u[0][0] * m[4][0], u[0][1] * m[4][1]];
        let raw_0_1 = [
            p0[0] * e(0, 0, 0) + p0[1] * e(0, 1, 0),
            p0[0] * e(0, 0, 1) + p0[1] * e(0, 1, 1),
        ];
        let z_0_1 = raw_0_1[0] + raw_0_1[1];
        m[0][0] = (1.0 - a) * m[0][0] + a * (raw_0_1[0] / z_0_1);
        m[0][1] = (1.0 - a) * m[0][1] + a * (raw_0_1[1] / z_0_1);

        // 1 -> 0: e[0] @ (u[1] * m[3])
        let p1 = [u[1][0] * m[3][0], u[1][1] * m[3][1]];
        let raw_1_0 = [
            e(0, 0, 0) * p1[0] + e(0, 0, 1) * p1[1],
            e(0, 1, 0) * p1[0] + e(0, 1, 1) * p1[1],
        ];
        let z_1_0 = raw_1_0[0] + raw_1_0[1];
        m[1][0] = (1.0 - a) * m[1][0] + a * (raw_1_0[0] / z_1_0);
        m[1][1] = (1.0 - a) * m[1][1] + a * (raw_1_0[1] / z_1_0);

        // 1 -> 2: (u[1] * m[0]) @ e[1]
        let p1_2 = [u[1][0] * m[0][0], u[1][1] * m[0][1]];
        let raw_1_2 = [
            p1_2[0] * e(1, 0, 0) + p1_2[1] * e(1, 1, 0),
            p1_2[0] * e(1, 0, 1) + p1_2[1] * e(1, 1, 1),
        ];
        let z_1_2 = raw_1_2[0] + raw_1_2[1];
        m[2][0] = (1.0 - a) * m[2][0] + a * (raw_1_2[0] / z_1_2);
        m[2][1] = (1.0 - a) * m[2][1] + a * (raw_1_2[1] / z_1_2);

        // 2 -> 1: e[1] @ (u[2] * m[5])
        let p2 = [u[2][0] * m[5][0], u[2][1] * m[5][1]];
        let raw_2_1 = [
            e(1, 0, 0) * p2[0] + e(1, 0, 1) * p2[1],
            e(1, 1, 0) * p2[0] + e(1, 1, 1) * p2[1],
        ];
        let z_2_1 = raw_2_1[0] + raw_2_1[1];
        m[3][0] = (1.0 - a) * m[3][0] + a * (raw_2_1[0] / z_2_1);
        m[3][1] = (1.0 - a) * m[3][1] + a * (raw_2_1[1] / z_2_1);

        // 2 -> 0: (u[2] * m[2]) @ e[2]
        let p2_0 = [u[2][0] * m[2][0], u[2][1] * m[2][1]];
        let raw_2_0 = [
            p2_0[0] * e(2, 0, 0) + p2_0[1] * e(2, 1, 0),
            p2_0[0] * e(2, 0, 1) + p2_0[1] * e(2, 1, 1),
        ];
        let z_2_0 = raw_2_0[0] + raw_2_0[1];
        m[4][0] = (1.0 - a) * m[4][0] + a * (raw_2_0[0] / z_2_0);
        m[4][1] = (1.0 - a) * m[4][1] + a * (raw_2_0[1] / z_2_0);

        // 0 -> 2: e[2] @ (u[0] * m[1])
        let p0_2 = [u[0][0] * m[1][0], u[0][1] * m[1][1]];
        let raw_0_2 = [
            e(2, 0, 0) * p0_2[0] + e(2, 0, 1) * p0_2[1],
            e(2, 1, 0) * p0_2[0] + e(2, 1, 1) * p0_2[1],
        ];
        let z_0_2 = raw_0_2[0] + raw_0_2[1];
        m[5][0] = (1.0 - a) * m[5][0] + a * (raw_0_2[0] / z_0_2);
        m[5][1] = (1.0 - a) * m[5][1] + a * (raw_0_2[1] / z_0_2);
    }

    let b0_raw = [u[0][0] * m[1][0] * m[4][0], u[0][1] * m[1][1] * m[4][1]];
    let z0 = b0_raw[0] + b0_raw[1];
    let b0 = [b0_raw[0] / z0, b0_raw[1] / z0];

    let b1_raw = [u[1][0] * m[0][0] * m[3][0], u[1][1] * m[0][1] * m[3][1]];
    let z1 = b1_raw[0] + b1_raw[1];
    let b1 = [b1_raw[0] / z1, b1_raw[1] / z1];

    let b2_raw = [u[2][0] * m[2][0] * m[5][0], u[2][1] * m[2][1] * m[5][1]];
    let z2 = b2_raw[0] + b2_raw[1];
    let b2 = [b2_raw[0] / z2, b2_raw[1] / z2];

    Tensor::new(&[b0, b1, b2], unary.device())
}

/// Evaluates CAVI responsibilities and ELBO for Gaussian mixture.
pub fn cavi_gmm_step(x: &Tensor, mu: &Tensor, sigma_sq: f64) -> Result<(Tensor, Tensor)> {
    let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let mu_vec: Vec<f32> = mu.flatten_all()?.to_vec1()?;
    let n = x_vec.len();
    let k = mu_vec.len();
    let s2 = sigma_sq as f32;

    let mut r = vec![vec![0.0f32; k]; n];
    let mut total_elbo = 0.0f32;

    for i in 0..n {
        let mut log_p = vec![0.0f32; k];
        let mut max_lp = f32::NEG_INFINITY;
        for c in 0..k {
            let diff = x_vec[i] - mu_vec[c];
            log_p[c] = -0.5f32 * (diff * diff) / s2;
            if log_p[c] > max_lp {
                max_lp = log_p[c];
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..k {
            sum_exp += (log_p[c] - max_lp).exp();
        }
        for c in 0..k {
            let r_ic = (log_p[c] - max_lp).exp() / sum_exp;
            r[i][c] = r_ic;
            let log_r_ic = r_ic.max(1e-8).ln();
            total_elbo += r_ic * (log_p[c] - log_r_ic);
        }
    }

    let resp = Tensor::new(r, x.device())?;
    let elbo = Tensor::new(&[total_elbo], x.device())?;
    Ok((resp, elbo))
}
