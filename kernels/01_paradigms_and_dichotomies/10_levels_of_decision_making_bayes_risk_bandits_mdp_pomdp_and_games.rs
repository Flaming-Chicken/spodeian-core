//! Dual-engine WebGPU + CPU Hardware Kernels for Category 01 Topic 10:
//! Levels of Decision Making: Bayes Risk, Bandits, MDP, POMDP, and Games.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL shader source for decision operations.
pub const DECISION_HIERARCHY_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> cost_matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_risk: array<f32>;

struct Uniforms {
    batch: u32,
    num_classes: u32,
    num_actions: u32,
    c_reject: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.batch) {
        return;
    }

    for (var a: u32 = 0u; a < params.num_actions; a = a + 1u) {
        var r: f32 = 0.0;
        for (var c: u32 = 0u; c < params.num_classes; c = c + 1u) {
            let p_val = probs[idx * params.num_classes + c];
            let cost = cost_matrix[a * params.num_classes + c];
            r = r + p_val * cost;
        }
        out_risk[idx * params.num_actions + a] = r;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderUniforms {
    batch: u32,
    num_classes: u32,
    num_actions: u32,
    c_reject: f32,
}

/// Level 0: Static Bayes decision rule on CPU.
pub fn bayes_decision_rule_cpu(
    probs: &[f32],
    cost_matrix: &[f32],
    batch: usize,
    num_classes: usize,
    num_actions: usize,
    c_reject: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut expected_risk = vec![0.0f32; batch * num_actions];
    let mut best_actions = vec![0.0f32; batch];
    let mut reject_mask = vec![0.0f32; batch];

    for b in 0..batch {
        let mut min_r = f32::INFINITY;
        let mut best_a = 0usize;

        for a in 0..num_actions {
            let mut r = 0.0f32;
            for c in 0..num_classes {
                let p_val = probs[b * num_classes + c];
                let cost = cost_matrix[a * num_classes + c];
                r += p_val * cost;
            }
            expected_risk[b * num_actions + a] = r;
            if r < min_r {
                min_r = r;
                best_a = a;
            }
        }
        best_actions[b] = best_a as f32;
        reject_mask[b] = if min_r > c_reject { 1.0f32 } else { 0.0f32 };
    }
    (expected_risk, best_actions, reject_mask)
}

/// Level 1: Contextual bandits OPE on CPU.
pub fn bandits_ope_cpu(
    target_probs: &[f32],
    logged_propensities: &[f32],
    rewards: &[f32],
    baseline_q: &[f32],
    batch: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut ips_weights = vec![0.0f32; batch];
    let mut doubly_robust = vec![0.0f32; batch];

    for i in 0..batch {
        let w = target_probs[i] / logged_propensities[i];
        ips_weights[i] = w;
        doubly_robust[i] = baseline_q[i] + w * (rewards[i] - baseline_q[i]);
    }
    (ips_weights, doubly_robust)
}

/// Level 2: Closed-loop MDP Bellman backup on CPU.
pub fn mdp_bellman_cpu(
    v_current: &[f32],
    transitions: &[f32],
    rewards: &[f32],
    num_states: usize,
    num_actions: usize,
    gamma: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut q_values = vec![0.0f32; num_states * num_actions];
    let mut v_next = vec![0.0f32; num_states];

    for s in 0..num_states {
        let mut max_q = f32::NEG_INFINITY;
        for a in 0..num_actions {
            let mut future_v = 0.0f32;
            for k in 0..num_states {
                let prob = transitions[s * (num_actions * num_states) + a * num_states + k];
                future_v += prob * v_current[k];
            }
            let q = rewards[s * num_actions + a] + gamma * future_v;
            q_values[s * num_actions + a] = q;
            if q > max_q {
                max_q = q;
            }
        }
        v_next[s] = max_q;
    }
    (q_values, v_next)
}

/// Level 3: POMDP Bayesian belief filter on CPU.
pub fn pomdp_belief_cpu(
    prior_belief: &[f32],
    transition: &[f32],
    observation_likelihood: &[f32],
    num_states: usize,
) -> Vec<f32> {
    let mut b_pred = vec![0.0f32; num_states];
    for s_prime in 0..num_states {
        let mut sum_val = 0.0f32;
        for s in 0..num_states {
            sum_val += prior_belief[s] * transition[s * num_states + s_prime];
        }
        b_pred[s_prime] = sum_val;
    }

    let mut b_unnorm = vec![0.0f32; num_states];
    let mut total_mass = 0.0f32;
    for s_prime in 0..num_states {
        let val = observation_likelihood[s_prime] * b_pred[s_prime];
        b_unnorm[s_prime] = val;
        total_mass += val;
    }

    let mut posterior = vec![0.0f32; num_states];
    let inv_mass = 1.0f32 / total_mass.max(1e-12);
    for s_prime in 0..num_states {
        posterior[s_prime] = b_unnorm[s_prime] * inv_mass;
    }
    posterior
}

/// Level 5: Differentiable 2-bidder auction mechanism on CPU.
pub fn differentiable_auction_cpu(bids: &[f32], batch: usize, tau: f32) -> (Vec<f32>, Vec<f32>) {
    let mut allocations = vec![0.0f32; batch * 2];
    let mut payments = vec![0.0f32; batch * 2];

    for b in 0..batch {
        let b0 = bids[b * 2];
        let b1 = bids[b * 2 + 1];

        let max_b = b0.max(b1);
        let e0 = ((b0 - max_b) / tau).exp();
        let e1 = ((b1 - max_b) / tau).exp();
        let sum_e = e0 + e1;

        let p0 = e0 / sum_e;
        let p1 = e1 / sum_e;
        allocations[b * 2] = p0;
        allocations[b * 2 + 1] = p1;

        let min_b = b0.min(b1);
        payments[b * 2] = p0 * min_b;
        payments[b * 2 + 1] = p1 * min_b;
    }
    (allocations, payments)
}

/// Dispatches Bayes Risk calculation to WebGPU with fallback to CPU.
pub fn bayes_decision_rule_gpu_or_cpu(
    device: Option<&Arc<wgpu::Device>>,
    queue: Option<&Arc<wgpu::Queue>>,
    probs: &[f32],
    cost_matrix: &[f32],
    batch: usize,
    num_classes: usize,
    num_actions: usize,
    c_reject: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (d, q) = match (device, queue) {
        (Some(d), Some(q)) => (d, q),
        _ => {
            return bayes_decision_rule_cpu(
                probs,
                cost_matrix,
                batch,
                num_classes,
                num_actions,
                c_reject,
            );
        }
    };

    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bayes_risk_shader"),
        source: wgpu::ShaderSource::Wgsl(DECISION_HIERARCHY_SHADER.into()),
    });

    let probs_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probs_buf"),
        contents: bytemuck::cast_slice(probs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let cost_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cost_buf"),
        contents: bytemuck::cast_slice(cost_matrix),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = (batch * num_actions * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_risk_buf"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_risk_buf"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = ShaderUniforms {
        batch: batch as u32,
        num_classes: num_classes as u32,
        num_actions: num_actions as u32,
        c_reject,
    };

    let uniform_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform_buf"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bayes_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bayes_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: probs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cost_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bayes_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bayes_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bayes_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bayes_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (batch as u32).div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, output_size);
    q.submit(Some(encoder.finish()));

    let buffer_slice = staging_buf.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        sender.send(res).unwrap();
    });

    let _ = d.poll(wgpu::Maintain::Wait);

    if receiver.recv().unwrap().is_ok() {
        let data = buffer_slice.get_mapped_range();
        let expected_risk: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        let mut best_actions = vec![0.0f32; batch];
        let mut reject_mask = vec![0.0f32; batch];
        for b in 0..batch {
            let mut min_r = f32::INFINITY;
            let mut best_a = 0usize;
            for a in 0..num_actions {
                let r = expected_risk[b * num_actions + a];
                if r < min_r {
                    min_r = r;
                    best_a = a;
                }
            }
            best_actions[b] = best_a as f32;
            reject_mask[b] = if min_r > c_reject { 1.0f32 } else { 0.0f32 };
        }
        (expected_risk, best_actions, reject_mask)
    } else {
        bayes_decision_rule_cpu(
            probs,
            cost_matrix,
            batch,
            num_classes,
            num_actions,
            c_reject,
        )
    }
}
