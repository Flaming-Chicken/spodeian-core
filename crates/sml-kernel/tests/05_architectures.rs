//! Category 05 Architectures Integration Test Suite for ai-kernel.

use ai_kernel::c05_architectures::t01_residual_mlp_mlp_mixer_and_gmlp as t01;
use candle_core::{Device, Result, safetensors::load};

#[path = "common/mod.rs"]
mod common;

#[test]
fn test_01_residual_mlp_mlp_mixer_and_gmlp_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t01::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Spatial Gating Unit Kernel Verification
    let w_sgu = tensors.get("w_sgu").unwrap().to_vec2::<f32>()?;
    let w_sgu_flat: Vec<f32> = w_sgu.into_iter().flatten().collect();
    let b_sgu = tensors.get("b_sgu").unwrap().to_vec2::<f32>()?;
    let b_sgu_flat: Vec<f32> = b_sgu.into_iter().flatten().collect();

    let b_sz = 2;
    let s_sz = 4;
    let d_sz = 4;
    let total_elements = b_sz * s_sz * d_sz;

    let z1 = vec![0.5f32; total_elements];
    let z2 = vec![1.0f32; total_elements];

    let sgu_cpu = t01::spatial_gating_cpu(&z1, &z2, &w_sgu_flat, &b_sgu_flat, b_sz, s_sz, d_sz);
    assert_eq!(sgu_cpu.len(), total_elements);

    let sgu_auto = pollster::block_on(t01::spatial_gating_auto(
        None,
        &z1,
        &z2,
        &w_sgu_flat,
        &b_sgu_flat,
        b_sz,
        s_sz,
        d_sz,
    ));

    for (i, (&c, &a)) in sgu_cpu.iter().zip(sgu_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "SGU CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t02_mixture_of_experts_top_k_expert_choice_and_soft_moe as t02;

#[test]
fn test_02_mixture_of_experts_top_k_expert_choice_and_soft_moe_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t02::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Soft MoE Dispatch Kernel Verification
    let num_tokens = 8;
    let d_model = 16;
    let total_slots = 4;

    let x = vec![0.25f32; num_tokens * d_model];
    let phi = vec![0.125f32; num_tokens * total_slots];

    let dispatch_cpu = t02::soft_moe_dispatch_cpu(&x, &phi, num_tokens, d_model, total_slots);
    assert_eq!(dispatch_cpu.len(), total_slots * d_model);

    let dispatch_auto =
        t02::soft_moe_dispatch_auto(None, &x, &phi, num_tokens, d_model, total_slots);
    for (i, (&c, &a)) in dispatch_cpu.iter().zip(dispatch_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "Dispatch CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    // 3. Soft MoE Combine Kernel Verification
    let slot_outputs = vec![0.5f32; total_slots * d_model];
    let psi = vec![0.25f32; num_tokens * total_slots];

    let combine_cpu =
        t02::soft_moe_combine_cpu(&psi, &slot_outputs, num_tokens, d_model, total_slots);
    assert_eq!(combine_cpu.len(), num_tokens * d_model);

    Ok(())
}

use ai_kernel::c05_architectures::t03_deepseek_moe_aux_loss_free_and_mixture_of_depths as t03;

#[test]
fn test_03_deepseek_moe_aux_loss_free_and_mixture_of_depths_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/03_deepseek_moe_aux_loss_free_and_mixture_of_depths.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t03::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. MoD Selective Gating Kernel Verification
    let num_tokens = 8;
    let d_model = 16;
    let total_elements = num_tokens * d_model;

    let x = vec![0.5f32; total_elements];
    let block_out = vec![1.2f32; total_elements];
    let gate = vec![0.8f32; num_tokens];
    let mask = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // 50% capacity

    let gating_cpu =
        t03::mod_selective_gating_cpu(&x, &block_out, &gate, &mask, num_tokens, d_model);
    assert_eq!(gating_cpu.len(), total_elements);

    let gating_auto =
        t03::mod_selective_gating_auto(None, &x, &block_out, &gate, &mask, num_tokens, d_model);
    for (i, (&c, &a)) in gating_cpu.iter().zip(gating_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "MoD Gating CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    // 3. Dynamic Bias Update Verification
    let b_dyn = vec![0.1f32, -0.05, 0.05, -0.1];
    let counts = vec![3.0f32, 5.0, 4.0, 4.0];
    let b_new = t03::dynamic_bias_update_cpu(&b_dyn, &counts, 0.1);
    assert_eq!(b_new.len(), 4);

    Ok(())
}

use ai_kernel::c05_architectures::t04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling as t04;

#[test]
fn test_04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t04::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Dynamic Parameter Merging (SMEAR Kernel Verification)
    let experts_w1 = tensors.get("experts_w1").unwrap().to_vec3::<f32>()?;
    let experts_w1_flat: Vec<f32> = experts_w1.into_iter().flatten().flatten().collect();
    let num_experts = 4;
    let in_dim = 16;
    let out_dim = 32;

    let routing_weights = vec![0.25f32, 0.25, 0.25, 0.25];

    let merged_cpu = t04::soup_weight_merge_cpu(
        &experts_w1_flat,
        &routing_weights,
        num_experts,
        in_dim,
        out_dim,
    );
    assert_eq!(merged_cpu.len(), in_dim * out_dim);

    let merged_auto = t04::soup_weight_merge_auto(
        None,
        &experts_w1_flat,
        &routing_weights,
        num_experts,
        in_dim,
        out_dim,
    );
    for (i, (&c, &a)) in merged_cpu.iter().zip(merged_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "Merged weights CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t05_kolmogorov_arnold_networks_kan as t05;

#[test]
fn test_05_kolmogorov_arnold_networks_kan_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t05::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Chebyshev Polynomial Basis Kernel Verification
    let x = vec![-1.5f32, -0.5, 0.0, 0.5, 1.2, 2.0];
    let degree = 4;

    let basis_cpu = t05::chebyshev_basis_cpu(&x, degree);
    assert_eq!(basis_cpu.len(), x.len() * (degree + 1));

    let basis_auto = t05::chebyshev_basis_auto(None, &x, degree);
    for (i, (&c, &a)) in basis_cpu.iter().zip(basis_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "Chebyshev basis CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth as t06;

#[test]
fn test_06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth_cpu_and_auto()
-> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t06::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Hyper-Connections Multi-Lane Update Kernel Verification
    let h1 = vec![1.0f32; 16];
    let h2 = vec![0.5f32; 16];
    let v = vec![0.2f32; 16];
    let beta0 = 0.5f32;
    let beta1 = 0.5f32;

    let update_cpu = t06::hyper_connections_update_cpu(&h1, &h2, &v, beta0, beta1);
    assert_eq!(update_cpu.len(), 16);

    let update_auto = t06::hyper_connections_update_auto(None, &h1, &h2, &v, beta0, beta1);
    for (i, (&c, &a)) in update_cpu.iter().zip(update_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "HC update CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t07_classic_to_modern_cnns_resnet_densenet_convnext as t07;

#[test]
fn test_07_classic_to_modern_cnns_resnet_densenet_convnext_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/01_classic_to_modern_cnns_resnet_densenet_convnext.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t07::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Global Response Normalization (GRN) Kernel Verification
    let b_sz = 2;
    let c_sz = 4;
    let h_sz = 4;
    let w_sz = 4;
    let total = b_sz * c_sz * h_sz * w_sz;

    let x = vec![0.5f32; total];
    let gamma = vec![1.2f32; c_sz];
    let beta = vec![0.1f32; c_sz];

    let grn_cpu = t07::global_response_norm_cpu(&x, &gamma, &beta, b_sz, c_sz, h_sz, w_sz, 1e-6);
    assert_eq!(grn_cpu.len(), total);

    let grn_auto =
        t07::global_response_norm_auto(None, &x, &gamma, &beta, b_sz, c_sz, h_sz, w_sz, 1e-6);
    for (i, (&c, &a)) in grn_cpu.iter().zip(grn_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "GRN CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t08_specialized_convolutions_and_channel_attention as t08;

#[test]
fn test_08_specialized_convolutions_and_channel_attention_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t08::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Channel Attention Recalibration Kernel Verification
    let b_sz = 2;
    let c_sz = 4;
    let h_sz = 8;
    let w_sz = 8;
    let total = b_sz * c_sz * h_sz * w_sz;

    let x = vec![0.5f32; total];
    let scale = vec![0.8f32; b_sz * c_sz];

    let att_cpu = t08::channel_attention_cpu(&x, &scale, b_sz, c_sz, h_sz, w_sz);
    assert_eq!(att_cpu.len(), total);

    let att_auto = t08::channel_attention_auto(None, &x, &scale, b_sz, c_sz, h_sz, w_sz);
    for (i, (&c, &a)) in att_cpu.iter().zip(att_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "Channel attention CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t09_unet_and_multiscale_feature_pyramids as t09;

#[test]
fn test_09_unet_and_multiscale_feature_pyramids_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t09::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. BiFPN Fast Normalized Feature Fusion Verification
    let total = 2 * 4 * 8 * 8;
    let p1 = vec![1.0f32; total];
    let p2 = vec![2.0f32; total];
    let w1 = 1.2f32;
    let w2 = 0.8f32;

    let fusion_cpu = t09::bifpn_fusion_cpu(&p1, &p2, w1, w2, 1e-4);
    assert_eq!(fusion_cpu.len(), total);

    let fusion_auto = t09::bifpn_fusion_auto(None, &p1, &p2, w1, w2, 1e-4);
    for (i, (&c, &a)) in fusion_cpu.iter().zip(fusion_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "BiFPN fusion CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t10_spatiotemporal_video_i3d_inflation_and_vivit_timesformer as t10;

#[test]
fn test_10_spatiotemporal_video_i3d_inflation_and_vivit_timesformer_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t10::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. I3D Weight Inflation Kernel Verification
    let conv2d_w_flat = tensors
        .get("conv2d_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected_flat = tensors
        .get("i3d_w")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let c_out = 4;
    let c_in = 4;
    let k_t = 3;
    let kh = 3;
    let kw = 3;

    let inflated_cpu = t10::i3d_inflate_cpu(&conv2d_w_flat, c_out, c_in, k_t, kh, kw);
    assert_eq!(inflated_cpu.len(), expected_flat.len());

    let inflated_auto = t10::i3d_inflate_auto(None, &conv2d_w_flat, c_out, c_in, k_t, kh, kw);
    for (i, (&c, &a)) in inflated_cpu.iter().zip(inflated_auto.iter()).enumerate() {
        assert!(
            (c - a).abs() < 1e-5,
            "I3D inflate CPU vs Auto mismatch at {i}: {c} vs {a}"
        );
        assert!(
            (c - expected_flat[i]).abs() < 1e-5,
            "I3D inflate vs Golden mismatch at {i}: {c} vs {}",
            expected_flat[i]
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t11_wavelet_scattering_wtconv_steerable_and_morphological_networks as t11;

#[test]
fn test_11_wavelet_scattering_wtconv_steerable_and_morphological_networks_cpu_and_auto()
-> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t11::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Morphological Dilation Kernel Verification
    let input_flat = tensors
        .get("input")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let se_flat = tensors
        .get("morph_se")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected_flat = tensors
        .get("morph_output")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    let b = 2;
    let c = 4;
    let h = 8;
    let w = 8;

    let morph_cpu = t11::morphological_dilation_cpu(&input_flat, &se_flat, b, c, h, w);
    assert_eq!(morph_cpu.len(), expected_flat.len());

    let morph_auto = t11::morphological_dilation_auto(None, &input_flat, &se_flat, b, c, h, w);
    for (i, (&cp, &au)) in morph_cpu.iter().zip(morph_auto.iter()).enumerate() {
        assert!(
            (cp - au).abs() < 1e-5,
            "Morphological CPU vs Auto mismatch at {i}: {cp} vs {au}"
        );
        assert!(
            (cp - expected_flat[i]).abs() < 1e-5,
            "Morphological vs Golden mismatch at {i}: {cp} vs {}",
            expected_flat[i]
        );
    }

    Ok(())
}

use ai_kernel::c05_architectures::t12_rnn_lstm_gru_and_attention_seq2seq as t12;

#[test]
fn test_12_rnn_lstm_gru_and_attention_seq2seq_cpu_and_auto() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/03_recurrent_ssm_and_memory/01_rnn_lstm_gru_and_attention_seq2seq.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    // 1. Linear Anchor on CPU
    let test_input = tensors.get("test_input").unwrap().to_vec2::<f32>()?;
    let test_input_flat: Vec<f32> = test_input.into_iter().flatten().collect();
    let weight = tensors.get("weight").unwrap().to_vec2::<f32>()?;
    let weight_flat: Vec<f32> = weight.into_iter().flatten().collect();
    let bias = tensors.get("bias").unwrap().to_vec1::<f32>()?;
    let golden_out = tensors.get("test_output").unwrap().to_vec2::<f32>()?;
    let golden_out_flat: Vec<f32> = golden_out.into_iter().flatten().collect();

    let anchor_calc = t12::linear_anchor_auto(&test_input_flat, &weight_flat, &bias, 8, 4, 4);
    for (i, (&calc, &gold)) in anchor_calc.iter().zip(golden_out_flat.iter()).enumerate() {
        assert!(
            (calc - gold).abs() < 1e-5,
            "Linear anchor error at {i}: {calc} vs {gold}"
        );
    }

    // 2. Fused LSTM Step Kernel Verification on Step 0
    let x_seq = tensors.get("x_seq").unwrap().to_vec3::<f32>()?;
    let h0 = tensors.get("h0").unwrap().to_vec2::<f32>()?;
    let c0_flat: Vec<f32> = tensors
        .get("c0")
        .unwrap()
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let w_ih = tensors.get("lstm_w_ih").unwrap().to_vec2::<f32>()?;
    let w_hh = tensors.get("lstm_w_hh").unwrap().to_vec2::<f32>()?;
    let b_ih = tensors.get("lstm_b_ih").unwrap().to_vec1::<f32>()?;
    let b_hh = tensors.get("lstm_b_hh").unwrap().to_vec1::<f32>()?;

    let batch_size = 2;
    let d_in = 4;
    let d_hid = 4;
    let total_gates = 4 * d_hid;

    // Compute step 0 gates on CPU: x0 @ w_ih^T + b_ih + h0 @ w_hh^T + b_hh
    let mut gates = vec![0.0f32; batch_size * total_gates];
    for b in 0..batch_size {
        for g in 0..total_gates {
            let mut sum = b_ih[g] + b_hh[g];
            for p in 0..d_in {
                sum += x_seq[b][0][p] * w_ih[g][p];
            }
            for p in 0..d_hid {
                sum += h0[b][p] * w_hh[g][p];
            }
            gates[b * total_gates + g] = sum;
        }
    }

    let (h_cpu, c_cpu) = t12::fused_lstm_step_cpu(&gates, &c0_flat, batch_size, d_hid);
    let (h_auto, c_auto) = t12::fused_lstm_step_auto(None, &gates, &c0_flat, batch_size, d_hid);

    let lstm_seq_out = tensors.get("lstm_seq_out").unwrap().to_vec3::<f32>()?;
    for b in 0..batch_size {
        for d in 0..d_hid {
            let idx = b * d_hid + d;
            let gold_h = lstm_seq_out[b][0][d];
            assert!(
                (h_cpu[idx] - h_auto[idx]).abs() < 1e-5,
                "LSTM step CPU vs Auto mismatch"
            );
            assert!(
                (c_cpu[idx] - c_auto[idx]).abs() < 1e-5,
                "LSTM step CPU vs Auto mismatch"
            );
            assert!(
                (h_cpu[idx] - gold_h).abs() < 1e-5,
                "LSTM step vs Golden mismatch: {} vs {}",
                h_cpu[idx],
                gold_h
            );
        }
    }

    Ok(())
}
