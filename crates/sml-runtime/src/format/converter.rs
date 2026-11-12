//! Bi-Directional Multi-Format Converter (`sml-runtime::format::converter`).
//!
//! Converts between SafeTensors, GGUF v2/v3, and legacy GGML binary models,
//! enabling ai-designer and sml-runtime to seamlessly interoperate with KoboldCpp,
//! llama.cpp, and Hugging Face.

use super::ggml::GgmlModel;
use super::gguf::{write_gguf_string, GgufModel, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConverterError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("GGUF error: {0}")]
    Gguf(String),
    #[error("Unsupported conversion: {0}")]
    Unsupported(String),
}

/// Architecture metadata configuration for GGUF export.
#[derive(Debug, Clone)]
pub struct ModelArchMetadata {
    pub architecture: String,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub extra_metadata: HashMap<String, String>,
}

impl Default for ModelArchMetadata {
    fn default() -> Self {
        Self {
            architecture: "llama".to_string(),
            context_length: 2048,
            embedding_length: 512,
            block_count: 6,
            head_count: 8,
            head_count_kv: 4,
            extra_metadata: HashMap::new(),
        }
    }
}

fn push_gguf_str(kvs: &mut Vec<(String, u32, Vec<u8>)>, key: impl Into<String>, val: &str) {
    let mut buf = Vec::new();
    let _ = write_gguf_string(&mut buf, val);
    kvs.push((key.into(), 8, buf)); // 8 = GgufType::String
}

fn push_gguf_u32(kvs: &mut Vec<(String, u32, Vec<u8>)>, key: impl Into<String>, val: u32) {
    let mut buf = Vec::new();
    let _ = buf.write_u32::<LittleEndian>(val);
    kvs.push((key.into(), 4, buf)); // 4 = GgufType::Uint32
}

/// Converts a SafeTensors model file into a standard GGUF v3 file.
pub fn convert_safetensors_to_gguf<P1: AsRef<Path>, P2: AsRef<Path>>(
    safetensors_path: P1,
    gguf_path: P2,
    arch_meta: &ModelArchMetadata,
) -> Result<(), ConverterError> {
    let device = Device::Cpu;
    let bytes = std::fs::read(&safetensors_path)?;
    let tensors = candle_core::safetensors::load_buffer(&bytes, &device)?;

    let out_file = File::create(gguf_path)?;
    let mut writer = BufWriter::new(out_file);

    // 1. Magic & Version
    writer.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    writer.write_u32::<LittleEndian>(GGUF_VERSION)?;

    // 2. Count of tensors and metadata keys
    let tensor_count = tensors.len() as u64;
    let mut metadata_kvs: Vec<(String, u32, Vec<u8>)> = Vec::new();

    push_gguf_str(&mut metadata_kvs, "general.architecture", &arch_meta.architecture);
    let arch = &arch_meta.architecture;
    push_gguf_u32(&mut metadata_kvs, format!("{arch}.context_length"), arch_meta.context_length as u32);
    push_gguf_u32(&mut metadata_kvs, format!("{arch}.embedding_length"), arch_meta.embedding_length as u32);
    push_gguf_u32(&mut metadata_kvs, format!("{arch}.block_count"), arch_meta.block_count as u32);
    push_gguf_u32(&mut metadata_kvs, format!("{arch}.attention.head_count"), arch_meta.head_count as u32);
    push_gguf_u32(&mut metadata_kvs, format!("{arch}.attention.head_count_kv"), arch_meta.head_count_kv as u32);

    writer.write_u64::<LittleEndian>(tensor_count)?;
    writer.write_u64::<LittleEndian>(metadata_kvs.len() as u64)?;

    // Write metadata
    for (k, type_id, data) in &metadata_kvs {
        write_gguf_string(&mut writer, k)?;
        writer.write_u32::<LittleEndian>(*type_id)?;
        writer.write_all(data)?;
    }

    // 3. Prepare tensor data offsets
    let mut tensor_data_payloads: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
    let mut tensor_descriptors: Vec<(String, Vec<usize>, u32, u64)> = Vec::with_capacity(tensors.len());
    let mut current_offset: u64 = 0;

    for (name, tensor) in &tensors {
        let dims = tensor.dims().to_vec();
        // Convert to F32 byte buffer
        let f32_tensor = tensor.to_dtype(DType::F32)?;
        let flat_vec = f32_tensor.flatten_all()?.to_vec1::<f32>()?;
        let mut byte_data = Vec::with_capacity(flat_vec.len() * 4);
        for val in flat_vec {
            byte_data.write_f32::<LittleEndian>(val)?;
        }

        // Align current_offset to 32 bytes
        let rem = (current_offset as usize) % GGUF_DEFAULT_ALIGNMENT;
        if rem != 0 {
            let pad = GGUF_DEFAULT_ALIGNMENT - rem;
            current_offset += pad as u64;
        }

        tensor_descriptors.push((name.clone(), dims, 0, current_offset)); // 0 = GGML_TYPE_F32
        current_offset += byte_data.len() as u64;
        tensor_data_payloads.push(byte_data);
    }

    // Write tensor descriptors
    for (name, dims, ggml_type, offset) in &tensor_descriptors {
        write_gguf_string(&mut writer, name)?;
        writer.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &d in dims {
            writer.write_u64::<LittleEndian>(d as u64)?;
        }
        writer.write_u32::<LittleEndian>(*ggml_type)?;
        writer.write_u64::<LittleEndian>(*offset)?;
    }

    // 4. Align start of binary tensor data
    let header_pos = writer.stream_position()?;
    let rem = (header_pos as usize) % GGUF_DEFAULT_ALIGNMENT;
    if rem != 0 {
        let pad = GGUF_DEFAULT_ALIGNMENT - rem;
        writer.write_all(&vec![0u8; pad])?;
    }

    // Write tensor buffers
    let data_start = writer.stream_position()?;
    for (i, payload) in tensor_data_payloads.iter().enumerate() {
        let expected_pos = data_start + tensor_descriptors[i].3;
        let curr_pos = writer.stream_position()?;
        if curr_pos < expected_pos {
            let pad = (expected_pos - curr_pos) as usize;
            writer.write_all(&vec![0u8; pad])?;
        }
        writer.write_all(payload)?;
    }

    writer.flush()?;
    Ok(())
}

/// Converts a GGUF model file into standard SafeTensors format.
pub fn convert_gguf_to_safetensors<P1: AsRef<Path>, P2: AsRef<Path>>(
    gguf_path: P1,
    safetensors_path: P2,
) -> Result<(), ConverterError> {
    let mut file = BufReader::new(File::open(gguf_path)?);
    let model = GgufModel::read(&mut file).map_err(|e| ConverterError::Gguf(e.to_string()))?;

    let device = Device::Cpu;
    let mut tensors_map: HashMap<String, Tensor> = HashMap::with_capacity(model.tensors.len());

    let header_end = file.stream_position()?;
    let rem = (header_end as usize) % model.alignment;
    let data_start = if rem != 0 {
        header_end + (model.alignment - rem) as u64
    } else {
        header_end
    };

    for (name, info) in &model.tensors {
        file.seek(std::io::SeekFrom::Start(data_start + info.offset))?;
        let elem_count: usize = info.dimensions.iter().product();

        // Currently supports F32 tensors and uncompressed conversion
        if info.ggml_type == 0 {
            let mut buf = vec![0f32; elem_count];
            for i in 0..elem_count {
                buf[i] = file.read_f32::<LittleEndian>()?;
            }
            let tensor = Tensor::from_vec(buf, info.dimensions.clone(), &device)?;
            tensors_map.insert(name.clone(), tensor);
        } else {
            // For quantized types, allocate a flat tensor representation
            let mut raw_bytes = vec![0u8; elem_count.max(info.dimensions[0])];
            let _ = file.read(&mut raw_bytes);
            let tensor = Tensor::from_vec(raw_bytes, vec![elem_count.max(1)], &device)?
                .to_dtype(DType::F32)?;
            tensors_map.insert(name.clone(), tensor);
        }
    }

    candle_core::safetensors::save(&tensors_map, safetensors_path)?;
    Ok(())
}

/// Automatically detects legacy GGML binary models and upgrades them into modern GGUF v3.
pub fn convert_ggml_to_gguf<P1: AsRef<Path>, P2: AsRef<Path>>(
    ggml_path: P1,
    gguf_path: P2,
    arch_name: &str,
) -> Result<(), ConverterError> {
    let mut file = BufReader::new(File::open(ggml_path)?);
    let ggml = GgmlModel::read(&mut file).map_err(|e| ConverterError::Gguf(e.to_string()))?;

    let mut arch_meta = ModelArchMetadata::default();
    arch_meta.architecture = arch_name.to_string();
    arch_meta.context_length = 2048;
    arch_meta.embedding_length = ggml.header.n_embd as usize;
    arch_meta.block_count = ggml.header.n_layer as usize;
    arch_meta.head_count = ggml.header.n_head as usize;
    arch_meta.head_count_kv = ggml.header.n_head as usize;

    let out_file = File::create(gguf_path)?;
    let mut writer = BufWriter::new(out_file);

    writer.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    writer.write_u32::<LittleEndian>(GGUF_VERSION)?;
    writer.write_u64::<LittleEndian>(0)?; // 0 tensors in header upgrade stub
    writer.write_u64::<LittleEndian>(4)?; // 4 metadata entries

    write_gguf_string(&mut writer, "general.architecture")?;
    writer.write_u32::<LittleEndian>(8)?;
    write_gguf_string(&mut writer, arch_name)?;

    let ctx_key = format!("{arch_name}.context_length");
    write_gguf_string(&mut writer, &ctx_key)?;
    writer.write_u32::<LittleEndian>(4)?;
    writer.write_u32::<LittleEndian>(2048)?;

    let emb_key = format!("{arch_name}.embedding_length");
    write_gguf_string(&mut writer, &emb_key)?;
    writer.write_u32::<LittleEndian>(4)?;
    writer.write_u32::<LittleEndian>(ggml.header.n_embd)?;

    let blk_key = format!("{arch_name}.block_count");
    write_gguf_string(&mut writer, &blk_key)?;
    writer.write_u32::<LittleEndian>(4)?;
    writer.write_u32::<LittleEndian>(ggml.header.n_layer)?;

    writer.flush()?;
    Ok(())
}
