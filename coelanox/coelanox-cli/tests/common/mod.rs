//! Shared helpers for `coelanox-cli` integration tests (IPC, tiny containers).
#![allow(dead_code)]
// Each integration-test binary links this module independently; not every helper is used in every crate.

use coelanox_core::format::{CoelanoxContainer, CoelanoxManifest, CompressionType, HardwareTarget, ModelFormat};
use coelanox_core::ir::{self, DataType, OpType, Shape, UniversalIR};
use coelanox_core::CoelanoxResult;
use coelanox_packager::optimizers::{CpuOptimizer, HardwareOptimizer};
use std::io::{BufRead, BufReader, Read};
use std::process::ChildStderr;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

pub fn minimal_relu_container() -> CoelanoxResult<(tempfile::NamedTempFile, usize)> {
    let manifest = CoelanoxManifest::new(
        "ipc_smoke".into(),
        ModelFormat::Onnx,
        vec![vec![1, 10]],
        vec![vec![1, 10]],
        1024,
    );
    let expected_input: usize = manifest
        .input_shapes
        .iter()
        .map(|s| s.iter().product::<usize>())
        .sum();

    let mut universal_ir = UniversalIR::new("ipc_smoke".to_string());
    let shape = Shape::new(vec![1, 10]);
    let tid_in = universal_ir.add_intermediate_tensor("in".into(), shape.clone(), DataType::Float32);
    let tid_out = universal_ir.add_intermediate_tensor("out".into(), shape, DataType::Float32);
    let node_id = universal_ir.add_node(OpType::Relu);
    {
        let node = universal_ir
            .get_node_mut(node_id)
            .expect("Relu node");
        node.add_input_tensor(tid_in);
        node.add_output_tensor(tid_out);
    }
    universal_ir.sort_nodes_for_execution()?;
    let optimized_ir = CpuOptimizer::new().optimize(&universal_ir, HardwareTarget::Cpu)?;
    let ir_data = ir::serialize_optimized_ir_for_container(&optimized_ir)?;

    let container = CoelanoxContainer::new(
        manifest,
        vec![0u8; 1024],
        vec![],
        CompressionType::None,
        ir_data,
    );
    let file = tempfile::NamedTempFile::new()?;
    container.write_binary_file(file.path())?;
    Ok((file, expected_input))
}

pub fn f32_payload(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for x in values {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// One framed IPC response from stdout.
pub enum IpcFrame {
    Ok(Vec<f32>),
    Err { code: u32, msg: String },
}

pub fn read_ipc_frame(stdout: &mut impl Read) -> Result<IpcFrame, String> {
    let mut magic = [0u8; 4];
    stdout.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic == b"CROk" {
        let mut len_b = [0u8; 4];
        stdout.read_exact(&mut len_b).map_err(|e| e.to_string())?;
        let n = u32::from_le_bytes(len_b) as usize;
        let mut blob = vec![0u8; n];
        stdout.read_exact(&mut blob).map_err(|e| e.to_string())?;
        return Ok(IpcFrame::Ok(
            blob.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ));
    }
    if &magic == b"CREr" {
        let mut code_b = [0u8; 4];
        stdout.read_exact(&mut code_b).map_err(|e| e.to_string())?;
        let code = u32::from_le_bytes(code_b);
        let mut len_b = [0u8; 4];
        stdout.read_exact(&mut len_b).map_err(|e| e.to_string())?;
        let n = u32::from_le_bytes(len_b) as usize;
        let mut msg = vec![0u8; n];
        stdout.read_exact(&mut msg).map_err(|e| e.to_string())?;
        return Ok(IpcFrame::Err {
            code,
            msg: String::from_utf8_lossy(&msg).into_owned(),
        });
    }
    Err(format!("unexpected IPC magic: {:?}", magic))
}

/// Success path only; CREr becomes `Err` string (for smoke tests expecting CROk).
pub fn read_ipc_ok_only(stdout: &mut impl Read) -> Result<Vec<f32>, String> {
    match read_ipc_frame(stdout)? {
        IpcFrame::Ok(v) => Ok(v),
        IpcFrame::Err { code, msg } => Err(format!("CREr {code}: {msg}")),
    }
}

pub fn wait_serve_ready(stderr: ChildStderr, timeout: Duration) {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(l) = line else { break };
            if l.contains("COELANOX_SERVE_READY") {
                tx.send(()).ok();
            }
        }
    });
    rx.recv_timeout(timeout)
        .expect("serve should print COELANOX_SERVE_READY");
}
