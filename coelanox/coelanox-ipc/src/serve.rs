//! Framed binary IPC on **stdin/stdout** (stderr is reserved for logs / `COELANOX_SERVE_READY`).
//!
//! ## Protocol v1 (`IPC_PROTOCOL_VERSION = 1`)
//!
//! All multi-byte integers are **little-endian**. Payloads are **raw `f32` little-endian** (4 bytes per value).
//!
//! ### Client → server (stdin)
//!
//! **`CREQ`** — run one inference  
//! - 4 bytes: ASCII `CREQ`  
//! - 4 bytes: `u32` `payload_bytes` (length of tensor blob)  
//! - `payload_bytes`: concatenated **input** tensor in **manifest input order** (flattened row-major), `f32` LE.  
//!   Total `f32` count must equal the sum of `product(shape)` over `manifest.input_shapes`.
//!
//! **`CNXT`** — graceful shutdown (4 bytes only)  
//! - Server stops the loop after sending no further responses for that message.
//!
//! Any other 4-byte prefix → error response on stdout, loop continues.
//!
//! ### Server → client (stdout)
//!
//! **`CROk`** — success  
//! - 4 bytes: `CROk`  
//! - 4 bytes: `u32` `out_bytes` (length of output blob)  
//! - `out_bytes`: output tensor(s) flattened **`f32` LE** (same ordering as `run` JSON output).
//!
//! **`CREr`** — error  
//! - 4 bytes: `CREr`  
//! - 4 bytes: `u32` `code` (application-defined; e.g. 400 = bad request, 500 = execution)  
//! - 4 bytes: `u32` `msg_len`  
//! - `msg_len` bytes: UTF-8 message (for humans / SDKs).

use coelanox_core::format::CoelanoxManifest;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_orchestrator::CoelanoxRuntime;
use std::io::{self, Read, Write};
use tracing::{info, warn};

/// Bump when the on-wire framing or message semantics change.
pub const IPC_PROTOCOL_VERSION: u32 = 1;

const MAGIC_REQ: &[u8; 4] = b"CREQ";
const MAGIC_OK: &[u8; 4] = b"CROk";
const MAGIC_ERR: &[u8; 4] = b"CREr";
const MAGIC_EXIT: &[u8; 4] = b"CNXT";

/// Cap single-request tensor payload (bytes) to avoid OOM on malformed length.
const MAX_IPC_PAYLOAD_BYTES: u32 = 512 * 1024 * 1024;

const ERR_BAD_MAGIC: u32 = 400;
const ERR_PAYLOAD_LIMIT: u32 = 413;
const ERR_SHAPE: u32 = 422;
const ERR_EXEC: u32 = 500;

/// Options for [`run_serve_stdio`]. Container must already be loaded on `runtime`.
pub struct ServeOptions<'a> {
    pub container_path: &'a str,
    pub backend: &'a str,
}

/// Load is done by the CLI; this runs the request loop until stdin closes or **`CNXT`**.
pub fn run_serve_stdio(runtime: &CoelanoxRuntime, opts: &ServeOptions<'_>) -> CoelanoxResult<()> {
    info!(
        "Serve mode: container={} backend={} (IPC protocol v{})",
        opts.container_path, opts.backend, IPC_PROTOCOL_VERSION
    );

    let container = runtime.get_container_info(opts.container_path)?;
    let expected_input = tensor_elements(&container.manifest, true);
    let expected_output = tensor_elements(&container.manifest, false);

    eprintln!(
        "COELANOX_SERVE_READY version={} protocol={} input_elements={} output_elements={}",
        env!("CARGO_PKG_VERSION"),
        IPC_PROTOCOL_VERSION,
        expected_input,
        expected_output
    );

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    loop {
        let mut magic = [0u8; 4];
        match stdin.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                info!("serve: end of stdin");
                break;
            }
            Err(e) => return Err(CoelanoxError::FileSystem(e)),
        }

        if &magic == MAGIC_EXIT {
            info!("serve: shutdown (CNXT)");
            break;
        }

        if &magic != MAGIC_REQ {
            warn!("serve: unknown magic {:?}", magic);
            write_response_err(&mut stdout, ERR_BAD_MAGIC, "expected CREQ or CNXT")?;
            continue;
        }

        let payload_len = match read_u32_le(&mut stdin) {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(CoelanoxError::FileSystem(e)),
        };

        if payload_len > MAX_IPC_PAYLOAD_BYTES {
            write_response_err(
                &mut stdout,
                ERR_PAYLOAD_LIMIT,
                "payload exceeds MAX_IPC_PAYLOAD_BYTES",
            )?;
            drain_skip(&mut stdin, payload_len)?;
            continue;
        }

        let mut buf = vec![0u8; payload_len as usize];
        if let Err(e) = stdin.read_exact(&mut buf) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(CoelanoxError::FileSystem(e));
        }

        let input = match bytes_to_f32_vec(&buf) {
            Ok(v) => v,
            Err(msg) => {
                write_response_err(&mut stdout, ERR_BAD_MAGIC, &msg)?;
                continue;
            }
        };

        if input.len() != expected_input {
            write_response_err(
                &mut stdout,
                ERR_SHAPE,
                &format!(
                    "input has {} f32 elements; manifest expects {}",
                    input.len(),
                    expected_input
                ),
            )?;
            continue;
        }

        match runtime.execute_with_backend(opts.container_path, &input, opts.backend) {
            Ok(out) => {
                write_response_ok(&mut stdout, &out)?;
            }
            Err(e) => {
                write_response_err(&mut stdout, ERR_EXEC, &e.to_string())?;
            }
        }
    }

    Ok(())
}

fn tensor_elements(m: &CoelanoxManifest, inputs: bool) -> usize {
    let shapes = if inputs {
        &m.input_shapes
    } else {
        &m.output_shapes
    };
    shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .sum()
}

fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn drain_skip<R: Read>(r: &mut R, mut n: u32) -> CoelanoxResult<()> {
    let mut buf = [0u8; 8192];
    while n > 0 {
        let take = std::cmp::min(n as usize, buf.len());
        r.read_exact(&mut buf[..take])
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        n -= take as u32;
    }
    Ok(())
}

fn bytes_to_f32_vec(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() % 4 != 0 {
        return Err(format!(
            "payload length {} is not a multiple of 4 (f32 LE)",
            b.len()
        ));
    }
    Ok(b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn write_response_ok<W: Write>(w: &mut W, data: &[f32]) -> CoelanoxResult<()> {
    let bytes = f32_vec_to_bytes(data);
    w.write_all(MAGIC_OK)?;
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

fn write_response_err<W: Write>(w: &mut W, code: u32, msg: &str) -> CoelanoxResult<()> {
    let msg = truncate_utf8(msg, 65535);
    let bytes = msg.as_bytes();
    w.write_all(MAGIC_ERR)?;
    w.write_all(&code.to_le_bytes())?;
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)?;
    w.flush()?;
    Ok(())
}

fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}
