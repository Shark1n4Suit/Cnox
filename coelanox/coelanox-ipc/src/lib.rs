//! IPC transport for **`coelanox serve`**: a long-lived process that loads one `.cnox` and
//! (eventually) exchanges framed tensor requests on stdin/stdout.
//!
//! This crate **wraps the runtime**, not the container artifact. The `.cnox` file is unchanged;
//! verification and execution semantics stay in `coelanox-orchestrator`.

pub mod serve;

pub use serve::{run_serve_stdio, ServeOptions, IPC_PROTOCOL_VERSION, MAX_IPC_PAYLOAD_BYTES};
