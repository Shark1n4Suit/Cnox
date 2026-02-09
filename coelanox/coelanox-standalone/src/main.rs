//! Minimal binary: load one .cnox and run inference.
//!
//! Usage:
//!   coelanox-run model.cnox [input.json]
//! If input.json is omitted, uses dummy input (all 1.0).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: coelanox-run <model.cnox> [input.json]");
        std::process::exit(1);
    }
    let path = &args[1];
    let input_data = if args.len() > 2 {
        let json = std::fs::read_to_string(&args[2])?;
        serde_json::from_str::<Vec<f32>>(&json)?
    } else {
        let output = coelanox_standalone::run_container_dummy_input(path)?;
        println!("Output (dummy input): {:?}", output);
        return Ok(());
    };
    let output = coelanox_standalone::run_container(path, &input_data)?;
    println!("Output: {:?}", output);
    Ok(())
}
