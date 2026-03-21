# Installing Coelanox from a release

This page describes how to install the Coelanox CLI from a pre-built release (no Rust toolchain required). For building from source, see the main [README](../README.md).

---

## Prerequisites

- A supported platform: Linux (x86_64) or Windows (x86_64). (macOS: build from source.)
- No runtime dependencies beyond the single binary; optional CLF backends go in `~/.coelanox/clf/` (see [OPERATIONS.md](OPERATIONS.md)).

---

## Steps

1. **Download**  
   Go to the [Releases](https://github.com/coelanox-ai/coelanox/releases) page and download the archive for your OS:
   - **Linux:** `coelanox-linux-x86_64.tar.gz`
   - **Windows:** `coelanox-windows-x86_64.zip`

2. **Extract**  
   Unzip or untar the archive into a folder of your choice.

3. **Add to PATH**
   - **Linux / macOS:**  
     - Option A: From the extracted folder run `./install.sh` to copy `coelanox` to `/usr/local/bin` (may require `sudo` for directory creation).  
     - Option B: Add the extracted folder to your `PATH` (e.g. in `~/.bashrc` or `~/.zshrc`: `export PATH="/path/to/extracted/folder/linux:$PATH"`).
   - **Windows:**  
     Add the folder that contains `coelanox.exe` to your system or user `PATH` (Settings → System → About → Advanced system settings → Environment variables).

4. **Verify**  
   Open a new terminal and run:
   ```bash
   coelanox --help
   ```

---

## Optional: one-liner (Linux/macOS)

If you have `curl` and want to install the latest Linux/macOS binary without visiting the Releases page, you can use a script that downloads and extracts the appropriate asset (replace `VERSION` and repo URL with your values):

```bash
# Example: install v1.0.0 for Linux
COELANOX_VER="v1.0.0"
REPO="coelanox-ai/coelanox"
curl -sL "https://github.com/${REPO}/releases/download/${COELANOX_VER}/coelanox-linux-x86_64.tar.gz" | tar -xzf -
./install.sh
```

---

## After install

- Run `coelanox verify -f model.cnox` to check a container’s integrity.
- Run `coelanox run -f model.cnox -o out.json` to execute (see [CLI_REFERENCE.md](CLI_REFERENCE.md)).
- Backend discovery and the default CLF path (`~/.coelanox/clf/`) are described in [OPERATIONS.md](OPERATIONS.md).
