# COELANOX Releases

Pre-built binaries for testing and deployment (no Rust required).

---

## Release 0.6.0-beta

**Linux x86_64:**

```
releases/coelanox-linux-x86_64.tar.gz
```

Extract and run:

```bash
tar -xzf coelanox-linux-x86_64.tar.gz -C /tmp/coelanox
cd /tmp/coelanox
./install.sh
# or: export PATH="/tmp/coelanox:$PATH"
coelanox --help
```

---

## Install from tarball

1. Extract: `tar -xzf coelanox-linux-x86_64.tar.gz -C /path/to/dir`
2. Add to PATH or run `./install.sh` from the extracted folder
3. Run `coelanox --help`

---

## Testing on another PC

1. Copy `coelanox-linux-x86_64.tar.gz` to the target machine (USB, SCP, etc.)
2. Extract and add to PATH
3. No Rust, no build tools required
