# CLF (Coelanox Library File) integration

The packager uses the **external CLF crate** at `../../../CLF` (e.g. `/home/shark/CLF` when this repo is at `/home/shark/cnox`). That crate provides the reader and canonical op_id registry.

---

## Current layout

- **External CLF** – Path dependency in `coelanox-packager`: `clf = { path = "../../../CLF" }`. The packager calls `clf::ClfReader::open(path)` and `build_code_section(op_ids, Skip)` using canonical op_ids from the CLF crate's `op_type_to_clf_id` (IR OpType → CLF OpType → op_id).
- **`coelanox/coelanox-clf/`** – Optional stub crate (same API) for builds without the external CLF; not used when the path dependency is set.
- **Memory HAL** – Unrelated to CLF: used for **backing** (e.g. NVMe layout). CLF is used only for **machine code / kernel blobs**.

---

## API your implementation must provide

Your CLF reader must implement the **same contract** as the stub so the packager can use it.

### 1. Load a CLF file

- **`ClfReader::open(path: &Path) -> ClfResult<Self>`**  
  Open the `.clf` file at `path` and return a reader instance.

### 2. Read the array

- **`read_blob_array(&self) -> ClfResult<Vec<u8>>`**  
  Return the **full payload** as one contiguous byte array (e.g. all kernel blobs concatenated, or the main weight/kernel array).  
  The packager uses this first; if non-empty, that becomes the container’s machine code.

- **`get_blob(&self, op_id: u32) -> Option<Vec<u8>>`**  
  Return a single blob for the given `op_id`. Used when the packager builds code by walking the IR and appending blobs in node order. Return `None` if the op_id is not in the CLF.

- **`list_op_ids(&self) -> Vec<u32>`**  
  Return all op_ids present in the file (for validation or tooling).

### 3. Errors

Use or mirror the stub’s **`ClfError`** (or map your errors into it):

- `Io(String)` – file not found / not readable
- `InvalidFormat(String)` – bad CLF header or structure
- `BlobNotFound(u32)` – optional, for missing op_id

### 4. OpType → op_id mapping

The packager maps IR `OpType` to `u32` via **`op_type_to_clf_id`** in `coelanox-packager/src/lib.rs`. Your CLF file should use the **same op_id numbering** so that `get_blob(op_id)` returns the correct kernel. If your format uses different ids, either:

- Remap in your reader (op_id from file → internal id), or  
- Change the mapping in the packager to match your spec.

---

## Adding your repo the professional way

### Option A: Git submodule (recommended)

1. Add your CLF repo as a submodule where the stub lives:

   ```bash
   cd coelanox
   git submodule add https://github.com/YOUR_ORG/your-clf-repo.git coelanox-clf-external
   ```

2. Point the workspace at the submodule:

   - Either **replace** the stub crate by making `coelanox-clf` a thin wrapper that re-exports your crate, or  
   - Rename the current stub to `coelanox-clf-stub` and add a **new** member `coelanox-clf` that is your submodule (same crate name, different path).

   Example: keep the stub for fallback and add your impl as a separate crate:

   - In `coelanox/Cargo.toml` add:  
     `"coelanox-clf-external",`  
     and point `coelanox-packager` to `coelanox-clf-external` instead of `coelanox-clf` when you want to use your implementation.

   Simpler: **replace** the contents of `coelanox/coelanox-clf/` with your project (or make `coelanox-clf` a small crate that only depends on your repo and implements the trait):

   - In your repo: implement the `ClfReader` trait (you can depend on `coelanox-clf` for the trait and error types only, or copy the trait definition into your repo and keep the same signature).
   - In this repo: in `coelanox/Cargo.toml`, change the member to your submodule path:

   ```toml
   members = [
       ...
       "coelanox-clf-external",  # was "coelanox-clf"
       ...
   ]
   ```

   Then in `coelanox-packager/Cargo.toml`:

   ```toml
   coelanox-clf = { path = "../coelanox-clf-external" }
   ```

   So “adding it to the repo” = add submodule + point workspace and packager at it.

3. After cloning Coelanox, users run:

   ```bash
   git submodule update --init --recursive
   ```

### Option B: Path dependency (local only)

If your CLF repo lives next to Coelanox on disk:

```text
parent/
  cnox/           # this repo
  your-clf-repo/  # your CLF impl
```

In `coelanox/coelanox-packager/Cargo.toml`:

```toml
coelanox-clf = { path = "../../your-clf-repo" }
```

Remove or don’t use the stub member for `coelanox-clf` in the workspace when using this. Good for local development before open-sourcing.

### Option C: Published crate (after open source)

Once your implementation is on crates.io:

```toml
coelanox-clf = "0.1"
```

No path or submodule needed; the stub can be retired or kept as an optional fallback behind a feature.

---

## Summary

- **Stub** in `coelanox/coelanox-clf/`: defines the API (load CLF, read the array via `read_blob_array` and/or `get_blob(op_id)`).
- **Packager** already loads CLF and reads the array when the backend is a `.clf` file.
- **Add your repo** by submodule (recommended), path dependency, or published crate, and implement the same API so the packager can use your reader without further changes.

If your format or API differs slightly (e.g. different op_id space), implement a thin adapter in your crate that matches this contract and maps to your internal format.
