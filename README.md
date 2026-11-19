# Unsloth GGUF Updater

A Python utility for checking and updating locally installed [Unsloth](https://huggingface.co/unsloth) Dynamic GGUF model weights from Hugging Face.

Designed for large local model collections where models can occupy hundreds of gigabytes or several terabytes. The updater discovers installed quantizations, compares them against upstream releases, and downloads changed weights without overwriting existing models.

## Features

- **Automatic model discovery** — Finds installed `UD-*` GGUF quantizations without a hardcoded model list.
- **Upstream update detection** — Checks Unsloth's Hugging Face repositories for corresponding weights.
- **Persistent SHA-256 cache** — Avoids repeatedly hashing hundreds of gigabytes of unchanged files.
- **Fast verification mode** — Compares filenames and file sizes without reading model contents.
- **Full integrity verification** — Verifies file contents against upstream SHA-256 checksums.
- **Incremental downloads** — Downloads only missing or changed GGUF shards.
- **Resumable downloads** — Retains staging data for reuse when downloads are interrupted.
- **Storage-efficient updates** — Hard-links identical shards where supported.
- **Non-destructive installation** — Preserves original model weights.
- **Per-file progress reporting** — Displays hashing progress, throughput, and completion status.
- **Model filtering** — Check or update individual models without scanning the entire collection.

The updater is not limited to a particular Unsloth Dynamic quantization version. It checks current upstream weights, whether they were produced using Dynamic V3.0, V3.1, or another revision.

## Requirements

- Python 3.10 or later
- `huggingface_hub`
- Hugging Face CLI (`hf`)
- Internet access to Hugging Face
- Sufficient disk space for updated GGUF shards

### Installation

On Debian or Ubuntu, install Python virtual environment support:

```bash
sudo apt install python3-venv
```

Create a dedicated virtual environment:

```bash
python3 -m venv /opt/venvs/unsloth-updater
source /opt/venvs/unsloth-updater/bin/activate
```

Install the Hugging Face library and CLI:

```bash
python -m pip install -U huggingface_hub
```

Verify the installation:

```bash
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
hf --help
```

The virtual environment keeps dependencies separate from the system Python installation, avoiding PEP 668 externally managed environment errors.

**Important:** Activate the virtual environment in each new shell session:

```bash
source /opt/venvs/unsloth-updater/bin/activate
```

Alternatively, invoke its Python executable directly:

```bash
PATH="/opt/venvs/unsloth-updater/bin:$PATH" \
  /opt/venvs/unsloth-updater/bin/python \
  update-unsloth.py --root /models
```

## Model directory structure

The updater expects a model root containing individual model directories. The default root is `/models`.

Example:

```text
/models/
├── DeepSeek-V4.1-Flash-1017/
│   └── UD-Q8_K_XL/
│       ├── ...-00001-of-00009.gguf
│       └── ...-00009-of-00009.gguf
├── GLM-5.3-Flash/
│   └── UD-Q4_K_XL/
│       ├── ...-00001-of-00006.gguf
│       └── ...-00006-of-00006.gguf
├── Kimi-K3/
│   ├── UD-Q2_K_XL/
│   └── UD-Q8_K_XL/
└── Qwen3.6-27B-MTP/
    └── Qwen3.6-27B-UD-Q8_K_XL.gguf
```

Both sharded GGUFs in quantization subdirectories and compatible single-file GGUFs directly inside model directories are supported.

The script derives model names from GGUF filenames and constructs corresponding Unsloth repository names.

It discovers only existing `UD-*` quantizations. It does not automatically download new quantization levels, and non-UD formats such as `Q8_0` and `Q5_K_M` are outside its current scope.

## Usage

### Check installed models

```bash
python update-unsloth.py --root /models
```

This is the default dry-run operation.

The updater discovers installed quantizations, fetches upstream metadata, and compares their SHA-256 checksums.

No model weights are downloaded or replaced.

### Fast update check

```bash
python update-unsloth.py --root /models --fast
```

Compares local filenames and file sizes against upstream metadata without hashing model contents.

This is useful for quickly checking very large model collections.

**Limitation:** Files with identical sizes but different contents cannot be distinguished. A successful fast check is not proof that the installed weights match upstream.

### Download updates

```bash
python update-unsloth.py --root /models --apply
```

For every discovered quantization, the updater:

1. Retrieves the latest upstream GGUF file metadata.
2. Checks local files using SHA-256 and the persistent hash cache.
3. Identifies missing or changed shards.
4. Reuses matching shards where possible.
5. Downloads required files into a staging directory.
6. Verifies the complete staged model against upstream checksums.
7. Installs the verified weights into a revision-specific directory.
8. Preserves the original model files.

`--apply` cannot be combined with `--fast`, because downloading and installing updates requires content verification.

### Update one model

```bash
python update-unsloth.py --root /models \
  --model GLM-5.3-Flash --apply
```

Model filtering is case-insensitive and supports partial name matches.

Multiple filters can be supplied:

```bash
python update-unsloth.py --root /models \
  --model GLM-5.3-Flash \
  --model DeepSeek-V4.1-Flash-1017 \
  --apply
```

### Force complete integrity verification

```bash
python update-unsloth.py --root /models --rehash
```

Recomputes SHA-256 hashes even when cached hashes are available.

This is useful when checking storage integrity or when cached metadata should not be trusted.

### Control download concurrency

```bash
python update-unsloth.py --root /models \
  --apply --workers 4
```

The default is two concurrent Hugging Face download workers.

## Command-line options

| Option | Description |
|---|---|
| `--root PATH` | Model collection root; default `/models` |
| `--apply` | Download and install changed weights |
| `--fast` | Compare filenames and sizes only |
| `--rehash` | Recalculate all SHA-256 hashes |
| `--model NAME` | Filter by model name; repeatable |
| `--workers N` | Hugging Face download workers; default `2` |
| `--cache-file PATH` | Override the persistent hash cache location |
| `--help` | Display command-line help |

`--fast` and `--rehash` are mutually exclusive.

## SHA-256 caching

Hashing a multi-hundred-gigabyte model can take several minutes or considerably longer, depending on storage throughput.

To avoid repeated work, the updater maintains a persistent SHA-256 cache in:

```text
~/.cache/unsloth-updater/
```

Each cache entry includes:

- Filesystem device identifier
- Inode number
- File size
- Nanosecond modification time
- Nanosecond change time
- Calculated SHA-256 digest

If these metadata values are unchanged, the previously computed hash can be reused.

Hashes are saved after each successfully processed file. An interrupted verification can therefore retain results from completed shards.

The first full scan of an existing collection still requires reading each file once. Subsequent scans can reuse the cached results.

The cache assumes normal filesystem metadata behavior. Use `--rehash` when stronger assurance is required.

## Installation and file preservation

Updated weights are installed beside their existing versions rather than replacing them.

Example:

```text
DeepSeek-V4.1-Flash-1017/
├── UD-Q8_K_XL/
│   └── Original GGUF shards
└── UD-Q8_K_XL-upstream-2688c0c3c2c3/
    ├── Updated GGUF shards
    └── .unsloth-update.json
```

Revision-specific directories are named using the upstream Hugging Face repository commit.

Each installed update contains a `.unsloth-update.json` manifest recording the repository, revision, quantization, file sizes, and SHA-256 checksums.

The original directory is not overwritten or automatically deleted.

When identical shards can be hard-linked, they share underlying filesystem storage rather than consuming space twice.

### Temporary files

Downloads are staged beneath each model directory:

```text
.unsloth-update-stage/
```

Completed updates are moved into their final directories after verification. Incomplete staging data is retained when an update fails or is interrupted.

Do not delete staging directories while an update is running.

## Integration with inference engines

The updater manages GGUF weight files. It does not automatically reconfigure inference engines or regenerate derived model data.

For systems using Project Maya or similar engines, preprocessing artifacts may depend on the exact model weights.

Examples include:

- Expert placement profiles
- Converted expert data
- Dense tensor packs
- Engram data
- Tokenizer assets
- Vision preprocessing files

These are preserved but not regenerated.

Before switching an inference engine to newly installed weights, verify compatibility and regenerate derived artifacts when necessary.

## Limitations

- Only discovers installed `UD-*` GGUF quantizations.
- Assumes corresponding models are published in Unsloth Hugging Face repositories using compatible naming conventions.
- Does not automatically discover arbitrary repository aliases.
- Does not automatically add new quantization formats.
- Does not update non-GGUF assets.
- Does not automatically activate new model versions.
- Does not delete obsolete weights or old update directories.
- Cached hashes rely on filesystem metadata remaining trustworthy.
- Fast mode cannot detect content changes that preserve file size.
- Repository changes do not necessarily correspond to changes in the Dynamic quantization algorithm.

A model whose repository or quantization cannot be located is skipped. Other errors are reported in the final summary.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Completed without processing errors, including runs with skipped models |
| `1` | One or more processing errors |
| `130` | Interrupted with Ctrl+C |

## Typical workflow

For routine usage:

```bash
# Activate the virtual environment
source /opt/venvs/unsloth-updater/bin/activate

# Quickly inspect possible updates
python update-unsloth.py --fast

# Download and verify upstream changes
python update-unsloth.py --apply
```

For occasional full integrity verification:

```bash
python update-unsloth.py --rehash
```

After an update, review newly installed directories and validate inference compatibility before modifying any active model configuration.

## Disclaimer

This is an independent utility for managing locally stored Unsloth GGUF weights. It is not an official Unsloth tool.

Repository layout, GGUF naming, and Hugging Face CLI behavior may change. Review reported repository mappings before enabling large downloads.

## References

- [Unsloth on Hugging Face](https://huggingface.co/unsloth)
- [Hugging Face Hub Python documentation](https://huggingface.co/docs/huggingface_hub)
- [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Unsloth documentation](https://docs.unsloth.ai/)
