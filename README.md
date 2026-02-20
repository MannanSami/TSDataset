# TSBugger Pipeline

Synthetic TypeScript conversation dataset generator. Produces multi-turn bug-fixing and code-generation conversations using LLM-driven bug injection, user simulation, and assistant responses.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup

```bash
uv sync
cp .env.example .env  # configure API keys
```

Enter your api keys in the `.env` file. The user model is for fallback, you may use the same key as the assistant or use a local model by setting a base url. If you're not using the fallback at all, those keys are unused but the simulator will log a warning when it can't reach it.

## Usage

### Validate setup

```bash
cd src
uv run test_setup.py        # check env, imports, model connections
uv run test_validators.py   # run validator unit tests
```

### Generate dataset

```bash
cd src
uv run generate_dataset.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `-n, --samples` | `10` | Number of samples to generate |
| `-o, --output-dir` | `../output` | Output directory for generated files |
| `-d, --dataset-path` | configured path | Path to source TypeScript dataset |
| `--code-gen-ratio` | `0.2` | Ratio of code generation samples (0.0-1.0) |
| `--multi-bug-ratio` | `0.4` | Ratio of multi-bug samples in bug fixing (0.0-1.0) |
| `--multi-turn-ratio` | `0.5` | Ratio of multi-turn conversations (0.0-1.0) |
| `--val-ratio` | `0.1` | Validation set ratio for train/val split |
| `--no-parquet` | off | Skip parquet generation, only save JSON |
| `--no-resume` | off | Start fresh, ignore existing samples |
| `--show-sample` | off | Print a sample conversation at the end |

### Examples

```bash
# Quick test run
uv run generate_dataset.py -n 2 --no-resume

# Full generation with custom ratios
uv run generate_dataset.py -n 100 --code-gen-ratio 0.3 --multi-turn-ratio 0.7

# JSON only, no parquet
uv run generate_dataset.py -n 50 --no-parquet
```

Progress is saved after each sample. Press `Ctrl+C` to interrupt safely — rerun the same command to resume.
