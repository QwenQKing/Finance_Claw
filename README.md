# Anonymous Research Code

This repository contains the anonymous implementation accompanying the submitted research paper.

## Contents

- `src/`: core method, retrieval, memory, temporal, evaluation, and governance modules
- `scripts/`: utility scripts
- `utils/`: shared utilities
- `datasets/`: benchmark data, organized as `datasets/<dataset-name>/data.json` and `eval.jsonl`
- `config.yaml`: example runtime configuration
- `requirements.txt`: Python dependencies

## Setup

Use Python 3.10 or later, install the dependencies, and provide the required model and embedding service settings through the runtime environment.

```bash
python -m pip install -r requirements.txt
```

## Usage

The main runner is located at `src/runner.py`. Dataset and output paths can be supplied through the project configuration and command-line arguments used by the runner.

Each dataset is stored directly under `datasets/` by name. The training split is named `data.json`, and the evaluation split is named `eval.jsonl`.
