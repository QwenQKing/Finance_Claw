# Anonymous Research Code

This repository contains the anonymous implementation accompanying the submitted research paper.

## Contents

- `src/`: core method, retrieval, memory, temporal, evaluation, and governance modules
- `scripts/`: utility scripts
- `utils/`: shared utilities
- `config.yaml`: example runtime configuration
- `requirements.txt`: Python dependencies

## Setup

Use Python 3.10 or later, install the dependencies, and provide the required model and embedding service settings through the runtime environment.

```bash
python -m pip install -r requirements.txt
```

## Usage

The main runner is located at `src/runner.py`. Dataset and output paths can be supplied through the project configuration and command-line arguments used by the runner.
