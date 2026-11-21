# thinking-machines

POC — Inkling 2 mini as a harness and image OCR. Includes a first end-to-end fine-tuning example built on the [Tinker](https://thinkingmachines.ai/) API.

## What's here

- `src/thinking_machines/first-training.py` — a Tinker LoRA fine-tuning example on `Qwen/Qwen3.5-4B` that teaches the model to answer in lowercase:
  - samples completions from the base model for a few prompts, restyles them to lowercase, and filters out answers truncated at `max_tokens`;
  - builds training data with the chat template and per-token loss weights (prompt tokens weighted 0, completion tokens weighted 1);
  - runs 10 training steps with cross-entropy loss (Adam, lr 2e-4) and prints the loss per step;
  - saves a sampler checkpoint (with a Playground URL) and prints a before/after comparison between the base and the fine-tuned model.
- `scripts/envcrypt.py` — encrypts/decrypts `.env` with symmetric GPG (AES256) as `.env-$USER.gpg`, so secrets can be committed safely.

## Setup

Requires Python >= 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Dependencies: `tinker`, `tinker-cookbook`, `python-dotenv`.

### Secrets

Secrets live in `.env` (gitignored); the encrypted copy `.env-$USER.gpg` is what gets committed. `.envrc` (direnv) points `UV_ENV_FILE` at `.env` so `uv run` loads it automatically.

```bash
# restore .env from the encrypted copy
uv run scripts/envcrypt.py decrypt

# re-encrypt after editing .env
uv run scripts/envcrypt.py encrypt
```

The passphrase is prompted by `gpg` (pinentry) or read from the `ENVCRYPT_PASSWORD` environment variable. Use `--force` on `decrypt` to overwrite an existing `.env`.

Put your Tinker API key in `.env` (`TINKER_API_KEY=...`).

## Running the training example

```bash
uv run src/thinking_machines/first-training.py
```

It prints the training loss for each step, a Playground link for the saved checkpoint, and side-by-side base vs. fine-tuned responses for a few test prompts.
