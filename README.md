# Fitness LLM Colab Project

This directory is a complete, Colab-ready project for building a fitness chatbot with:

1. Supervised fine-tuning with QLoRA on Mistral 7B.
2. A FAISS-based RAG knowledge base built from seed knowledge and your exercise database.
3. A standalone FastAPI chat service that can use either the fine-tuned local model or Groq as a fallback.

The project is intentionally separated from the existing app so you can upload this folder to Colab and run it step by step without disturbing the current repository.

## Preferred Entry Point

If you want a single-file Colab workflow, open and run:

`Fitness_LLM_All_In_One_Colab.ipynb`

That notebook loads the full generated dataset, builds the RAG index, fine-tunes the model, and tests the result in one place.

## What Is Included

```text
fitness_llm_colab/
  Fitness_LLM_All_In_One_Colab.ipynb
  data/
    dataset_summary.json
    full_training_examples.json
    knowledge_seed.json
    seed_training_examples.json
  integration/
    fastapi_chat_app.py
  scripts/
    01_build_dataset.py
    02_validate_dataset.py
    03_finetune_qlora.py
    04_test_model.py
    05_build_rag_index.py
    06_test_retrieval.py
    07_chat_groq.py
  src/fitness_llm/
    __init__.py
    config.py
    dataset_builder.py
    llm_engine.py
    prompts.py
    rag.py
  .env.example
  pyproject.toml
  requirements-colab.txt
```

## Important Reality Check

The included training examples are a seed dataset so the pipeline runs end to end. They are not enough for production-quality fine-tuning.

For a real model, expand the dataset to at least 500 to 1000 high-quality conversation samples before training.

## What To Upload To Colab

Minimum upload:

1. This entire `fitness_llm_colab` directory.

Optional but recommended upload:

1. `free-exercise-db/exercises/` from the repository root if you want the RAG index to include your full exercise database.

If you upload both, keep them side by side in Colab, for example:

```text
/content/fitness_llm_colab
/content/exercises
```

## Colab Setup Commands

Run these commands in Colab cells.

### 1. Enter the project directory

```bash
%cd /content/fitness_llm_colab
```

### 2. Install dependencies

```bash
!pip install -q -r requirements-colab.txt
!pip install -q -e .
```

If you are on an A100 and want flash attention, run this extra command:

```bash
!pip install -q flash-attn --no-build-isolation
```

### 3. Log in to Hugging Face

```python
from huggingface_hub import login
login("YOUR_HF_TOKEN")
```

Or set an environment variable:

```python
import os
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
```

### 4. Build the training dataset

```bash
!python scripts/01_build_dataset.py
```

Output:

```text
artifacts/datasets/fitness_training_data.jsonl
```

### 5. Validate the dataset

```bash
!python scripts/02_validate_dataset.py
```

### 6. Build the RAG knowledge base

If you uploaded your exercise JSON files to `/content/exercises`:

```bash
!python scripts/05_build_rag_index.py --exercise-dir /content/exercises
```

If you only want to use the seed knowledge:

```bash
!python scripts/05_build_rag_index.py
```

Outputs:

```text
artifacts/rag/fitness_knowledge.index
artifacts/rag/fitness_knowledge_chunks.pkl
```

### 7. Fine-tune the model with QLoRA

```bash
!python scripts/03_finetune_qlora.py \
  --dataset artifacts/datasets/fitness_training_data.jsonl \
  --output-dir artifacts/models/fitness-mistral-qlora-final \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --max-seq-length 1024
```

Notes:

1. On a T4, reduce `--batch-size` to `1` if you hit out-of-memory.
2. The script auto-switches between `fp16` and `bf16` based on the GPU.
3. The included dataset is only for smoke testing. Replace it with your expanded dataset before serious training.

### 8. Test the fine-tuned model

```bash
!python scripts/04_test_model.py \
  --adapter-dir artifacts/models/fitness-mistral-qlora-final \
  --question "I am intermediate, training 4 days for hypertrophy. Should I increase my bench press weight?"
```

### 9. Test retrieval quality

```bash
!python scripts/06_test_retrieval.py --query "How many sets per week for hypertrophy?"
```

### 10. Use Groq fallback immediately

Set the API key first:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Then run:

```bash
!python scripts/07_chat_groq.py --message "What should I do if my squat stalled for 3 weeks?"
```

## Run The Standalone Chat API

After you have built the RAG index, and optionally after fine-tuning, you can run the standalone FastAPI service.

### Local model mode

```bash
!uvicorn integration.fastapi_chat_app:app --host 0.0.0.0 --port 8000
```

This expects:

1. `artifacts/rag/fitness_knowledge.index`
2. `artifacts/rag/fitness_knowledge_chunks.pkl`
3. `artifacts/models/fitness-mistral-qlora-final/` if you want the local fine-tuned model

### Groq mode

If you want to skip local model hosting, set:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Then start the same API. The engine can use Groq fallback without loading a local adapter.

## Example Chat Request

```bash
!curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "message": "What can replace overhead press if my shoulder hurts?",
    "user_profile": {"level": "intermediate", "goal": "hypertrophy", "days": 4},
    "history": []
  }'
```

## How To Expand The Dataset Properly

Edit `data/seed_training_examples.json` or point `scripts/01_build_dataset.py` to a larger JSON file with the same schema.

Each example must look like this:

```json
{
  "category": "program_advice",
  "user": "My bench press has stalled for 3 weeks. What should I do?",
  "assistant": "A 3-week plateau is common at intermediate level. First check recovery, especially sleep and protein. If those are fine, deload for one week by cutting volume roughly in half, then return to normal loading. If the lift still stalls, change the rep range for a short block."
}
```

## How To Use Your Existing Repository Data

You already have a useful exercise data source in this repository:

1. `free-exercise-db/exercises/*.json`

When running `scripts/05_build_rag_index.py`, pass that directory through `--exercise-dir`.

## Recommended Build Order

1. Build dataset.
2. Validate dataset.
3. Build RAG index.
4. Test Groq chat first.
5. Fine-tune with QLoRA.
6. Test local model.
7. Run FastAPI service.

## Common Problems

### CUDA out of memory

Use:

```bash
!python scripts/03_finetune_qlora.py --batch-size 1 --gradient-accumulation 8
```

### Dataset validation fails

Run:

```bash
!python scripts/02_validate_dataset.py --dataset artifacts/datasets/fitness_training_data.jsonl
```

### Hugging Face model download fails

You are usually missing an authenticated token for a gated model.

### Retrieval is weak

Upload the full exercise database and rebuild the index with `--exercise-dir`.

## Production Notes

For production quality, you should do three things before deployment:

1. Replace the seed dataset with a large curated dataset.
2. Store chat history and user profile in your real backend instead of sending placeholders.
3. Add moderation, rate limits, and model loading telemetry.
