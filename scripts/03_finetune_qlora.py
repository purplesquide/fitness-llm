from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from fitness_llm.config import ProjectPaths, Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a fitness coach model with QLoRA.")
    parser.add_argument("--dataset", type=Path, default=ProjectPaths().dataset_path)
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths().default_adapter_dir)
    parser.add_argument("--base-model", type=str, default=Settings().base_model_id)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _configure_precision() -> tuple[bool, bool, torch.dtype]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA fine-tuning.")

    major, _minor = torch.cuda.get_device_capability()
    use_bf16 = major >= 8
    use_fp16 = not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return use_bf16, use_fp16, compute_dtype


def main() -> None:
    args = parse_args()
    settings = Settings()

    token = os.getenv("HF_TOKEN") or settings.hf_token
    if token:
        login(token=token, add_to_git_credential=False)

    use_bf16, use_fp16, compute_dtype = _configure_precision()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    def format_sample(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_dataset = split["train"].map(format_sample)
    eval_dataset = split["test"].map(format_sample)

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_fp16,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        load_best_model_at_end=True,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
