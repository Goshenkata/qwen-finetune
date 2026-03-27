#!/usr/bin/env python3
"""Fine-tune a Qwen3.5 model with bf16 LoRA using Unsloth for code duplication detection."""

import argparse
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig


LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_SEQ_LEN = 4096  # ~4hr runtime on laptop GPU; longer seqs truncate context but keep response


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen3.5 for code duplication detection")
    p.add_argument("--model", default="unsloth/Qwen3.5-0.8B", help="HF model name")
    p.add_argument("--train-file", default="train.json")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--gguf-dir", default="qwen-duplication-gguf")
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs with max steps (for test runs)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"=== Fine-tuning {args.model} ===")
    print(f"Settings: epochs={args.epochs}, lr={args.lr}, lora_r={LORA_R}, lora_alpha={LORA_ALPHA}")

    # --- Load training data ---
    with open(args.train_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} training examples")

    # Convert to ShareGPT conversations format
    conversations = []
    for item in raw_data:
        conversations.append({
            "conversations": [
                {"from": "human", "value": item["prompt"]},
                {"from": "gpt", "value": item["result"]},
            ]
        })

    train_dataset = Dataset.from_list(conversations)
    print(f"Training examples: {len(train_dataset)}")

    # --- Load model (bf16 LoRA) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=False,
        load_in_16bit=True,
        max_seq_length=MAX_SEQ_LEN,
    )

    # Apply chat template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    # Format dataset (convert ShareGPT from/value → role/content for apply_chat_template)
    _role_map = {"human": "user", "gpt": "assistant"}

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": _role_map[m["from"]], "content": m["value"]} for m in convo],
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    # --- Apply LoRA ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
    )

    # --- Train ---
    training_args = dict(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=args.output_dir,
        max_seq_length=MAX_SEQ_LEN,
        save_strategy="epoch",
        save_total_limit=1,
        dataset_num_proc=2,
        packing=False,
    )
    if args.max_steps > 0:
        training_args["max_steps"] = args.max_steps

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**training_args),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training complete. Stats: {trainer_stats.metrics}")

    # --- Save LoRA adapter ---
    lora_dir = args.output_dir + "/lora_adapter"
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA adapter saved to {lora_dir}")

    # --- Save GGUF ---
    print(f"Exporting GGUF to {args.gguf_dir} ...")
    model.save_pretrained_gguf(args.gguf_dir, tokenizer, quantization_method="q4_k_m")
    print(f"GGUF saved to {args.gguf_dir}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
