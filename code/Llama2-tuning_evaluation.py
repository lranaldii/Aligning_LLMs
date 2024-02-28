#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset
from tqdm import tqdm
from utils import check_matching

def train_model(dataset_path, base_model_name, refined_model_path, epochs, learning_rate):
    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  
    
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map={"": 0} # 1, auto
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Training Params
    train_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )
    
    # Trainer
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params
    )
    
    # Training
    fine_tuning.train()
    
    # Save Model
    fine_tuning.model.save_pretrained(refined_model_path)

def evaluate_model(dataset, refined_model_path):
    correct_preds = 0
    for text in tqdm(dataset):
        answer = generate(prompt)    
        if check_matching(content, text['target']):
            correct_preds += 1

    acc = (correct_preds * 100) / len(dataset)
    print(f"Accuracy: {acc}%")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the base model.")
    parser.add_argument("--refined_model_path", type=str, required=True, help="Path to save the refined model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training.")
    parser.add_argument("--action", type=str, choices=["train", "evaluate"], required=True, help="Whether to train or evaluate the model.")

    args = parser.parse_args()

    if args.action == "train":
        train_model(args.dataset_path, args.base_model_name, args.refined_model_path, args.epochs, args.learning_rate)
    elif args.action == "evaluate":
        evaluate_model(Dataset.from_pandas(pd.read_csv(args.dataset_path)), args.refined_model_path)
