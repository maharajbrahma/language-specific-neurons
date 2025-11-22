import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def build_wikipedia_token_ids(lang="en", model_name="meta-llama/Llama-2-7b-hf", output_dir="data", max_samples=1000000):

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"id.{lang}.train.llama")

    print(f"Loading Wikipedia dataset for language: {lang}")
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
    print(f"Loaded {len(dataset):,} articles total")

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Using subset of {len(dataset):,} articles")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    all_ids = []

    print("Tokenizing...")
    for text in tqdm(dataset["text"], total=len(dataset)):
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
        tokenized = tokenizer(text, add_special_tokens=False).input_ids
        if tokenized:
            all_ids.extend(tokenized)

    ids_tensor = torch.tensor(all_ids, dtype=torch.long)

    print(f"Tokenized {len(all_ids):,} tokens total. Saving to {output_path}")
    torch.save(ids_tensor, output_path, _use_new_zipfile_serialization=True)

    print("Saved tensor:", output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", type=str, default="en", help="Wikipedia language code, e.g. 'en', 'zh', 'fr'")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer model name")
    parser.add_argument("-o", "--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=1000, help="Number of articles to process for testing")
    args = parser.parse_args()

    build_wikipedia_token_ids(args.lang, args.model, args.output_dir, args.max_samples)
