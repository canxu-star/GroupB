import argparse
import os
import yaml
import torch
from argparse import Namespace
from data_processor import get_processed_dataset

def main():
    parser = argparse.ArgumentParser(description="Wikitext-2 Data Processing Script using YAML config.")
    
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the YAML configuration file.")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'validation', 'test'], help="The dataset split to process.")
    parser.add_argument('--num_proc', type=int, help="Override number of CPU cores from config.")
    parser.add_argument('--output_dir', type=str, help="Override output directory from config.")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # --- START OF MODIFIED BLOCK ---
    # 手动设置不支持YAML的值
    config_dict['attention_mask_ids']['invalid_id'] = float("-inf")
    # --- END OF MODIFIED BLOCK ---

    config = Namespace(**{k: Namespace(**v) for k, v in config_dict.items()})

    if args.num_proc is not None:
        config.defaults.num_proc = args.num_proc
    if args.output_dir is not None:
        config.defaults.output_dir = args.output_dir

    print("--- Starting data processing with the following settings ---")
    print(f"Config File: {args.config}")
    print(f"Dataset Split: {args.split}")
    print(f"Number of Processes: {config.defaults.num_proc}")
    print(f"Output Directory: {config.defaults.output_dir}")
    print(f"Attention Mask Invalid ID: {config.attention_mask_ids.invalid_id}")
    print("-" * 50)

    processed_dataset = get_processed_dataset(config=config, split=args.split)

    output_dir = config.defaults.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"wikitext-2-raw-v1-processed-{args.split}")
    
    print(f"\n--- Saving processed dataset to: {save_path} ---")
    processed_dataset.save_to_disk(save_path)
    print("--- Dataset saved successfully! ---")

    if len(processed_dataset) > 0:
        print("\n--- Statistics for the first sample ---")
        sample = processed_dataset[0]
        
        # --- START OF MODIFIED BLOCK ---
        # 将 "attention_mask" 改为 "attention_mask_ids"
        # 将 "input_ids" 改为 "output_ids" (因为mask操作是在output_ids上进行的)
        effective_fields = torch.sum(sample["attention_mask_ids"] != config.attention_mask_ids.invalid_id).item()
        corrupted_tokens = torch.sum(sample["attention_mask_ids"] == config.attention_mask_ids.error_id).item()
        masked_tokens = torch.sum(sample["output_ids"] == config.special_tokens.mask_token_id).item()
        # --- END OF MODIFIED BLOCK ---
        
        print(f"Total effective fields (non-padded/dropped): {effective_fields}")
        print(f"Number of corrupted tokens (mask={config.attention_mask_ids.error_id}): {corrupted_tokens}")
        print(f"Number of masked tokens: {masked_tokens}")

if __name__ == '__main__':
    main()
