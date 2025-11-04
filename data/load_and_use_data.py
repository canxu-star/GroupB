import os
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import torch
import random
# from transformers import AutoTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional

def decode_tokenids(token_ids: List[List[int]],tag:bool) -> List[str] :
    """
    Decode a sequence with visible mask tokens.
    """
    # tokenizer = setup_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    if(tag):
        token_ids=token_ids[::2]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text
def main():
    # 1. 定义已保存数据集的路径
    # 这个路径必须与 run_processing.py 中保存的路径完全一致
    output_dir = "./processed_data"
    split = "train"
    dataset_path = os.path.join(output_dir, f"wikitext-2-raw-v1-processed-{split}")

    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误：找不到数据集路径 '{dataset_path}'")
        print("请确保您已经成功运行了 run_processing.py 来生成数据。")
        return

    # 2. 从磁盘加载数据集
    print(f"--- 正在从 '{dataset_path}' 加载数据集... ---")
    try:
        # 这是关键步骤！
        loaded_dataset = load_from_disk(dataset_path)
        
        # set_format("torch") 确保数据集返回 PyTorch 张量
        # 这一步通常是必需的，因为从磁盘加载后格式可能需要重新设置
        loaded_dataset.set_format("torch")

        print("--- 数据集加载成功！ ---")
        print(loaded_dataset)

    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return

    # 3. 验证数据 (可选，但推荐)
    if len(loaded_dataset) > 0:
        print("\n--- 检查第一个样本 ---")
        first_sample = loaded_dataset[0]
        print("样本包含的键:", first_sample.keys())
        print("output_ids 的形状:", first_sample["output_ids"].shape)
        print("attention_mask_ids 的形状:", first_sample["attention_mask_ids"].shape)

        print("\noutput_ids (first 400):", first_sample["output_ids"][:400].tolist())
        print("\nAttention Mask (first 400):", first_sample["attention_mask_ids"][:400].tolist())
        print("\nLabels (first 400):", first_sample["labels_ids"][:400].tolist())

    # 4. 创建 PyTorch DataLoader 用于模型训练
    # 这是将数据集送入训练循环的标准方法
    batch_size = 4  # 您可以根据您的模型和GPU显存来调整
    
    # shuffle=True 在每个 epoch 开始时打乱数据，这对于训练至关重要
    train_dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)

    print(f"\n--- 创建 DataLoader 成功 (Batch Size: {batch_size}) ---")

    # 5. 模拟一个训练循环的批次迭代
    print("--- 从 DataLoader 中获取第一个批次 (batch) ---")
    
    # 从迭代器中获取一个批次的数据
    try:
        first_batch = next(iter(train_dataloader))

        print("批次包含的键:", first_batch.keys())
        
        # 检查批次中张量的形状
        # 注意形状现在是 [batch_size, sequence_length]
        print("批次中 'output_ids' 的形状:", first_batch["output_ids"].shape)
        print("批次中 'attention_mask_ids' 的形状:", first_batch["attention_mask_ids"].shape)
        print("批次中 'labels_ids' 的形状:", first_batch["labels_ids"].shape)

        print("\n现在，您可以将这个 'first_batch' 字典直接输入到您的模型中进行训练了。")

        print("\noutput text:",decode_tokenids(first_batch["output_ids"][0][:400].tolist(),1))
        # print("Attention Mask text:",decode_tokenids(sample["input_ids"]))
        print("\nlabel text:",decode_tokenids(first_batch["labels_ids"][0][:400].tolist(),1))
        # 示例：
        # model_inputs = {
        #     "input_ids": first_batch["output_ids"],
        #     "attention_mask": first_batch["attention_mask_ids"],
        #     "labels": first_batch["labels_ids"]
        # }
        # outputs = model(**model_inputs)
        # loss = outputs.loss
        # loss.backward()
        # ...

    except StopIteration:
        print("数据加载器为空，无法获取批次。")


if __name__ == '__main__':
    main()
