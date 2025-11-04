import torch
import random
# from transformers import AutoTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional

# --- 配置参数 ---

TOKENIZER_NAME = "GSAI-ML/LLaDA-8B-Instruct"

# 数据集名称
DATASET_NAME = "Salesforce/wikitext"
DATASET_SUBSET = "wikitext-2-raw-v1"

# 特殊 Token ID
DEL_TOKEN_ID = 126339
PAD_TOKEN_ID = 126081
MASK_TOKEN_ID = 126336

# 处理参数
MIN_LEN = 50
MAX_LEN = 1024
FINAL_LEN = 2048
P1_CORRUPT_PROB = 0.15  # p1: token被改错的概率
P2_DROP_PROB = 0.3     # p2: 随机丢弃token的概率
P3_MASK_PROB = 0.5     # p3: 剩余token被mask的概率

ERROR_ID = 2
INVALID_ID = float("-inf")
EFFECTIVE_ID = 0

# 特殊token排除列表
special_ids_to_exclude = {DEL_TOKEN_ID, PAD_TOKEN_ID, MASK_TOKEN_ID}


def setup_tokenizer(name: str) -> AutoTokenizer:
    """加载并设置分词器，确保特殊token已定义。"""
    tokenizer = AutoTokenizer.from_pretrained(name)
    print("SPECIAL_TOKENS_ATTRIBUTES:",tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
    print("pad_token_id:",tokenizer.pad_token_id)
    # print("mask_token_id:",tokenizer.mask_token_id)
    if  tokenizer.bos_token_id is not None:
        special_ids_to_exclude.add(tokenizer.bos_token_id)
    if  tokenizer.eos_token_id is not None:
        special_ids_to_exclude.add(tokenizer.eos_token_id)
    if  tokenizer.unk_token_id is not None:
        special_ids_to_exclude.add(tokenizer.unk_token_id)
    if  tokenizer.sep_token_id is not None:
        special_ids_to_exclude.add(tokenizer.sep_token_id)
    if  tokenizer.pad_token_id is not None:
        special_ids_to_exclude.add(tokenizer.pad_token_id)
    if  tokenizer.cls_token_id is not None:
        special_ids_to_exclude.add(tokenizer.cls_token_id)
    if  tokenizer.mask_token_id is not None:
        special_ids_to_exclude.add(tokenizer.mask_token_id)
    print("tokenizer.SPECIAL_TOKENS_ATTRIBUTES:", special_ids_to_exclude)
    # LLaMA分词器通常没有默认的pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = PAD_TOKEN_ID
        
    special_tokens_to_add = []
    all_special_ids = tokenizer.all_special_ids
    if DEL_TOKEN_ID not in all_special_ids:
        special_tokens_to_add.append(f"<DEL:{DEL_TOKEN_ID}>")
    if MASK_TOKEN_ID not in all_special_ids:
        special_tokens_to_add.append(f"<MASK:{MASK_TOKEN_ID}>")
        
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def process_single_sample(
    text: str, 
    tokenizer: AutoTokenizer,
    vocab_size: int
) -> Optional[Dict[str, List[int]]]:
    """
    对单个文本样本执行完整的转换逻辑。
    """
    # 1. 分词和过滤
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not (MIN_LEN <= len(token_ids) <= MAX_LEN):
        return None

    # 2. 交错插入 'del' token 并创建初始 attention_mask
    original_len = len(token_ids)
    # print("original_len:",original_len)
    interleaved_ids = [INVALID_ID] * (original_len * 2)
    attention_mask = [INVALID_ID] * (original_len * 2)
    for i in range(original_len):
        interleaved_ids[2 * i] = token_ids[i]
        interleaved_ids[2 * i + 1] = DEL_TOKEN_ID
        attention_mask[2 * i] = EFFECTIVE_ID

    

    # 3. 按p1概率引入错误
    token_indices = list(range(original_len))
    num_to_corrupt = int(original_len * P1_CORRUPT_PROB)
    corrupt_indices = random.sample(token_indices, k=num_to_corrupt)
    # print("len, corrupt_indices:",len(corrupt_indices), corrupt_indices)
    for i in corrupt_indices:
        # 获取当前位置的原始token ID，也加入排除列表
        original_token_id = interleaved_ids[2 * i]
        exclude_ids = special_ids_to_exclude.union({original_token_id})

        # 随机选择一个错误的token ID，直到它不是特殊token或原始token
        random_token_id = random.randint(0, vocab_size - 1)
        while random_token_id in exclude_ids:
            random_token_id = random.randint(0, vocab_size - 1)
        
        interleaved_ids[2 * i] = random_token_id
        attention_mask[2 * i] = ERROR_ID      # 标记为改错
        attention_mask[2 * i + 1] = EFFECTIVE_ID  # 激活其后的del
    # 4. 创建用于计算loss的label
    labels = list(interleaved_ids)

    # 5. 按p2概率随机丢弃（在attention_mask中置INVALID_ID）
    effective_indices = [i for i, val in enumerate(attention_mask) if val == EFFECTIVE_ID]
    
    # 计算需要丢弃的数量
    num_to_drop = int(len(effective_indices) * P2_DROP_PROB)
    # print("丢弃的数量：", num_to_drop)
    # 随机选择并丢弃
    if num_to_drop > 0:
        drop_indices = random.sample(effective_indices, k=num_to_drop)
        for i in drop_indices:
            attention_mask[i] = INVALID_ID

    # 6. 按p3概率对剩余token进行mask
    # 找到所有被激活的DEL token的位置 (attention_mask == EFFECTIVE_ID 且是DEL_TOKEN_ID)
    activated_del_indices = [i for i, val in enumerate(attention_mask) if val == EFFECTIVE_ID and interleaved_ids[i] == DEL_TOKEN_ID]
    # print("del-mask的数量：",len(activated_del_indices))
    # 将所有被激活的DEL token都替换为MASK token
    for i in activated_del_indices:
        interleaved_ids[i] = MASK_TOKEN_ID

    # 找到剩余的、未被修改的普通token的位置 (attention_mask == 1 且不是DEL_TOKEN_ID)
    remaining_normal_token_indices = [i for i, val in enumerate(attention_mask) if val == EFFECTIVE_ID and interleaved_ids[i] != DEL_TOKEN_ID]
    # print("remaining_normal_token_indices", len(remaining_normal_token_indices))
    # 在这些剩余的普通token中，按P3概率随机选择一部分进行mask
    num_to_mask = int(len(remaining_normal_token_indices) * P3_MASK_PROB)
    # print("num_to_mask", num_to_mask)
    if num_to_mask > 0:
        mask_indices = random.sample(remaining_normal_token_indices, k=num_to_mask)
        # print("mask_indices", len(mask_indices))
    
        for i in mask_indices:
            interleaved_ids[i] = MASK_TOKEN_ID

    # 7. 填充到最终长度
    current_len = len(interleaved_ids)
    if current_len < FINAL_LEN:
        padding_len = FINAL_LEN - current_len
        interleaved_ids.extend([PAD_TOKEN_ID] * padding_len)
        attention_mask.extend([INVALID_ID] * padding_len)
        labels.extend([PAD_TOKEN_ID] * padding_len)

    return {
        "input_ids": token_ids,
        "output_ids": interleaved_ids,
        "attention_mask_ids": attention_mask,
        "labels_ids": labels,
    }


def get_processed_dataset(
    split: str,
    num_proc: int = 16,
    cache_dir: Optional[str] = None
) -> Dataset:
    """
    加载、处理并返回最终的数据集。
    """
    print("--- Setting up tokenizer ---")
    tokenizer = setup_tokenizer(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size

    print(f"--- Loading dataset {DATASET_NAME}/{DATASET_SUBSET} [{split}]   vocabulary size: {vocab_size} ---")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, cache_dir=cache_dir, split=split)
    print("原始数据集：",dataset.shape)
    def preprocess_batch(batch: Dict[str, List]) -> Dict[str, List]:
        """
        用于 .map() 的批处理函数。
        """
        processed_samples = [process_single_sample(text, tokenizer, vocab_size) for text in batch["text"]]
        
        # 过滤掉返回None的样本 (长度不符合要求的)
        valid_samples = [s for s in processed_samples if s is not None]
        
        if not valid_samples:
            return {"input_ids": [], "output_ids": [], "attention_mask_ids": [], "labels_ids": []}

        # 将样本列表转换为批次字典
        batch_result = {
            "input_ids": [s["input_ids"] for s in valid_samples],
            "output_ids": [s["output_ids"] for s in valid_samples],
            "attention_mask_ids": [s["attention_mask_ids"] for s in valid_samples],
            "labels_ids": [s["labels_ids"] for s in valid_samples],
        }
        return batch_result

    print("--- Starting dataset processing ---")
    processed_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,  # 较小的批次大小，因为单样本处理逻辑复杂
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=True, # 建议开启以复用结果
        desc="Running data processing"
    )

    # 设置为torch格式，方便后续在PyTorch中使用
    processed_dataset.set_format("torch")
    
    print("--- Processing finished ---")
    return processed_dataset
def decode_tokenids(token_ids: List[List[int]],tag:bool) -> List[str]:
    """
    Decode a sequence with visible mask tokens.
    """
    tokenizer = setup_tokenizer(TOKENIZER_NAME)
    if(tag):
        token_ids=token_ids[::2]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text
   


if __name__ == '__main__':
    # --- 使用示例 ---
    
    # 获取处理后的训练集
    # 使用多核CPU可以显著提速，请根据您的机器配置调整 num_proc
    train_dataset = get_processed_dataset(split="train", num_proc=8)

    print("\n--- Processed Dataset Info ---")
    print(f"train_dataset.shape: {train_dataset.shape}")

    # 打印第一个样本以供检查
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("\n--- First Sample ---")
        print("input_ids shape:", sample["input_ids"].shape)
        print("output_ids IDs shape:", sample["output_ids"].shape)
        print("Attention Mask shape:", sample["attention_mask_ids"].shape)
        print("Labels shape:", sample["labels_ids"].shape)
        print("\ninput_ids IDs (first 400):", sample["input_ids"][:400].tolist())
        print("\noutput_ids (first 400):", sample["output_ids"][:400].tolist())
        print("\nAttention Mask (first 400):", sample["attention_mask_ids"][:400].tolist())
        print("\nLabels (first 400):", sample["labels_ids"][:400].tolist())


        print("\ninput text:",decode_tokenids(sample["input_ids"][:400].tolist(),0))
        print("\noutput text:",decode_tokenids(sample["output_ids"][:400].tolist(),1))
        # print("Attention Mask text:",decode_tokenids(sample["input_ids"]))
        print("\nlabel text:",decode_tokenids(sample["labels_ids"][:400].tolist(),1))
    
        # 检查 attention_mask 中的值
        unique_values = torch.unique(sample["attention_mask_ids"])
        print("\nUnique values in attention mask:", unique_values.tolist())

        # tokenizer

