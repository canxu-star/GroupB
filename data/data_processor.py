import torch
import random
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional, Tuple, Set
from argparse import Namespace

def setup_tokenizer(name: str, pad_token_id: int, initial_exclude_ids: Set[int]) -> Tuple[AutoTokenizer, Set[int]]:
    """
    加载分词器并返回分词器对象和完整的特殊ID排除列表。
    """
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = pad_token_id
    
    # 从初始列表开始
    special_ids_to_exclude = initial_exclude_ids.copy()

    # 动态添加分词器自带的特殊token ID
    # --- START OF ADDED BLOCK ---
    if tokenizer.bos_token_id is not None:
        special_ids_to_exclude.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids_to_exclude.add(tokenizer.eos_token_id)
    if tokenizer.unk_token_id is not None:
        special_ids_to_exclude.add(tokenizer.unk_token_id)
    if tokenizer.sep_token_id is not None:
        special_ids_to_exclude.add(tokenizer.sep_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids_to_exclude.add(tokenizer.pad_token_id)
    if tokenizer.cls_token_id is not None:
        special_ids_to_exclude.add(tokenizer.cls_token_id)
    if tokenizer.mask_token_id is not None:
        special_ids_to_exclude.add(tokenizer.mask_token_id)
    # --- END OF ADDED BLOCK ---

    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Final set of special IDs to exclude: {special_ids_to_exclude}")
    
    return tokenizer, special_ids_to_exclude

def process_single_sample(
    text: str, 
    tokenizer: AutoTokenizer,
    config: Namespace
) -> Optional[Dict[str, List[int]]]:
    """对单个文本样本执行完整的转换逻辑。"""
    vocab_size = tokenizer.vocab_size
    
    # 特殊 Token ID
    DEL_TOKEN_ID = config.special_tokens.del_token_id
    PAD_TOKEN_ID = config.special_tokens.pad_token_id
    MASK_TOKEN_ID = config.special_tokens.mask_token_id

    MIN_LEN = config.processing_params.min_len
    MAX_LEN = config.processing_params.max_len
    FINAL_LEN = config.processing_params.final_len
    P1_CORRUPT_RATE = config.augmentation_probs.p1_corrupt_rate
    P2_DROP_RATE = config.augmentation_probs.p2_drop_rate
    P3_MASK_RATE = config.augmentation_probs.p3_mask_rate
    
    attn_ids = config.attention_mask_ids
    ERROR_ID = attn_ids.error_id
    INVALID_ID = attn_ids.invalid_id
    EFFECTIVE_ID = attn_ids.effective_id

    # 1. 分词和过滤
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not (MIN_LEN <= len(token_ids) <= MAX_LEN):
        return None

    # 2. 交错插入 'del' token 并创建初始 attention_mask
    original_len = len(token_ids)
    interleaved_ids = [INVALID_ID] * (original_len * 2)
    attention_mask = [INVALID_ID] * (original_len * 2)
    for i in range(original_len):
        interleaved_ids[2 * i] = token_ids[i]
        interleaved_ids[2 * i + 1] = DEL_TOKEN_ID
        attention_mask[2 * i] = EFFECTIVE_ID

    # 3. 按p1概率引入错误
    special_ids_to_exclude = config.special_ids_to_exclude # 从config获取完整的排除列表
    token_indices = list(range(original_len))
    num_to_corrupt = int(original_len * P1_CORRUPT_RATE)
    corrupt_indices = random.sample(token_indices, k=num_to_corrupt)
    
    for i in corrupt_indices:
        original_token_id = interleaved_ids[2 * i]
        exclude_ids = special_ids_to_exclude.union({original_token_id})
        # 随机选择一个错误的token ID，直到它不是特殊token或原始token
        random_token_id = random.randint(0, vocab_size - 1)
        while random_token_id in exclude_ids:
            random_token_id = random.randint(0, vocab_size - 1)
        
        interleaved_ids[2 * i] = random_token_id
        attention_mask[2 * i] = ERROR_ID
        attention_mask[2 * i + 1] = EFFECTIVE_ID
    # 4. 创建用于计算loss的label   
    labels = list(interleaved_ids)

    # 5. 按p2概率随机丢弃
    effective_indices = [i for i, val in enumerate(attention_mask) if val == EFFECTIVE_ID]
    num_to_drop = int(len(effective_indices) * P2_DROP_RATE)
    if num_to_drop > 0:
        drop_indices = random.sample(effective_indices, k=num_to_drop)
        for i in drop_indices:
            attention_mask[i] = INVALID_ID

    # 6. 按p3概率对剩余token进行mask
    # 找到所有被激活的DEL token的位置
    activated_del_indices = [i for i, val in enumerate(attention_mask) if val == EFFECTIVE_ID and interleaved_ids[i] == DEL_TOKEN_ID]
    for i in activated_del_indices:
        interleaved_ids[i] = MASK_TOKEN_ID

    remaining_normal_token_indices = [i for i, val in enumerate(attention_mask) if val == EFFECTIVE_ID and interleaved_ids[i] != DEL_TOKEN_ID]
    num_to_mask = int(len(remaining_normal_token_indices) * P3_MASK_RATE)
    if num_to_mask > 0:
        mask_indices = random.sample(remaining_normal_token_indices, k=num_to_mask)
        for i in mask_indices:
            interleaved_ids[i] = MASK_TOKEN_ID

    # 7.1 填充/截断 interleaved_ids
    current_len_interleaved = len(interleaved_ids)
    if current_len_interleaved < FINAL_LEN:
        interleaved_ids.extend([PAD_TOKEN_ID] * (FINAL_LEN - current_len_interleaved))
    interleaved_ids = interleaved_ids[:FINAL_LEN]

    # 7.2 填充/截断 attention_mask
    current_len_attn = len(attention_mask)
    if current_len_attn < FINAL_LEN:
        attention_mask.extend([INVALID_ID] * (FINAL_LEN - current_len_attn))
    attention_mask = attention_mask[:FINAL_LEN]

    # 7.3 填充/截断 labels (这是最关键的修复)
    current_len_labels = len(labels)
    if current_len_labels < FINAL_LEN:
        labels.extend([PAD_TOKEN_ID] * (FINAL_LEN - current_len_labels))
    labels = labels[:FINAL_LEN]

    # # 7.4 填充/截断原始的 token_ids
    # current_len_orig = len(token_ids)
    # if current_len_orig < FINAL_LEN:
    #     token_ids.extend([PAD_TOKEN_ID] * (FINAL_LEN - current_len_orig))
    # token_ids = token_ids[:FINAL_LEN]

    return {
        # "input_ids": token_ids,
        "output_ids": interleaved_ids,
        "attention_mask_ids": attention_mask,
        "labels_ids": labels,
    }

def get_processed_dataset(
    config: Namespace,
    split: str,
) -> Dataset:
    """加载、处理并返回最终的数据集。"""
    print("--- Setting up tokenizer ---")
    # 从config中获取您定义的特殊token
    initial_exclude_ids = set(vars(config.special_tokens).values())
    
    tokenizer, exclude_set = setup_tokenizer(
        config.identifiers.tokenizer_name, 
        config.special_tokens.pad_token_id,
        initial_exclude_ids
    )
    # 将完整的排除列表存入config，以便传递给 process_single_sample
    config.special_ids_to_exclude = exclude_set

    print(f"--- Loading dataset {config.identifiers.dataset_name}/{config.identifiers.dataset_subset} [{split}] ---")
    dataset = load_dataset(
        config.identifiers.dataset_name, 
        config.identifiers.dataset_subset, 
        cache_dir=config.defaults.cache_dir,
        split=split, 
        trust_remote_code=True
    )

    def preprocess_batch(batch: Dict[str, List]) -> Dict[str, List]:
        processed_samples = [process_single_sample(text, tokenizer, config) for text in batch["text"]]
        valid_samples = [s for s in processed_samples if s is not None]
        
        if not valid_samples:
            return { "output_ids": [], "attention_mask_ids": [], "labels_ids": []}

        return {
            # "input_ids": [s["input_ids"] for s in valid_samples],
            "output_ids": [s["output_ids"] for s in valid_samples],
            "attention_mask_ids": [s["attention_mask_ids"] for s in valid_samples],
            "labels_ids": [s["labels_ids"] for s in valid_samples],
        }

    print(f"--- Starting dataset processing for split '{split}' ---")
    processed_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        num_proc=config.defaults.num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc=f"Processing {split} split"
    )

    processed_dataset.set_format("torch")
    print(f"--- Processing for split '{split}' finished ---")
    return processed_dataset
