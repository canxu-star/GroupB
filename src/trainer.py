import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transformers import AutoModel,AutoTokenizer,get_scheduler,DataLoader
import json
import os
from tqdm import tqdm
from configuration import TotalConfig
import sys
from data import load_from_disk
# 一个示例的数据集读取库
from src import dataprepare

def load_yaml_config(config_path):
    """加载YAML配置文件。"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_path}' 不存在。")
        sys.exit(1)

def parse_bash_overrides():
    """
    解析所有来自命令行的未知参数，并将它们转换为字典。
    例如: --learning_rate 0.05 --batch_size 64
    会转换成: {'learning_rate': '0.05', 'batch_size': '64'}
    """
    parser = argparse.ArgumentParser(description="动态参数解析器")
    # parse_known_args 会把所有未在 argparse 中定义的参数收集到一个列表中
    _, unknown_args = parser.parse_known_args()
    
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        # 确保它是一个参数键 (例如 --key)
        if arg.startswith('--'):
            key = arg.lstrip('-')
            # 确保后面还有一个值
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith('--'):
                value_str = unknown_args[i+1]
                # 尝试将值转换为更合适的类型（int, float）
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        # 处理布尔值
                        if value_str.lower() == 'true':
                            value = True
                        elif value_str.lower() == 'false':
                            value = False
                        else:
                            value = value_str # 保持为字符串
                overrides[key] = value
                i += 2
            else:
                # 处理像 --enable_feature 这样的开关参数
                overrides[key] = True
                i += 1
        else:
            i += 1
            
    return overrides

def update_config_with_nested_keys(defaults, overrides):
    """
    用包含点分键名的 overrides 字典递归更新 defaults 字典。
    """
    for flat_key, value in overrides.items():
        keys = flat_key.split('.')
        d = defaults
        # 遍历路径，除了最后一个键
        for key in keys[:-1]:
            # 如果路径不存在，则创建字典
            d = d.setdefault(key, {})
        # 在最深层设置值
        d[keys[-1]] = value
    return defaults

def generate_labels_for_head2(attention_mask: torch.Tensor, masked_text_ids: torch.Tensor) -> torch.Tensor:
    """
    Genera etiquetas para el segundo cabezal de salida usando un algoritmo de una sola pasada inversa.

    Args:
        attention_mask (torch.Tensor): El tensor de máscara de atención, de forma (batch_size, seq_len).
        masked_text_ids (torch.Tensor): El tensor de IDs de texto enmascarados, de forma (batch_size, seq_len).

    Returns:
        torch.Tensor: El tensor de etiquetas generado, de forma (batch_size, seq_len).
    """
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    
    # 1. Crear un tensor de etiquetas lleno de ceros
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Pre-calcular condiciones para eficiencia dentro del bucle
    is_neg_inf = torch.isneginf(attention_mask)
    is_not_eos = masked_text_ids != 126081  # ID del token EOS
    is_special_marker_2 = attention_mask == 2
    
    # Iterar sobre cada muestra en el lote
    for b in range(batch_size):
        # 2. Mantener un contador sum, inicializado a 0 para cada muestra
        cumulative_sum = 0
        
        # Iterar en reversa a través de la secuencia
        for i in range(seq_len - 1, -1, -1):
            
            # Regla C: Si la máscara no es -inf, asignar el sum y reiniciar
            if not is_neg_inf[b, i]:
                labels[b, i] = cumulative_sum
                cumulative_sum = 0
            # Si la máscara es -inf, aplicar las reglas de acumulación
            else:
                # Regla A: Índice par, -inf, y no es un token EOS
                if i % 2 == 0 and is_not_eos[b, i]:
                    cumulative_sum += 1
                
                # Regla B: Índice impar, -inf, y la máscara anterior es 2
                # Se necesita i > 0 para evitar un índice fuera de los límites
                elif i > 0 and i % 2 != 0 and is_special_marker_2[b, i - 1]:
                    cumulative_sum += 1
                    
    return labels

# ****** 新增开始: 评估函数 ******
def evaluate_model(model, dataloader, device, loss1_fn, loss2_fn, w1, w2):
    """在验证集上评估模型并返回平均损失。"""
    model.eval()  # 设置为评估模式
    total_eval_loss = 0
    
    with torch.no_grad():  # 在评估期间不计算梯度
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            original_attention_mask = batch["attention_mask"]
            
            attention_bias_for_model = original_attention_mask.clone()
            attention_bias_for_model[attention_bias_for_model == 2] = 0
            
            outputs = model(
                input_ids=batch["masked_text_ids"],
                attention_bias=attention_bias_for_model,
                pos_idx=batch["attention_mask"] # 假设pos_idx与attention_mask相同
            )
            
            # --- 损失计算逻辑 (与训练循环中完全一致) ---
            # loss1
            logits_head1 = outputs.logits
            clean_labels = batch["labels_ids"]
            mask_token_id = 126336
            active_loss = batch["masked_text_ids"].view(-1) == mask_token_id
            active_logits = logits_head1.view(-1, model.config.vocab_size)[active_loss]
            active_labels = clean_labels.view(-1)[active_loss]
            
            if active_logits.shape[0] > 0:
                loss1_val = loss1_fn(active_logits, active_labels)
            else:
                loss1_val = torch.tensor(0.0, device=device)

            # loss2
            logits_head2 = outputs.insert_logits
            labels_head2 = generate_labels_for_head2(
                attention_mask=original_attention_mask,
                masked_text_ids=batch["masked_text_ids"]
            )
            loss2_val = loss2_fn(logits_head2.view(-1, model.config.vocab_size), labels_head2.view(-1))
            
            total_loss = w1 * loss1_val + w2 * loss2_val
            total_eval_loss += total_loss.item()
            
    model.train()  # 评估结束后，恢复到训练模式
    return total_eval_loss / len(dataloader)
# ****** 新增结束: 评估函数 ******

def main():
    output_dir = "./processed_data"
    
    # ****** 修改开始: 定义训练和验证集路径 ******
    train_dataset_path = os.path.join(output_dir, "wikitext-2-raw-v1-processed-train")
    eval_dataset_path = os.path.join(output_dir, "wikitext-2-raw-v1-processed-validation") # 假设验证集文件名是这个
    # ****** 修改结束 ******
    
    # 1. 加载YAML配置文件 (默认参数)
    default_config = load_yaml_config('./src/config.yaml')
    
    # 2. 批量解析来自bash的所有覆盖参数
    bash_overrides = parse_bash_overrides()
    
    # 3. 合并配置，实现优先级 (bash > yaml)
    final_config = update_config_with_nested_keys(default_config, bash_overrides)

    # 4. 使用配置类来传递配置
    config = TotalConfig.from_dict(**final_config)

    print("--- 最终生效的配置 ---")
    print(config)

    """
    1:初始化,包括device,
    """
    device=torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
    model=AutoModel.from_pretrained(config.model_path)
    model.to(device)
    
    # 分离参数组
    base_model_params, head1_params, head2_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("model.transformer.ff_out"):
            head1_params.append(param)
        elif name.startswith("model.transformer.ff_proj_head2") or name.startswith("model.transformer.up_proj_head2") or name.startswith("model.transformer.ff_2out_head"):
            head2_params.append(param)
        else:
            base_model_params.append(param)

    total_params = len(list(filter(lambda p: p.requires_grad, model.parameters())))
    separated_params = len(base_model_params) + len(head1_params) + len(head2_params)
    assert total_params == separated_params, "参数分组不完整！"

    # ****** 修改开始: 加载训练和验证数据集 ******
    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)
        
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    train_dataloader = DataLoader(train_dataset, batch_size=config.trainer.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.trainer.batch_size, shuffle=False)
    # ****** 修改结束 ******
    
    optimizer=optim.AdamW(model.parameters(),lr=config.learning_rate)

    num_epochs = config.epochs
    # ****** 修改开始: 总步数按dataloader计算 ******
    num_training_steps = num_epochs * len(train_dataloader)
    # ****** 修改结束 ******
    
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    """
    2:损失定义和超参数设置
    """
    w1=config.trainer.w1
    w2=config.trainer.w2
    loss1_fn=nn.CrossEntropyLoss()
    loss2_fn=nn.CrossEntropyLoss()
    
    # ****** 新增开始: 提前停止和模型保存相关变量 ******
    eval_steps = 40
    early_stopping_patience = 3
    early_stopping_threshold = 0.01
    early_stopping_counter = 0
    best_eval_loss = float('inf')
    global_step = 0
    training_should_stop = False
    
    # 确保保存模型的目录存在
    checkpoint_dir = config.trainer.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    # ****** 新增结束 ******

    """
    3:训练:
    """
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            original_attention_mask = batch["attention_mask"]
            
            attention_bias_for_model = original_attention_mask.clone()
            attention_bias_for_model[attention_bias_for_model == 2] = 0

            outputs = model(
                input_ids=batch["masked_text_ids"],
                attention_bias=attention_bias_for_model, # 使用处理过的mask
                pos_idx=batch["attention_mask"]
            )
        
            # ****** 修改开始: 修正输出访问方式和损失变量名 ******
            # a. 计算 loss1 (LLaDA风格)
            logits_head1 = outputs.logits
            clean_labels = batch["labels_ids"]
        
            mask_token_id = 126336
            active_loss = batch["masked_text_ids"].view(-1) == mask_token_id
            active_logits = logits_head1.view(-1, model.config.vocab_size)[active_loss]
            active_labels = clean_labels.view(-1)[active_loss]
            
            if active_logits.shape[0] > 0:
                loss1_val = loss1_fn(active_logits, active_labels)
            else:
                loss1_val = torch.tensor(0.0, device=device)

            # b. 计算 loss2 (你的自定义任务)
            logits_head2 = outputs.insert_logits
            labels_head2 = generate_labels_for_head2(
                attention_mask=original_attention_mask,
                masked_text_ids=batch["masked_text_ids"]
            )
            loss2_val = loss2_fn(logits_head2.view(-1, model.config.vocab_size), labels_head2.view(-1))

            total_loss = w1 * loss1_val + w2 * loss2_val
            # ****** 修改结束 ******
            
            total_loss.backward()

            with torch.no_grad():
                for param in head1_params:
                    if param.grad is not None:
                        param.grad.div_(w1)
                for param in head2_params:
                    if param.grad is not None:
                        param.grad.div_(w2)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # ****** 修改开始: 更新进度条和步数 ******
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Step {global_step}, Loss: {total_loss.item():.4f}")
            # ****** 修改结束 ******
            
            # ****** 新增开始: 定期评估、提前停止和模型保存逻辑 ******
            if global_step % eval_steps == 0:
                current_eval_loss = evaluate_model(model, eval_dataloader, device, loss1_fn, loss2_fn, w1, w2)
                print(f"\nStep {global_step}: Eval Loss = {current_eval_loss:.4f}, Best Eval Loss = {best_eval_loss:.4f}")

                # 检查是否需要保存最佳模型
                if current_eval_loss < best_eval_loss:
                    print(f"Eval loss improved from {best_eval_loss:.4f} to {current_eval_loss:.4f}. Saving model...")
                    best_eval_loss = current_eval_loss
                    model.save_pretrained(checkpoint_dir)
                    early_stopping_counter = 0 # 只要有提升，就重置计数器
                # 检查提前停止条件 (允许损失有0.01的恶化容忍度)
                elif current_eval_loss > best_eval_loss + early_stopping_threshold:
                    early_stopping_counter += 1
                    print(f"Eval loss did not improve for {early_stopping_counter} time(s).")
                else: # 如果损失没有变得更差（或在容忍范围内），也重置计数器
                    early_stopping_counter = 0

                if early_stopping_counter >= early_stopping_patience:
                    print(f"Stopping training early after {early_stopping_patience} evaluations without improvement.")
                    training_should_stop = True
                    break # 跳出当前epoch的循环
            # ****** 新增结束 ******
            
        if training_should_stop:
            break # 跳出所有epoch的循环


if __name__ == "__main__":
    main()