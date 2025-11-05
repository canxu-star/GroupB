import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transformers import AutoModel, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from configuration import TotalConfig
import sys
from datasets import load_from_disk
# 一个示例的数据集读取库
# from src import dataprepare

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
    parser = argparse.ArgumentParser(description="动态参数解析器")
    _, unknown_args = parser.parse_known_args()
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg.lstrip('-')
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith('--'):
                value_str = unknown_args[i+1]
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        if value_str.lower() == 'true':
                            value = True
                        elif value_str.lower() == 'false':
                            value = False
                        else:
                            value = value_str
                overrides[key] = value
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1
    return overrides

def update_config_with_nested_keys(defaults, overrides):
    for flat_key, value in overrides.items():
        keys = flat_key.split('.')
        d = defaults
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    return defaults

def generate_labels_for_head2(attention_mask: torch.Tensor, output_ids: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    is_neg_inf = torch.isneginf(attention_mask)
    is_not_eos = output_ids != 126081
    is_special_marker_2 = attention_mask == 2
    for b in range(batch_size):
        cumulative_sum = 0
        for i in range(seq_len - 1, -1, -1):
            if not is_neg_inf[b, i]:
                labels[b, i] = cumulative_sum
                cumulative_sum = 0
            else:
                if i % 2 == 0 and is_not_eos[b, i]:
                    cumulative_sum += 1
                elif i > 0 and i % 2 != 0 and is_special_marker_2[b, i - 1]:
                    cumulative_sum += 1
    return labels

# ****** 新增: 工具函数 - bf16 支持检测与转换 ******
def device_supports_bf16(device: torch.device) -> bool:
    """检查当前设备是否支持 bfloat16(主要针对 CUDA GPU)。"""
    # 在较新版本 PyTorch，CUDA 设备通常会支持 bfloat16（硬件相关）
    # 这里采用保守检测：若 CUDA 可用并且 torch.cuda.is_available() 则假定支持 bf16（你可以改成更严格的检查）
    if device.type == 'cuda' and torch.cuda.is_available():
        # 如果需要更强的约束，可以通过 capability 或 torch.cuda.get_device_properties(n).major 来判断
        return True
    return False

def cast_model_to_bf16(model):
    """把模型的参数和缓冲全部转换为 bfloat16（原位转换）。"""
    for param in model.parameters():
        param.data = param.data.to(dtype=torch.bfloat16)
        if param.grad is not None:
            param.grad = param.grad.to(dtype=torch.bfloat16)
    # buffers（如 layernorm 的 running stats）保持原 dtype（通常 fp32），不建议转换
    # 如果想也转换 buffers，可在此处处理 model.buffers()
    return model

def cast_grads_to_bf16(model):
    """显式将所有参数的 grad 转为 bf16（通常在 backward 后调用）。"""
    for p in model.parameters():
        if p.grad is not None:
            # 有时 p.grad 是稀疏或不同 dtype，做安全转换
            p.grad = p.grad.to(dtype=torch.bfloat16)

def cast_optimizer_state_to_bf16(optimizer):
    """将 optimizer.state 的 tensor 值转换为 bfloat16（若存在）。"""
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    try:
                        # 有些状态张量必须保留为浮点类型
                        state[k] = v.to(dtype=torch.bfloat16)
                    except Exception:
                        # 如果转换失败，保持原样
                        pass

# ****** 新增结束 ******

# ****** 新增开始: 评估函数（保持原样，但确保 eval 下无 autocast 影响） ******
def evaluate_model(model, dataloader, device, loss1_fn, loss2_fn, w1, w2):
    model.eval()
    print("the model have been switched to eval model!")
    total_eval_loss = 0
    sum=0
    with torch.no_grad():
        print("no_grad:open!")
        for batch in dataloader:
            print("batch{sum} is evaluating!")
            batch = {k: v.to(device) for k, v in batch.items()}
            original_attention_mask = batch["attention_mask_ids"]
            print("got the attn mask!")
            attention_bias_for_model = original_attention_mask.clone()
            attention_bias_for_model[attention_bias_for_model == 2] = 0

            # 在验证里同样使用 autocast(bfloat16) 以保持一致（若设备不支持则不会启用）
            if device_supports_bf16(device):
                print("autocast()is running!")
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch["output_ids"],
                        attention_bias=attention_bias_for_model,
                        pos_idx=batch["attention_mask_ids"]
                    )
            else:
                outputs = model(
                    input_ids=batch["output_ids"],
                    attention_bias=attention_bias_for_model,
                    pos_idx=batch["attention_mask_ids"]
                )

            logits_head1 = outputs.logits
            clean_labels = batch["labels_ids"]
            mask_token_id = 126336
            active_loss = batch["output_ids"].view(-1) == mask_token_id
            active_logits = logits_head1.view(-1, model.config.vocab_size)[active_loss]
            active_labels = clean_labels.view(-1)[active_loss]
            print("calculate over")
            if active_logits.shape[0] > 0:
                loss1_val = loss1_fn(active_logits, active_labels)
            else:
                loss1_val = torch.tensor(0.0, device=device)

            logits_head2 = outputs.insert_logits
            labels_head2 = generate_labels_for_head2(
                attention_mask=original_attention_mask,
                output_ids=batch["output_ids"]
            )
            loss2_val = loss2_fn(logits_head2.view(-1, model.config.vocab_size), labels_head2.view(-1))

            total_loss = w1 * loss1_val + w2 * loss2_val
            total_eval_loss += total_loss.item()

    model.train()
    return total_eval_loss / len(dataloader)
# ****** 新增结束 ******

def main():
    output_dir = "./data/processed_data"
    train_dataset_path = os.path.join(output_dir, "wikitext-2-raw-v1-processed-train")
    eval_dataset_path = os.path.join(output_dir, "wikitext-2-raw-v1-processed-validation")

    default_config = load_yaml_config('./src/config.yaml')
    bash_overrides = parse_bash_overrides()
    final_config = update_config_with_nested_keys(default_config, bash_overrides)
    config = TotalConfig.from_dict(final_config)

    print("--- 最终生效的配置 ---")
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

    # 加载模型（先用默认 dtype，然后按需转换到 bf16）
    model = AutoModel.from_pretrained(config.model.path,trust_remote_code=True)
    model.to(device)

    # 检查 bf16 支持
    use_bf16 = config.trainer.use_bf16 and device_supports_bf16(device)
    if not use_bf16:
        print("警告: 目标设备不支持 bf16 或配置禁用 bf16，训练将在默认精度下进行。")

    # 如果希望把参数都转为 bf16（显存更小），可以做显式转换
    if use_bf16:
        # 参数转换为 bf16（注意：buffers 如 running_mean/running_var 建议保持 fp32）
        # 先把模型参数 & buffers 移到 device（已执行），再做 dtype 转换
        for name, buf in model.named_buffers():
            # 跳过一些不宜转换的 buffer（比如 batchnorm 统计量）
            # 这里我们只转换参数，buffers 保持原样，以保数值稳定
            pass
        # 转换模型参数 dtype（in-place）
        for param in model.parameters():
            param.data = param.data.to(dtype=torch.bfloat16, device=device)

    # 分离参数组（保持原来的分组逻辑）
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

    # 加载数据集
    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    train_dataloader = DataLoader(train_dataset, batch_size=config.data.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.data.train_batch_size, shuffle=False)

    # 优化器（保持原有实现）
    lr = config.optim.lr
    if isinstance(lr, str):
        lr = float(lr)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 若 user 指定要在 optimizer 中也尽可能使用 bf16，我们在创建 optimizer 后尽量把 state 转为 bf16
    # 注意：optimizer.state 在第一次 step 之前为空。我们在每次 step 后显式转换 state dtype（见训练循环）
    num_epochs = config.trainer.total_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    w1 = config.trainer.w1
    w2 = config.trainer.w2
    loss1_fn = nn.CrossEntropyLoss()
    loss2_fn = nn.CrossEntropyLoss()

    eval_steps = 40
    early_stopping_patience = 3
    early_stopping_threshold = 0.01
    early_stopping_counter = 0
    best_eval_loss = float('inf')
    global_step = 0
    training_should_stop = False

    checkpoint_dir = config.trainer.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    progress_bar = tqdm(range(num_training_steps))
    model.train()

    # 主训练循环：使用 autocast(bfloat16) for forward；在 backward 后把 grads 转为 bf16；在 optimizer.step 后把 optimizer state 转为 bf16
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            original_attention_mask = batch["attention_mask_ids"]

            attention_bias_for_model = original_attention_mask.clone()
            attention_bias_for_model[attention_bias_for_model == 2] = 0

            # 前向：若支持 bf16，则在 autocast(bfloat16) 环境中 forward；否则正常 forward
            if use_bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch["output_ids"],
                        attention_bias=attention_bias_for_model,
                        pos_idx=batch["attention_mask_ids"]
                    )
            else:
                outputs = model(
                    input_ids=batch["output_ids"],
                    attention_bias=attention_bias_for_model,
                    pos_idx=batch["attention_mask_ids"]
                )

            # loss 计算（与原逻辑一致）
            logits_head1 = outputs.logits
            clean_labels = batch["labels_ids"]
            mask_token_id = 126336
            active_loss = batch["output_ids"].view(-1) == mask_token_id
            active_logits = logits_head1.view(-1, model.config.vocab_size)[active_loss]
            active_labels = clean_labels.view(-1)[active_loss]

            if active_logits.shape[0] > 0:
                loss1_val = loss1_fn(active_logits, active_labels) / active_labels.numel()
            else:
                loss1_val = torch.tensor(0.0, device=device)

            logits_head2 = outputs.insert_logits
            labels_head2 = generate_labels_for_head2(
                attention_mask=original_attention_mask,
                output_ids=batch["output_ids"]
            )
            token_count_head2 = (attention_bias_for_model == 0).sum()
            if token_count_head2 > 0:
                loss2_val = loss2_fn(
                    logits_head2.view(-1, model.config.vocab_size),
                    labels_head2.view(-1)
                ) / token_count_head2
            else:
                loss2_val = torch.tensor(0.0, device=device)
            """
            loss2_val = loss2_fn(logits_head2.view(-1, model.config.vocab_size), labels_head2.view(-1))
            """
            total_loss = w1 * loss1_val + w2 * loss2_val

            # backward：在 autocast 下 backward（若 use_bf16），然后显式把梯度转换为 bf16（以确保 optimizer state 后续也是 bf16）
            total_loss.backward()

            # 把梯度转为 bf16（如果设备支持并且希望这么做）
            if use_bf16:
                cast_grads_to_bf16(model)

            # 你的特殊梯度缩放逻辑（保持不变）
            with torch.no_grad():
                for param in head1_params:
                    if param.grad is not None:
                        param.grad.div_(w1)
                for param in head2_params:
                    if param.grad is not None:
                        param.grad.div_(w2)

            # optimizer step：执行更新
            optimizer.step()
            # 在 step 后把 optimizer.state 内的张量尽量转换为 bf16（若你希望 optimizer state 常驻 bf16）
            """
            if use_bf16:
                cast_optimizer_state_to_bf16(optimizer)

            """

            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Step {global_step}, Loss: {total_loss.item():.4f},Loss1: {loss1_val.item():.4f}, Loss2: {loss2_val.item():.4f}")

            if global_step % eval_steps == 0:
                current_eval_loss = evaluate_model(model, eval_dataloader, device, loss1_fn, loss2_fn, w1, w2)
                print(f"\nStep {global_step}: Eval Loss = {current_eval_loss:.4f}, Best Eval Loss = {best_eval_loss:.4f}")

                if current_eval_loss < best_eval_loss:
                    print(f"Eval loss improved from {best_eval_loss:.4f} to {current_eval_loss:.4f}. Saving model...")
                    best_eval_loss = current_eval_loss
                    # 保存模型时注意：HuggingFace 的 save_pretrained 可能期望参数为 fp32/float32
                    # 为安全起见，我们在保存前把模型参数临时转换回 fp32
                    if use_bf16:
                        # 保存前把参数复制为 fp32 到 CPU 再保存（避免直接覆写 GPU bf16 参数）
                        save_temp = {n: p.detach().to(dtype=torch.float32, device='cpu') for n, p in model.named_parameters()}
                        # 将这些数据回写到 model 的 state_dict 再保存（更安全的方法是使用 state_dict 并手工转换）
                        # 这里用简单方式：先转换模型到 cpu float32，保存，再恢复到 cuda bf16
                        model_cpu = AutoModel.from_pretrained(config.model_path)  # reload base to get same structure
                        # 更简单：直接使用 model.state_dict(), 然后手动转换并保存
                        sd = model.state_dict()
                        sd_fp32 = {k: v.to(dtype=torch.float32, device='cpu') for k, v in sd.items()}
                        torch.save(sd_fp32, os.path.join(checkpoint_dir, f"pytorch_model_step{global_step}.pt"))
                        print(f"Saved bf16->fp32 converted state to {checkpoint_dir}")
                    else:
                        model.save_pretrained(checkpoint_dir)
                    early_stopping_counter = 0
                elif current_eval_loss > best_eval_loss + early_stopping_threshold:
                    early_stopping_counter += 1
                    print(f"Eval loss did not improve for {early_stopping_counter} time(s).")
                else:
                    early_stopping_counter = 0

                if early_stopping_counter >= early_stopping_patience:
                    print(f"Stopping training early after {early_stopping_patience} evaluations without improvement.")
                    training_should_stop = True
                    break
        if training_should_stop:
            break

if __name__ == "__main__":
    main()
