import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transformers import AutoModel,AutoTokenizer,get_scheduler
import json
import os
from tqdm import tqdm
# 一个示例的数据集读取库
from src import dataprepare

def load_config(config_path):
        """加载 JSON 配置文件"""
        if not os.path.exists(config_path):
            print(f"警告：配置文件 {config_path} 不存在，将使用代码中定义的默认值。")
            return {}
        with open(config_path, 'r') as f:
            return json.load(f)


def main():
    # 1. 加载 JSON 文件作为默认配置
    default_config = load_config('config.json')

    # 2. 创建 ArgumentParser 对象
    #    description 参数会在使用 -h 或 --help 时显示
    parser = argparse.ArgumentParser(description="一个读取 JSON 配置并接受命令行参数覆盖的示例脚本。")

    # 3. 添加参数，并使用从 JSON 加载的值作为默认值 (default)
    #    我们使用 .get() 方法来安全地获取值，以防 JSON 文件中缺少某个键
    parser.add_argument('--learning_rate', type=float, default=default_config.get('learning_rate'),
                        help=f"设置学习率 (默认: {default_config.get('learning_rate')})")

    parser.add_argument('--model_path', type=str, default=default_config.get('model_path'),
                        help=f"设置模型路径 (默认: {default_config.get('model_path')})")

    parser.add_argument('--epochs', type=int, default=default_config.get('epochs'),
                        help=f"设置训练轮数 (默认: {default_config.get('epochs')})")
    
    # 对于布尔值，使用 action='store_true' 或 'store_false' 更为常见
    # 这里我们演示一种更通用的方法，以便与JSON的 true/false 对应
    parser.add_argument('--use_gpu', type=lambda x: (str(x).lower() == 'true'), default=default_config.get('use_gpu', False),
                        help=f"是否使用 GPU (True/False, 默认: {default_config.get('use_gpu')})")


    # 4. 解析参数。此时，优先级已经自动处理好了。
    #    命令行 > JSON 默认值
    final_config = parser.parse_args()

    # 5. 打印并使用最终的配置
    print("---最终生效的配置---")
    print(f"学习率 (Learning Rate): {final_config.learning_rate}")
    print(f"模型路径 (Model path):   {final_config.model_path}")
    print(f"训练轮数 (Epochs):        {final_config.epochs}")
    print(f"是否使用 GPU (Use GPU):   {final_config.use_gpu}")
    print("------------------------")

    # 在这里，你可以基于 final_config 中的值执行你的主要任务
    # 例如：train_model(config=final_config)

    """
    1:初始化,包括device,
    """
    device=torch.device("cuda" if torch.cuda.is_available() and final_config.use_gpu else "cpu")
    model=AutoModel.from_pretrained(final_config.model_path)
    model.to(device)
    tokenizer=AutoTokenizer.from_pretrained(final_config.model_path)
    data_loader=dataprepare(tokenizer,final_config.dataset_path)
    optimizer=optim.AdamW(model.parameters(),lr=final_config.learning_rate)

    num_epochs = 3
    num_training_steps = num_epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    """
    2:损失定义和超参数设置
    """
    w1=final_config.w1
    w2=final_config.w2
    loss1=nn.CrossEntropyLoss()
    loss2=nn.MSELoss()

    """
    3:训练:
    """
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            # 将数据移动到GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(
            input_ids=batch["masked_text_ids"],
            attention_mask=batch["attention_mask"]
            # 我们不需要在这里传入labels，因为我们在外部计算loss
            )
        
            # ----------------- 损失计算核心 -----------------
            # a. 计算 loss1 (LLaDA风格)
            logits_head1 = outputs["logits"]
            clean_labels = batch["clean_text_ids"]
        
            # 关键：只对被mask的位置计算loss，这是LLaDA的精髓
            # masked_text_ids中被mask的位置通常有一个特定的ID (例如tokenizer.mask_token_id)
            mask_token_id = 126336 # 从你的special_tokens_map.json看，mask_token_id可能是这个或另一个值
            active_loss = batch["masked_text_ids"].view(-1) == mask_token_id
        
            active_logits = logits_head1.view(-1, model.config.vocab_size)[active_loss]
            active_labels = clean_labels.view(-1)[active_loss]
        
            loss1 = loss1(active_logits, active_labels)

            # b. 计算 loss2 (你的自定义任务)
            output_head2 = outputs["new_head_output"]
            labels_head2 = batch["label_for_head_2"]
            # 假设输出和标签已经是正确的形状
            loss2 = loss2(output_head2, labels_head2)

            # c. 结合损失
            total_loss = w1 * loss1 + w2 * loss2
            # ----------------------------------------------------

            # 反向传播
            total_loss.backward()

            # 更新权重
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}")




    pass

if __name__ == "__main__":
    main()