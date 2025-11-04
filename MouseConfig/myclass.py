import torch
import torch.nn as nn
from typing import Optional

# 假设 LLaDAModelLM 已经可以被导入
from modeling_mouse import MouseModelLM 

class LLaDAWithSequenceHead(nn.Module):
    """
    一个包装器，为 LLaDA 模型添加一个序列级别的输出头。
    
    参数:
        base_model: 一个预训练好的 LLaDAModelLM 实例。
        output_dim: 最终输出向量的维度。
                    例如,对于3分类任务,output_dim=3;
                    对于回归任务,output_dim=1。
    """
    def __init__(self, base_model, output_dim: int):
        super().__init__()
        self.mouse = base_model
        
        # 获取基础模型的隐藏层维度
        hidden_size = self.mouse.config.d_model

        # 定义新的输出头
        # 这是一个简单的线性层，将聚合后的序列向量映射到最终的输出维度
        self.sequence_head = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None
    ):
        """
        前向传播函数。
        """
        # 1. 通过基础的 LLaDA 模型获取输出
        #    我们必须设置 output_hidden_states=True 才能访问所有token的隐藏状态
        outputs = self.llada(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 2. 提取最后一层的隐藏状态
        #    它的形状是 [batch_size, seq_len, d_model]
        last_hidden_state = outputs.hidden_states[-1]
        
        # 3. 对隐藏状态进行平均池化 (Mean Pooling)
        #    这是将序列信息聚合为单个向量的关键步骤
        #    我们必须使用 attention_mask 来确保只对有效token进行平均，忽略[PAD]token
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # 将 attention_mask 从 [batch, seq_len] 扩展到 [batch, seq_len, d_model]
        # 以便进行逐元素的乘法
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # 将所有被掩码（padding）的token向量置为0
        masked_hidden_state = last_hidden_state * mask_expanded
        
        # 计算所有有效token向量的和
        sum_hidden = torch.sum(masked_hidden_state, dim=1)
        
        # 计算有效token的数量（为了安全，避免除以0）
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # 计算平均值，得到聚合后的序列向量
        pooled_output = sum_hidden / sum_mask # 形状: [batch_size, d_model]
        
        # 4. 将聚合后的向量送入我们自定义的输出头
        final_output = self.sequence_head(pooled_output) # 形状: [batch_size, output_dim]
        
        return final_output