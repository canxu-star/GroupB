# ./src/configuration.py

from dataclasses import dataclass, field
from typing import Any, Dict

# 为了处理从字典到对象的自动转换，我们定义一个辅助函数
def from_dict(data_class, data):
    """递归地将字典转换为dataclass实例。"""
    if data is None:
        return data_class()
    
    # 获取dataclass的所有字段及其类型
    fields = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
    
    init_args = {}
    for key, value in data.items():
        if key in fields:
            field_type = fields[key]
            # 如果字段类型本身也是一个dataclass，则递归转换
            if hasattr(field_type, '__dataclass_fields__'):
                init_args[key] = from_dict(field_type, value)
            else:
                init_args[key] = value
                
    return data_class(**init_args)

@dataclass
class dataConfig:
    name: str = "Salesforce/wikitext"
    path: str = "./data/wikitext-2-raw-v1"
    save_path: str = "./data/processed_data"
    """
    using above parameter to point the datasets` path
    'path' has higher priority than 'name' when both are provided
    using 'path' to load local datasets
    using 'name' to load datasets from huggingface datasets hub
    processed data will be saved to 'save_path'
    """
    train_batch_size: int = 8
    micro_batch_size_per_gpu: int = 2
    """
    above parameter to set the batch size during training
    the 'micro..' only enables when you use multi-gpu to train the model
    the effective batch size = micro_batch_size_per_gpu * num_gpus
    """
    train_name: str = "train"
    val_name: str = "eval"
    """
    when you train,the script check the 'total_epochs' and combine it with 'save_path' and 'train/eval_name',
        the script will load the data from the path:'{save_path}/{train/eval_name}_{total_epochs}.parquet'
    """
    prompt_key: Any = None
    response_key: Any = None
    """
    declare above to avoid the addition of noise on prompt,
        then prompt will combine with response
    """
    max_length: int = 1024
    """
    the longest length of the tokenized,overflow part will be truncated
    """
    balance_dp_token: bool = True
    err_prob: float = 0.05
    """
    the probability of error injection during training
    TODO:
    尝试在不同的epoch使用不同的错误概率
    TODO:
    尝试不同的加错策略:平权/重要性
    """
    noise_prob_policy: str = "Linear"
    """
    the policy to adjust the noise probability during training
    above parameter can be Linear/Fixed/Importance to choose the policy of noising
    - Linear: linearly increase the noise probability from 0 to t(255) during training
    - Fixed: keep the noise probability to err_prob during training
    - Importance: adjust the noise probability according to the token importance
    TODO:
    重要性加噪和重要性加错可能需要大模型计算置信度
    """

@dataclass
class fsdpConfig:
    pass

@dataclass
class loraConfig:
    open: bool = False
    alpha: int = 16
    """
    above parameter to set the lora configuration
    open: whether to use lora during training
    alpha: the lora alpha parameter
    """
    rank: int = 8
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"])

@dataclass
class modelConfig:
    path_from_pretrained: str = "facebook/opt-1.3b"
    """
    The path or identifier of the pretrained model to load.
    """
    fsdp_config: fsdpConfig = field(default_factory=fsdpConfig)
    
    """
    Directory to save model checkpoints during training.
    """
    continue_train: bool = False
    lora: loraConfig = field(default_factory=loraConfig)

@dataclass
class OptimizerConfig:
    lr: float = 5e-5
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.01
    warmup_steps_ratio: float = 0.1
    clip_grad_norm: float = 1.0

@dataclass
class TrainingConfig:
    default_local_dir: str = "./sft_model"
    default_hdfs_dir: str = "hdfs://tmp/experiments/gsm8k/gemma-1.1-7b-it/"
    check_point_saves: str = "./checkpoints"
    project_name: str = "shushurun-sft"
    experiment_name: str = "test"
    total_epochs: int = 3
    total_training_steps: int = None
    logger: list = field(default_factory=lambda: ["console"])
    save_checkpoint_steps: int = 10
    early_stopping: dict = field(default_factory=lambda: {
        "open": True,
        "monitor": "eval_loss",
        "patience": 3,
        "delta": 0.01
    })

@dataclass
class diffusionConfig:
    diffusion_steps:int=64
    decoding_strategy:str="ddpm"
    token_reweight:bool=True
    alpha:float=0.5
    gamma:float=0.5
@dataclass
class TotalConfig:
    """顶级配置类，整合所有子配置。"""
    mode: str="train_sft"
    """
    above config can be tokenize/train_sft/train_rl to choose the running mode
    - tokenize: only do tokenization and data preparation.
        If you don`t have any dataset,you should use this mode to prepare you data because our dataloader is stastic
    - train_sft: supervised fine-tuning mode
    - train_rl: reinforcement learning mode,we use the Trace-RL
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: dataConfig = field(default_factory=dataConfig)
    model: modelConfig = field(default_factory=modelConfig)
    optim:OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer:TrainingConfig = field(default_factory=TrainingConfig)
    diffusion:diffusionConfig = field(default_factory=diffusionConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """
        一个类方法工厂,用于从字典创建AppConfig实例。
        """
        return from_dict(cls, data)