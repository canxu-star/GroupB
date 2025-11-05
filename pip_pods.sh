#!/bin/bash

pip install transformers
pip install flash-attn --no-build-isolation
pip install datasets
pip install hf-transfer

huggingface-cli download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir ./model \
  --local-dir-use-symlinks False

if [ -d "./MouseConfig" ]; then
    echo "覆盖 ./model 中的配置文件..."
    cp -r ./MouseConfig/* ./model/
    echo "配置文件覆盖完成。"
else
    echo "未找到 ./MouseConfig 文件夹，跳过覆盖。"
fi