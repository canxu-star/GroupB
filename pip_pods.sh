pip install transformers
pip install flash-attn --no-build-isolation
pip install datasets
pip install hf-transfer

huggingface-cli download LoveFlowers793/MouseMDM \
  --local-dir ./model \
  --local-dir-use-symlinks False
