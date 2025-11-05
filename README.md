# GroupB
This is a cooperation project of training FlexMDM based on LLaDA-8B-base  
<img src="./image/shushu.jpg">


---
### Download our works
run following to git clone
```bash
git clone https://github.com/canxu-star/GroupB.git
```
---
## data_process
Data processing is required before training
```bash
python run_processing.py --split train --num_proc 8
python run_processing.py --split validation --num_proc 4
python run_processing.py --split test --output_dir ./custom_output
```

---
### Download requirements and models
run following to download requirements:
```bash
bash pip_pods.sh
```

---
### Run the train script
run following to train SFT model
```bash
bash train_SFT.sh
```

run following to train RL model
```bash
bash train_RL.sh
```

---
