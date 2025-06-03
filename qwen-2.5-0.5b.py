import pandas as pd 
import numpy as np
import torch 
import tiktoken 
import matplotlib.pyplot as plt
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, Qwen2ForSequenceClassification

data = pd.read_csv("train.csv")
label_to_text = {
    0:	"Algebra",
    1:	"Geometry and Trigonometry",
    2:	"Calculus and Analysis",
    3:	"Probability and Statistics",
    4:  "Number Theory",
    5:	"Combinatorics and Discrete Math",
    6:  "Linear Algebra",
    7:	"Abstract Algebra and Topology"
}
data['label_text'] = data['label'].map(label_to_text)
###
# format the dataset
###
dataset = []
def format_dataset(data):
    for i in range(len(data)):
        datapoint = []
        datapoint.append({"role": "user", "content": data['Question'][i]})
        datapoint.append({"role": "assistant", "content": str(data['label'][i])})
        dataset.append(datapoint)
format_dataset(data)
print(dataset[0])

## Configure the SFT (supervised fine-tuning)
sft_config = SFTConfig(
    # Group 1: memory-related
    gradient_checkpointing=True,    # this saves a LOT of memory
    gradient_checkpointing_kwargs={'use_reentrant': False}, 
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,  
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=4, 
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=True,
    ## GROUP 2: Dataset-related
    max_seq_length=6668,
    # Dataset
    # packing a dataset means no padding is needed
    packing=False,
    ## GROUP 3: These are typical training parameters
    num_train_epochs=2, #TODO: train only 2 epochs for now 
    learning_rate=3e-4,
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim='paged_adamw_8bit',       
    ## GROUP 4: Logging parameters
    logging_steps=10,
    logging_dir='./logs',
    output_dir='./qwen2.5-0.5b-v1',
    report_to='none'
)

num_labels = 8
# load the tokenizer and pretrained weights 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = Qwen2ForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B", num_labels=num_labels, problem_type="multi_label_classification"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset,
)
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print("inputs:", batch['input_ids'][0], "outputs:", batch['labels'][0])

###
# train for 2 epochs
###
trainer.train()