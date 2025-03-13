#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torch --index-url https://download.pytorch.org/whl/cu118')
get_ipython().system('pip install transformers datasets accelerate peft bitsandbytes sentencepiece')
get_ipython().system('pip install evaluate')
get_ipython().system('pip install nltk')
get_ipython().system('pip install rouge_score')


# In[14]:


import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. PyTorch will use the GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. PyTorch will use the CPU.')


# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# Load DeepSeek model in 8-bit mode
model_name = "deepseek-ai/deepseek-llm-7b-base" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA for low-rank adaptation
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # LoRA applied to attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()


# In[31]:


from datasets import load_dataset
from transformers import AutoTokenizer

# Load SciTLDR dataset (FullText version)
dataset = load_dataset("allenai/scitldr", "FullText")

# Tokenization function
def tokenize_function(examples):
    # Convert list of sentences into a single string
    inputs = ["Prompt: " + " ".join(doc) for doc in examples["source"]]
    targets = [" ".join(summary) if isinstance(summary, list) else summary for summary in examples["target"]]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# In[11]:


import evaluate
from nltk.tokenize import word_tokenize

# Load evaluation metrics
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# Define a function to compute metrics
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE evaluation
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    # METEOR evaluation
    meteor_scores = meteor.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "meteor": meteor_scores["meteor"]
    }


# In[29]:


training_args = TrainingArguments(
    output_dir="./deepseek_scitldr_lora",
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Adjust based on VRAM (2 is safe for 1080 Ti)
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,  # Mixed precision training
    push_to_hub=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    label_names=["labels"]
)


# In[ ]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


# In[ ]:


results = trainer.evaluate(tokenized_datasets["test"])
print("Final Test Evaluation:")
print(results)


# In[ ]:


model.save_pretrained("./deepseek_scitldr_lora")
tokenizer.save_pretrained("./deepseek_scitldr_lora")


# In[ ]:


# ---------
model = AutoModelForCausalLM.from_pretrained("./deepseek_scitldr_lora")
tokenizer = AutoTokenizer.from_pretrained("./deepseek_scitldr_lora")
input_text = "Summarize: Deep learning has achieved state-of-the-art results in many NLP tasks."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))

huggingface-cli login  # Log in first
model.push_to_hub("your_username/deepseek_scitldr_lora")

