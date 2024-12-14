import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = 'llama3'
wandb_project = 'your-wandb-project'
run_name = 'your-run-name'
os.environ["WANDB_PROJECT"] = wandb_project

import wandb
wandb.init(project=wandb_project, name=run_name)

model_path = 'path/to/model'
tokenizer_path = 'path/to/tokenizer'
json_dataset_path = 'path/to/dataset'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import LoraConfig

def get_custom_dataset(path):
    dataset = load_dataset('json', data_files=path, split='train')
    return dataset

def length_filter(dataset, tokenizer, thres):
    length_distribution = {}
    tokenized_dataset = [tokenizer.encode(x['text']) for x in dataset]
    lens = [len(x) for x in tokenized_dataset]
    for length in lens:
        length_distribution[length // 100] = length_distribution.get(length // 100, 0) + 1
    print("Length distribution:")
    for length_range, frequency in sorted(length_distribution.items(), key=lambda x: x[0]):
        start = length_range * 100
        end = (length_range + 1) * 100 - 1
        print(f"Length range: {start}-{end}, Frequency: {frequency}")
    ok = [x<=thres for x in lens]
    print('{}% within length treshold'.format(sum(ok)/len(ok)*100))
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x['text']))<=thres)
    return dataset

if __name__ == '__main__':

    output_dir = os.path.join('./checkpoints', model_name, run_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_custom_dataset(json_dataset_path)
    dataset = length_filter(dataset, tokenizer, thres=768)
    dataset = dataset.shuffle(seed=42)

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,
        # logging strategies
        logging_strategy='steps',
        logging_steps=1,
        report_to='wandb',
        run_name=run_name,
        # evaluation strategies
        evaluation_strategy='no',
        # save strategies
        save_strategy='epoch',
        save_steps=1,
        # hyper-parameters
        optim='adamw_torch_fused',
        learning_rate=2e-5,
        warmup_ratio=0.0,
        lr_scheduler_type='linear',
        num_train_epochs=1,
        gradient_accumulation_steps=256,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template='Answer:\n',
            tokenizer=tokenizer,
        ),
    )

    trainer.train()