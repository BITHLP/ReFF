import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = 'llama3'
wandb_project = 'your-wandb-project'
run_name = 'your-run-name'
n_epoch = 3

import wandb
wandb.init(project=wandb_project, name=run_name)

model_path = 'path/to/model'
tokenizer_path = 'path/to/tokenizer'
json_dataset_path = 'path/to/dataset'

import random
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from format_reward import format_reward

def get_custom_dataset(path, tokenizer, thres):
    """
    This function loads a dataset from a JSON file, tokenizes the 'query' field, shuffles the dataset, filters out samples with input_ids length greater than the specified threshold, and sets the format to 'torch'.
    """
    dataset = load_dataset('json', data_files=path, split='train')
    dataset = dataset.shuffle(seed=42)
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample
    print('dataset size: {}'.format(len(dataset)))
    dataset = dataset.map(tokenize, batched=False)
    dataset = dataset.filter(lambda sample: len(sample) < thres, batched=False, input_columns=['input_ids'])
    print('dataset size: {}'.format(len(dataset)))
    dataset.set_format('torch')
    return dataset

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

if __name__ == '__main__':

    output_dir = os.path.join('./checkpoints', model_name, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model =AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map='auto',
        peft_config=peft_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_custom_dataset(json_dataset_path, tokenizer, thres=1536)

    ppo_config = PPOConfig(
        log_with='wandb',
        model_name=model_name,
        learning_rate=1.41e-5,
        remove_unused_columns=False,
        batch_size=32,
        mini_batch_size=2,
        gradient_accumulation_steps=16,
        init_kl_coef=0.05,
        horizon=1000,
        target=6,
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,
    }

    for epoch in range(n_epoch):

        print('epoch {}'.format(epoch))
        step = 0

        for batch in ppo_trainer.dataloader:
            print('step {}'.format(step))
            step += 1
            query_pts = batch['input_ids']
            response_pts = ppo_trainer.generate(
                query_pts,
                batch_size=8,
                return_prompt=False,
                **generation_kwargs,
            )
            batch['response'] = tokenizer.batch_decode(response_pts, skip_special_tokens=True)
            rewards = []
            for i in range(len(batch['query'])):
                t = batch['task'][i]
                q = batch[t+'_question'][i]
                a = batch['response'][i]
                rewards.append(format_reward(t, q, a))
            rewards = [torch.tensor(x, dtype=torch.float) for x in rewards]

            stats = ppo_trainer.step(query_pts, response_pts, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        save_dir = output_dir if n_epoch==0 else os.path.join(output_dir, 'epoch_{}'.format(epoch+1))
        ppo_trainer.save_pretrained(save_dir)