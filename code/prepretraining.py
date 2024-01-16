import argparse
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer , AutoModelForMaskedLM, set_seed
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler

import pandas as pd
import torch


class RePreDataset(Dataset):
    
    def __init__(self, tokenizer, data_path_list, block_size) :

        # path list를 전부 concat 후, sentence series를 list로 변환
        df_list = [pd.read_csv(path) for path in data_path_list]
        df = pd.concat(df_list)
        sentlist = df.sentence.to_list()

        batch_encoding = tokenizer(sentlist, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(ex, dtype=torch.long)} for ex in self.examples]

    def __len__(self) :
        return len(self.examples)

    def __getitem__(self, i) :
        return self.examples[i]


def prepare_dataset_for_pretraining(tokenizer, train_path, dev_path):
    
    train_dataset = RePreDataset(
        tokenizer=tokenizer,
        data=[train_path, dev_path],
        block_size=128,
    )

    # set mlm task
    # DataCollatorForSOP로 변경시 SOP 사용 가능 (DataCollatorForLanguageModeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15 # 0.3
    )

    eval_dataset = RePreDataset(
        tokenizer=tokenizer,
        data=[dev_path],
        block_size=128,
    )

    return train_dataset, data_collator, eval_dataset


def set_trainer_for_pretraining(
        model,
        data_collator,
        dataset,
        eval_dataset,
        epoch = 10,
        batch_size = 16,
        accumalation_step = 1,):

     # set training args
    training_args = TrainingArguments(
        report_to = 'tensorboard',
        output_dir='./',
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumalation_step,
        evaluation_strategy = 'steps',
        eval_steps=150,
        save_steps=150,
        save_total_limit=1,
        fp16=True,
        load_best_model_at_end=True,
        seed=42,
    )

    # set Trainer class for pre-training
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )

    return trainer


def parse_arguments() :

    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--model', type=str, default="klue/roberta-large")
    
    parser.add_argument('--train_dir', type=str, default="../dataset/train/train.csv")
    parser.add_argument('--dev_dir', type=str, default="../dataset/dev/dev.csv")
    parser.add_argument('--output_dir', type=str, default='./results_pre')
    
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=486)
    parser.add_argument('--accumalation_step', type=int, default=1)
    
    
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=int, default=0.01)

    args = parser.parse_args()

    return args




def train():

    # argparse
    args = parse_arguments()

    # 시드 설정
    set_seed(args.seed)

    # device 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    
    ### ========================================================================
    ### Tokenizer와 Model 준비
    ### ========================================================================
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.parameters
    model.to(device)
    

    ### ========================================================================
    ### 데이터셋 준비
    ### ========================================================================

    # 리스트를 넘겨준다.
    train_dataset, data_collator, eval_dataset = prepare_dataset_for_pretraining(tokenizer, args.train_dir, args.dev_dir)


    ### ========================================================================
    ### 트레이너
    ### ========================================================================
    
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        overwrite_output_dir = True,
        num_train_epochs = args.epoch,

        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        
        gradient_accumulation_steps = args.accumalation_step,

        learning_rate = args.lr,
        warmup_steps = args.warmup_steps,
        weight_decay = args.weight_decay,

        evaluation_strategy = 'steps',
        eval_steps = args.eval_steps,
        save_steps = args.eval_steps,
        save_total_limit = args.save_total_limit,
        
        fp16 = True,
        load_best_model_at_end = True,
        seed = args.seed,
        # report_to = 'wandb',
    )

    training_args.set_optimizer(name="adamw_torch", beta1=0.8, beta2=0.999, weight_decay = 0.01, learning_rate = 1e-04)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained("./pretrained")


if __name__ == '__main__':
    train()