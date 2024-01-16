import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, set_seed
from load_data import *
import argparse

#Optimizer library
from timm.optim import AdamP 

#Schedular library
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

class CustomTrainingArguments(TrainingArguments):
  def create_optimizer_and_scheduler(self, num_training_steps: int):
    optimizer = torch.optim.AdamP(self.model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, num_training_steps=num_training_steps)
    return optimizer, scheduler


def seed_everything(seed = 42) :
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def parse_arguments() :

  parser = argparse.ArgumentParser(description='Argparse Tutorial')

  parser.add_argument('--model', type=str, default="klue/roberta-large")
  
  parser.add_argument('--train_dir', type=str, default="../dataset/train/train.csv")
  parser.add_argument('--dev_dir', type=str, default="../dataset/dev/dev.csv")
  parser.add_argument('--output_dir', type=str, default='./results')
  
  parser.add_argument('--epoch', type=int, default=5)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--lr', type=float, default=5e-5)
  parser.add_argument('--seed', type=int, default=486)
  
  parser.add_argument('--save_total_limit', type=int, default=10)
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
  model_config = AutoConfig.from_pretrained(args.model)
  model_config.num_labels = 30
  model = AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
  
  print(model.config)
  
  model.parameters
  model.to(device)

  ### ========================================================================
  ### 데이터셋 준비
  ### ========================================================================
  # load dataset
  train_dataset = load_data(args.train_dir)
  dev_dataset = load_data(args.dev_dir) # validation용 데이터는 따로 만드셔야 합니다.

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  

  ### ========================================================================
  ### 트레이너
  ### ========================================================================
  
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
  '''
  training_args = TrainingArguments(
    output_dir=args.output_dir,                   # output directory
    save_total_limit=args.save_total_limit,       # number of total save model.
    save_steps=args.eval_steps,                   # model saving step.
    num_train_epochs=args.epoch,                  # total number of training epochs
    # optimizer 종류는 args에서도 변경 할 수 있지만 default로 AdamW로 되어 있고 AdamP는 포함이 안되어 있는 것 같아서 따로 함수로 만들었어요.
    #learning_rate=args.lr,                        # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    #training_args에 있는 스케쥴러의 경우 linear로 고정 되어 있는것 같아서 제가 새로운 함수 추가 했어요
    #warmup_steps=args.warmup_steps,               # number of warmup steps for learning rate scheduler
    #weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir='./logs',                         # directory for storing logs
    logging_steps=100,                            # log saving step.
    evaluation_strategy='steps',                  # evaluation strategy to adopt during training
                                                  # `no`: No evaluation during training.
                                                  # `steps`: Evaluate every `eval_steps`.
                                                  # `epoch`: Evaluate every end of epoch.
    eval_steps = args.eval_steps,                 # evaluation step.
    load_best_model_at_end = True 
  )
  '''
  ### ========================================================================
  ### optimizer and schedular 
  ### ========================================================================
  
  #optimizer의 경우 논문을 읽어보고 AdamP가 제일 좋은 것 같아서 일단 AdamP로 설정했지만 timm (https://timm.fast.ai/Optimizers)에 있는 다른 다양한 optimizer들을 적용할 수 있어요.
  #optimizer = AdamP(model.parameters(), lr=0.001)

  #더 다양한 schedular들도 추가 할 수 있습니다.
  #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
  #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='cos')
  


  training_args = CustomTrainingArguments(
    output_dir=args.output_dir,
    save_total_limit=args.save_total_limit,
    save_steps=args.eval_steps,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=args.eval_steps,
    load_best_model_at_end=True
)

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  
  # train model
  trainer.train()
  model.save_pretrained('./best_model')

if __name__ == '__main__':
  train()
