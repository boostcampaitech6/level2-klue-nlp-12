import random
import argparse
from typing import Any

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
import sklearn

import torch
import transformers
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

import wandb

##############################################
# 1. 현재 코드는 테스트용으로 작성된 코드입니다. #
# 2. CUSTOMIZE 부분을 수정하여 사용하시면 됩니다. #
# 3. 추후 체크포인트 콜백이 추가될 예정입니다. #

# last update: 2024.01.10 #
##############################################

####################################################################################
# 기본적인 시드 설정, metric 계산 함수 정의 부분

def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

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

def compute_metrics(pred, labels):
    """ validation을 위한 metrics function """
    labels = labels.cpu().numpy()
    preds = pred.argmax(-1).cpu().numpy()
    probs = pred.cpu().numpy().astype(np.float32)

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    # auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        # 'auprc' : auprc,
        'accuracy': acc,
    }

#######################################################################################



#######################################################################################
# Dataset, Dataloader, Model 정의

class Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
class Dataloader(pl.LightningModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

    def tokenizing(self, dataframe):
        concat_entity = []
        for e01, e02 in zip(dataframe['subject_entity'], dataframe['object_entity']):
            temp = ''
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)
        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(dataframe['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            )
        
        return tokenized_sentences
    
    def preprocessing(self, dataset):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity = []
        object_entity = []
        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        
        return out_dataset
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            train = pd.read_csv(self.train_path)
            val = pd.read_csv(self.dev_path)

            train = self.preprocessing(train)
            val = self.preprocessing(val)

            train_label = label_to_num(train['label'].values)
            val_label = label_to_num(val['label'].values)

            self.train_dataset = Dataset(self.tokenizing(train), train_label)
            self.val_dataset = Dataset(self.tokenizing(val), val_label)

        else:
            test = pd.read_csv(self.test_path)
            test = self.preprocessing(test)

            test_label = label_to_num(test['label'].values)

            self.test_dataset = Dataset(self.tokenizing(test), test_label)

            predict = pd.read_csv(self.predict_path)
            predict = self.preprocessing(predict)

            self.predict_dataset = self.tokenizing(predict)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    

class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=30)
        
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch['input_ids'], batch['labels']
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['input_ids'], batch['labels']
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss)

        log_metrics = compute_metrics(logits, y)
        self.log("val micro f1 score", log_metrics['micro f1 score'])
        # self.log("val auprc", log_metrics['auprc'])
        self.log("val accuracy", log_metrics['accuracy'])

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['input_ids'], batch['labels']
        logits = self(x)

        log_metrics = compute_metrics(logits, y)
        self.log("test micro f1 score", log_metrics['micro f1 score'])
        # self.log("test auprc", log_metrics['auprc'])
        self.log("test accuracy", log_metrics['accuracy'])

    def predict_step(self, batch, batch_idx):
        x = batch['input_ids']
        logits = self(x)

        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="klue/roberta-large")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--train_path", type=str, default="../dataset/train/train.csv")
    parser.add_argument("--dev_path", type=str, default="../dataset/train/dev.csv")
    parser.add_argument("--test_path", type=str, default="../dataset/train/dev.csv")
    parser.add_argument("--predict_path", type=str, default="../dataset/test/test_data.csv")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--logger_name", type=str, default="roberta-large")

    args = parser.parse_args()
    
    set_seed(42)

    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path, args.test_path, args.predict_path)

    model = Model(args.model_name, args.learning_rate)

    wandb_logger = WandbLogger(name=args.logger_name, project='level_2_RE', save_dir='./logs') #CUSTOMIZE wandb logger name and save directory

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.epoch,
        log_every_n_steps=1,
        callbacks=[early_stop_callback],
        logger=wandb_logger
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    model_filename = f"../model/model_{args.logger_name}.pt" #CUSTOMIZE model save directory
    torch.save(model, model_filename)