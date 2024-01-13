from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def parse_arguments() :

  parser = argparse.ArgumentParser(description='Argparse Tutorial')

  parser.add_argument('--model', type=str, default="klue/roberta-large")
  parser.add_argument('--model_dir', type=str, default="./best_model")
  parser.add_argument('--test_dir', type=str, default="../dataset/test/test_data.csv")
  parser.add_argument('--output', type=str, default='./prediction/submission_roberta_large_imsi.csv')
  parser.add_argument('--seed', type=int, default=486)

  args = parser.parse_args()

  return args


def main(args):

  # 시드 설정
  set_seed(args.seed)

  # device 설정
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  ### ========================================================================
  ### Tokenizer와 Model
  ### ========================================================================
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)


  ### ========================================================================
  ### Test dataset
  ### ========================================================================
  ## load test datset
  test_id, test_dataset, test_label = load_test_dataset(args.test_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ### ========================================================================
  ### Prediction
  ### ========================================================================
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.


  ### ========================================================================
  ### Export
  ### ========================================================================
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
  output.to_csv(args.output, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.  
  print('---- Finish! ----')


if __name__ == '__main__':
  
  # argparse
  args = parse_arguments() 
  main(args)
  
