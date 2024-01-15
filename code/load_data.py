import pickle as pickle
import os
import pandas as pd
import torch
import re
import json


class RE_Dataset(torch.utils.data.Dataset):
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

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  # 1. json 형태로 dict를 변환하기 위해 '를 "로 변경
  # 2. json 형태 변환 이전에 word에 해당하는 부분의 "를 모두 제거하고 단어의 시작과 끝에만 "를 추가
  # 3. json 형태로 load 이후 DataFrame으로 변환

  # 1
  sub = [s.replace("'", '"') for s in dataset['subject_entity']]
  obj = [o.replace("'", '"') for o in dataset['object_entity']]

  # 2
  for idx, sentence in enumerate(sub):
      search = re.search(r'\s".+",', sentence)
      sub[idx] = sentence[:search.span()[0]+1] + '"' + sentence[search.span()[0]+1:search.span()[1]-1].replace('"', '') + '"' + sentence[search.span()[1]-1:]

  for idx, sentence in enumerate(obj):
      search = re.search(r'\s".+",', sentence)
      obj[idx] = sentence[:search.span()[0]+1] + '"' + sentence[search.span()[0]+1:search.span()[1]-1].replace('"', '') + '"' + sentence[search.span()[1]-1:]
  
  # 3
  sub = pd.DataFrame([json.loads(s) for s in sub])
  obj = pd.DataFrame([json.loads(o) for o in obj])

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':sub['word'],'object_entity':obj['word'],'label':dataset['label'],
                                'sub_start_idx':sub['start_idx'], 'sub_end_idx':sub['end_idx'], 'sub_type':sub['type'], 'obj_start_idx':obj['start_idx'],'obj_end_idx':obj['end_idx'], 'obj_type':obj['type']})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  sentences = []

  # tokenizer.add_special_tokens({'additional_special_tokens': ['<SPC>', '</SPC>']})
  # tokenizer.add_special_tokens({'additional_special_tokens': ['[ORG]', '[PER]', '[DAT]', '[LOC]', '[POH]', '[NOH]']})
  tokenizer.add_special_tokens({'additional_special_tokens': ['[ORG]', '[PER]', '[DAT]', '[LOC]', '[POH]', '[NOH]', '[/ORG]', '[/PER]', '[/DAT]', '[/LOC]', '[/POH]', '[/NOH]']})


  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)

  for s_st, s_end, o_st, o_end, s, s_t, o_t in zip(dataset['sub_start_idx'], dataset['sub_end_idx'], dataset['obj_start_idx'], dataset['obj_end_idx'], dataset['sentence'], dataset['sub_type'], dataset['obj_type']):
      temp = ''

      # 스페셜 토큰 추가 #1
      # s_t = '[' + s_t + ']'
      # o_t = '[' + o_t + ']'
      # if s_st < o_st:
      #     temp = s[:s_st] + s_t + s[s_st:o_st] + o_t + s[o_st:]
      # else:
      #     temp = s[:o_st] + o_t + s[o_st:s_st] + s_t + s[s_st:]

      # 스페셜 토큰 추가 #2
      s_t_st = '[' + s_t + ']'
      o_t_st = '[' + o_t + ']'
      
      s_t_ed = '[/' + s_t + ']'
      o_t_ed = '[/' + o_t + ']'
      if s_st < o_st:
          temp = s[:s_st] + s_t_st + s[s_st:s_end+1] + s_t_ed + s[s_end+1:o_st] + o_t_st + s[o_st:o_end+1] + o_t_ed + s[o_end+1:]
      else:
          temp = s[:o_st] + o_t_st + s[o_st:o_end+1] + o_t_ed + s[o_end+1:s_st] + s_t_st + s[s_st:s_end+1] + s_t_ed + s[s_end+1:]

      # 스페셜 토큰 추가 (양쪽에)
      # if s_st < o_st:
      #     temp = s[:s_st] + '<SPC>' + s[s_st:s_end+1] + '</SPC>' + s[s_end+1:o_st] + '<SPC>' + s[o_st:o_end+1] + '</SPC>' + s[o_end+1:]
      # else:
      #     temp = s[:o_st] + '<SPC>' + s[o_st:o_end+1] + '</SPC>' + s[o_end+1:s_st] + '<SPC>' + s[s_st:s_end+1] + '</SPC>' + s[s_end+1:]
      sentences.append(temp)

  tokenized_sentences = tokenizer(
      concat_entity,
      sentences,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  
  return tokenized_sentences
