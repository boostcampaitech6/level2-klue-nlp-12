import pickle as pickle
import os
import pandas as pd
import torch
import re
import json
from ast import literal_eval

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

def preprocessing_dataset(dataset, options='default'):
  
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

  if options == "legacy" :
     
    subject_entity = []
    object_entity = []
    subject = []
    object = []
    
    for s,o in zip(dataset['subject_entity'], dataset['object_entity']):
        i = s[1:-1].split(',')[0].split(':')[1]
        j = o[1:-1].split(',')[0].split(':')[1]
        subject_entity.append(i)
        object_entity.append(j)
        subject.append(literal_eval(s))
        object.append(literal_eval(o))
    
    sub = pd.DataFrame(subject)
    obj = pd.DataFrame(object)


  else :

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
                            'sub_start_idx':sub['start_idx'], 'sub_end_idx':sub['end_idx'], 'obj_start_idx':obj['start_idx'], 'obj_end_idx':obj['end_idx'], 'sub_type':sub['type'], 'obj_type':obj['type']})
  return out_dataset

def load_data(dataset_dir, options='default'):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset, options)
  
  return dataset

def tokenized_dataset(dataset, tokenizer, options='default'):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  sentences = []

  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)

  # default: sub, obj, sentence를 [SEP] 토큰으로만 구분
  if options == 'default':
      sentences = list(dataset['sentence'])

  # special: sentence 내부의 sub, obj 시작, 끝 부분을 @, # 토큰으로 구분
  elif options == 'special':
      for s_st, s_end, o_st, o_end, s, s_t, o_t in zip(dataset['sub_start_idx'], dataset['sub_end_idx'], dataset['obj_start_idx'], dataset['obj_end_idx'], dataset['sentence'], dataset['sub_type'], dataset['obj_type']):
          temp = ''

          if s_st < o_st:
              temp = s[:s_st] + '@' + s[s_st:s_end+1] + '#' + s[s_end+1:o_st] + '@' + s[o_st:o_end+1] + '#' + s[o_end+1:]
          else:
              temp = s[:o_st] + '@' + s[o_st:o_end+1] + '#' + s[o_end+1:s_st] + '@' + s[s_st:s_end+1] + '#' + s[s_end+1:]
          sentences.append(temp)

  # type: sentence 내부의 sub, obj 시작, 끝 부분을 entity type에 따라 토큰으로 구분
  elif options == 'type':
      type_list = {'ORG':'@', 'PER':'#', 'DAT':'$', 'LOC':'&', 'POH':'%', 'NOH':'^'}
      for s_st, s_end, o_st, o_end, s, s_t, o_t in zip(dataset['sub_start_idx'], dataset['sub_end_idx'], dataset['obj_start_idx'], dataset['obj_end_idx'], dataset['sentence'], dataset['sub_type'], dataset['obj_type']):
          temp = ''

          if s_st < o_st:
              temp = s[:s_st] + type_list[s_t] + s[s_st:s_end+1] + type_list[s_t] + s[s_end+1:o_st] + type_list[o_t] + s[o_st:o_end+1] + type_list[o_t] + s[o_end+1:]
          else:
              temp = s[:o_st] + type_list[o_t] + s[o_st:o_end+1] + type_list[o_t] + s[o_end+1:s_st] + type_list[s_t] + s[s_st:s_end+1] + type_list[s_t] + s[s_end+1:]
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
