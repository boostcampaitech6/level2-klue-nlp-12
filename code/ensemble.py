import argparse
import pandas as pd
import numpy as np
from operator import add

import torch
import torch.nn.functional as F

def list_of_strings(arg):
    return arg.split(',')

def list_of_weights(arg):
    return list(map(float, arg.split(',')))

def call_csv_files(paths):
    df_list = []
    for path in paths:
        df = pd.read_csv(path)
        df_list.append(df)
    return df_list


# ==========================================================================
#                                   mean
# ==========================================================================
def ensemble_mean(paths):

    idx2label = {0: 'no_relation', 1: 'org:top_members/employees', 2: 'org:members', 3: 'org:product', 4: 'per:title',
                 5: 'org:alternate_names', 6: 'per:employee_of', 7: 'org:place_of_headquarters', 8: 'per:product', 9: 'org:number_of_employees/members', 
                 10: 'per:children', 11: 'per:place_of_residence', 12: 'per:alternate_names', 13: 'per:other_family', 14: 'per:colleagues', 
                 15: 'per:origin', 16: 'per:siblings', 17: 'per:spouse', 18: 'org:founded', 19: 'org:political/religious_affiliation', 
                 20: 'org:member_of', 21: 'per:parents', 22: 'org:dissolved', 23: 'per:schools_attended', 24: 'per:date_of_death',
                 25: 'per:date_of_birth', 26: 'per:place_of_birth', 27: 'per:place_of_death', 28: 'org:founded_by', 29: 'per:religion'}

    # 데이터프레임을 불러옵니다 -> list(pd.DataFrame)
    dataframes = call_csv_files(paths)
    num_of_df = len(dataframes)
    
    # 데이터프레임의 probs를 numpy.ndarray(float)로 변환합니다
    for idx, df in enumerate(dataframes):
        dataframes[idx]['probs']  = df['probs'].apply(lambda x: np.array([float(i) for i in x.strip('][').split(', ')]))

    # 마지막 데이터프레임을 기본 데이터프레임으로 초기화합니다 -> && 마지막에 반환할 데이터프레임
    default_df = dataframes[-1]

    # ensembling
    for df in dataframes[:-1]:
        default_df['probs'] = default_df['probs'] + df['probs']
    
    # 각 probs를 num_of_df로 나누어 평균으로 만듭니다
    default_df['probs'] = default_df['probs']/num_of_df
    
    # 새로운 probs에 맞게 pred_label 변경합니다
    for idx, row in default_df.iterrows():
        default_df.loc[idx, 'pred_label'] = idx2label[int(row['probs'].argmax())]

    # type(probs) np.adarray -> list(float)
    default_df['probs'] = default_df['probs'].map(np.ndarray.tolist)

    # 이름만 바꾸기
    ensembled_df = default_df

    return ensembled_df

# ==========================================================================
#                             weighted_sum
# ==========================================================================
def ensemble_weighted_sum(paths, weights):
    
    idx2label = {0: 'no_relation', 1: 'org:top_members/employees', 2: 'org:members', 3: 'org:product', 4: 'per:title',
                 5: 'org:alternate_names', 6: 'per:employee_of', 7: 'org:place_of_headquarters', 8: 'per:product', 9: 'org:number_of_employees/members', 
                 10: 'per:children', 11: 'per:place_of_residence', 12: 'per:alternate_names', 13: 'per:other_family', 14: 'per:colleagues', 
                 15: 'per:origin', 16: 'per:siblings', 17: 'per:spouse', 18: 'org:founded', 19: 'org:political/religious_affiliation', 
                 20: 'org:member_of', 21: 'per:parents', 22: 'org:dissolved', 23: 'per:schools_attended', 24: 'per:date_of_death',
                 25: 'per:date_of_birth', 26: 'per:place_of_birth', 27: 'per:place_of_death', 28: 'org:founded_by', 29: 'per:religion'}

    # 데이터프레임을 불러옵니다 -> list(pd.DataFrame)
    dataframes = call_csv_files(paths)
    
    # 가중치 리스트를 변수에 저장합니다
    weight_list = weights
    
    # 가중치가 정수로 주어졌을 경우 합이 1이 되도록 나누어줍니다
    if sum(weight_list) > 1:
        weight_list = [val/sum(weight_list) for val in weight_list]

    # 데이터프레임의 probs를 numpy.ndarray(float)로 변환합니다
    for idx, df in enumerate(dataframes):
        dataframes[idx]['probs']  = df['probs'].apply(lambda x: np.array([float(i) for i in x.strip('][').split(', ')]))

    # 마지막 데이터프레임을 기본 데이터프레임으로 초기화합니다 -> && 마지막에 반환할 데이터프레임
    default_df = dataframes[-1]
    default_df['probs'] = default_df['probs'] * weight_list[-1]
    
    # 각 데이터프레임에 가중치를 곱합니다
    for idx, df in enumerate(dataframes[:-1]):
        default_df['probs'] = default_df['probs'] + df['probs'] * weight_list[idx]
    
    # 새로운 probs에 맞게 pred_label 변경합니다
    for idx, row in default_df.iterrows():
        default_df.loc[idx, 'pred_label'] = idx2label[int(row['probs'].argmax())]

    # type(probs) np.adarray -> list(float)
    default_df['probs'] = default_df['probs'].map(np.ndarray.tolist)

    # 이름만 바꾸
    ensembled_df = default_df

    return ensembled_df




def ensemble_weight(pathlist, weights, softmax=False) :

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

    # df로 변환해주고
    df_list = [pd.read_csv(path) for path in pathlist]

    # str -> float으로 일괄 변경
    probs_list = [df.probs.apply(lambda x: [float(e) for e in x.strip("][").split(", ")]) for df in df_list]

    # 리턴할 df 생성
    ensemble_df = df_list[0].copy()
    softed_probs = []
    pred_list = []

    # score를 가중치로 변경
    if softmax :
        weights = F.softmax(torch.Tensor(weights), dim=0).tolist()

    # row마다
    for idx in range(len(df_list[0])) :

        # 확률을 모아서
        prob_list = [np.array(prob[idx]) for prob in probs_list]

        # 가중합을 누적하고
        final_probs = np.zeros(30)
        for i, prob in enumerate(prob_list) :
            final_probs += prob*weights[i]

        # 분모로 나눠서
        final_probs /= sum(weights)

        # probs에 추가
        softed_probs.append(str(final_probs.tolist()))
        pred_list.append(label_list[torch.argmax(torch.Tensor(final_probs))])

    # 해당 열에 할당
    ensemble_df.probs = softed_probs
    ensemble_df.pred_label = pred_list

    return ensemble_df


def parse_arguments() :
    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--str_list', type=list_of_strings)
    parser.add_argument('--weight_list', type=list_of_weights) #      정수 입력값도 받습니다
    parser.add_argument('--output_path', type=str, default='./prediction/output.csv')
    parser.add_argument('--technique', type=str, default='mean') #    options:    mean || weighted_sum || softmax

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    # argparse
    args = parse_arguments()

    if args.technique == 'mean':
        df = ensemble_mean(args.str_list)
    elif args.technique == 'weighted_sum':
        df = ensemble_weighted_sum(args.str_list, args.weight_list)
    elif args.technique == 'softmax':
        df = ensemble_weight(args.str_list, args.weight_list, softmax=True)

    df.to_csv(f'{args.output_path}', index=False)