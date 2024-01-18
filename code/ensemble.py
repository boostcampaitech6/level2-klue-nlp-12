import argparse
import pandas as pd
import numpy as np
from operator import add

def list_of_strings(arg):
    return arg.split(',')

def list_of_weights(arg):
    return list(map(float, arg.split(',')))

def parse_arguments() :
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--str-list', type=list_of_strings)
    parser.add_argument('--weight-list', type=list_of_weights)
    parser.add_argument('--output-path', type=str, default='./output.csv')
    parser.add_argument('--technique', type=str, default='mean') #    options:    mean || weighted_sum || 

    args = parser.parse_args()
    return args

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
def ensemble_weighted_sum(paths):
    
    # 데이터프레임을 불러옵니다 -> list(pd.DataFrame)
    dataframes = call_csv_files(paths)
    num_of_df = len(dataframes)
    
    # 가중치 리스트를 변수에 저장합니다
    weight_list = args.weight_list

    # 데이터프레임의 probs를 numpy.ndarray(float)로 변환합니다
    for idx, df in enumerate(dataframes):
        dataframes[idx]['probs']  = df['probs'].apply(lambda x: np.array([float(i) for i in x.strip('][').split(', ')]))

    # 마지막 데이터프레임을 기본 데이터프레임으로 초기화합니다 -> && 마지막에 반환할 데이터프레임
    default_df = dataframes[-1]
    default_df['probs'] = default_df['probs'] * weight_list[-1]
    
    # 각 데이터프레임에 가중치를 곱합니다
    

    return

if __name__ == '__main__':
    
    # argparse
    args = parse_arguments()

    idx2label = {0: 'no_relation', 1: 'org:top_members/employees', 2: 'org:members', 3: 'org:product', 4: 'per:title',
                 5: 'org:alternate_names', 6: 'per:employee_of', 7: 'org:place_of_headquarters', 8: 'per:product', 9: 'org:number_of_employees/members', 
                 10: 'per:children', 11: 'per:place_of_residence', 12: 'per:alternate_names', 13: 'per:other_family', 14: 'per:colleagues', 
                 15: 'per:origin', 16: 'per:siblings', 17: 'per:spouse', 18: 'org:founded', 19: 'org:political/religious_affiliation', 
                 20: 'org:member_of', 21: 'per:parents', 22: 'org:dissolved', 23: 'per:schools_attended', 24: 'per:date_of_death',
                 25: 'per:date_of_birth', 26: 'per:place_of_birth', 27: 'per:place_of_death', 28: 'org:founded_by', 29: 'per:religion'}
    
    if args.technique == 'mean':
        df = ensemble_mean(args.str_list)
    elif args.technique == 'weighted_sum':
        df = ensemble_weighted_sum(args.str_list)

    df.to_csv(f'{args.output_path}', index=False)