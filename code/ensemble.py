import argparse
import pandas as pd
import numpy as np
from operator import add

def list_of_strings(arg):
    return arg.split(',')

def parse_arguments() :
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--str-list', type=list_of_strings)

    args = parser.parse_args()
    return args

def call_csv_files(paths):
    df_list = []
    for path in paths:
        df = pd.read_csv(path)
        df_list.append(df)
    return df_list

def main():

    # argparse
    args = parse_arguments()

    # 데이터프레임을 불러옵니다 -> list(pd.DataFrame)
    dataframes = call_csv_files(args.str_list)
    num_of_df = len(dataframes)
    
    # 데이터프레임의 probs를 numpy.ndarray(float)로 변환
    for idx, df in enumerate(dataframes):
        dataframes[idx]['probs']  = df['probs'].apply(lambda x: np.array([float(i) for i in x.strip('][').split(', ')]))

    # 마지막 데이터프레임을 기본 데이터프레임으로 초기화 -> && 마지막에 반환할 데이터프레임
    default_df = dataframes[-1]

    # ensembling
    for df in dataframes[:-1]:
        default_df['probs'] = default_df['probs'] + df['probs']
    
    # 각 probs를 num_of_df로 나누어 평균으로 만듦
    default_df['probs'] = default_df['probs']/num_of_df
    
    # 새로운 probs에 맞게 pred_label 변경
    for idx, row in default_df.iterrows():
        default_df.loc[idx, 'pred_label'] = idx2label[int(row['probs'].argmax())]
    
    
    # 이름만 바꾸기
    ensembled_df = default_df

    return ensembled_df

if __name__ == '__main__':
    idx2label = {0: 'no_relation', 1: 'org:top_members/employees', 2: 'org:members', 3: 'org:product', 4: 'per:title',
                 5: 'org:alternate_names', 6: 'per:employee_of', 7: 'org:place_of_headquarters', 8: 'per:product', 9: 'org:number_of_employees/members', 
                 10: 'per:children', 11: 'per:place_of_residence', 12: 'per:alternate_names', 13: 'per:other_family', 14: 'per:colleagues', 
                 15: 'per:origin', 16: 'per:siblings', 17: 'per:spouse', 18: 'org:founded', 19: 'org:political/religious_affiliation', 
                 20: 'org:member_of', 21: 'per:parents', 22: 'org:dissolved', 23: 'per:schools_attended', 24: 'per:date_of_death',
                 25: 'per:date_of_birth', 26: 'per:place_of_birth', 27: 'per:place_of_death', 28: 'org:founded_by', 29: 'per:religion'}
    main()