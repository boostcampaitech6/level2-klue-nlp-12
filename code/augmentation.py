import os
import re
import pickle
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from enum import Enum
from typing import Union

# from soynlp.normalizer import emoticon_normalize, repeat_normalize
# from hanspell import spell_checker
# from konlpy.tag import Okt, Kkma

from bs4 import BeautifulSoup
from selenium import webdriver
import random
import requests
from kiwipiepy import Kiwi
import time
from quickspacer import Spacer
import json



import warnings
warnings.filterwarnings('ignore')




class Adverb_Augmentation():

    def __init__(self, data_path, save_path) :
        self.data_path = data_path
        self.save_path = save_path
        self.df = pd.read_csv(data_path)
        self.kiwi = Kiwi()
        self.spacing = Spacer().space

    # 경로 생성 메서드
    def create_folder(self, save_path:str)->None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def _adverb_detector(self, sentence):

        # POS info
        pos_list = [(x[0], x[1]) for x in self.kiwi.tokenize(sentence)] # (token, pos)

        adverb_list = []
        for pos in pos_list:
            if pos[1] == "MAG" and len(pos[0]) > 1: # 1음절 부사는 제외함.
                adverb_list.append(pos[0])
        return adverb_list


    def _get_gloss(self, word):
        res = requests.get("https://dic.daum.net/search.do?q=" + word, timeout=5)
        time.sleep(random.uniform(0.5,2.5))
        soup = BeautifulSoup(res.content, "html.parser")
        try:
            # 첫 번째 뜻풀이.
            meaning = soup.find('span', class_='txt_search')
        except AttributeError:
            return word
        if not meaning:
            return word

        # parsing 결과에서 한글만 추출
        meaning = re.findall('[가-힣]+', str(meaning))
        meaning = ' '.join(meaning)

        # 띄어쓰기 오류 교정 (위 에 -> 위에)
        # meaning = spell_checker.check(meaning).as_dict()['checked'].strip()
        meaning = self.spacing([meaning.replace(" ", "")])
        return meaning[0].strip()

    def adverb_gloss_replacement(self, sentence):
        adverb_list = self._adverb_detector(sentence)
        if adverb_list:
            # 부사들 중에서 1개만 랜덤으로 선택합니다.
            adverb = random.choice(adverb_list)
            try:
                gloss = self._get_gloss(adverb)
                sentence = sentence.replace(adverb, gloss)
            except:
                print('except: ', sentence)
                pass
        return sentence

    def concat(self, df_list) -> pd.DataFrame :
        """주어진 DataFrame을 결합하고 중복 처리, 인덱스 재정렬을 수행하는 메서드

        Args:
            df_list (list[pd.DataFrame]): DataFrame으로 이루어진 List

        Returns:
            pd.DataFrame: 결합된 DataFrame
        """

        df_concat = pd.concat(df_list)
        df_concat.drop_duplicates(subset=['sentence'], inplace=True)
        df_concat.reset_index(drop=True, inplace=True)
        return df_concat 

    def augmentation(self, file_name='train_augment.csv', save=True) :

        df_orig = self.df.copy()
        features = df_orig["label"].unique().tolist()
        features.remove("no_relation")

        augdf_list = []

        # no_relation을 제외한 feature마다 돌아가면서
        for feat in features :
            print("Now processing feature {}...".format(feat))
            
            df_select = df_orig[df_orig["label"] == feat]
            df_aug = self.get_augmented_df(df_select)
            augdf_list.append(df_aug)
            
        df_concat = self.concat(augdf_list)
        df_augment = self.concat(df_orig, df_concat)

        if save :
            df_augment.to_csv(os.path.join(self.save_path, file_name), index=False)
        else :
            return df_augment

    # df를 받아, 증강 df 리턴
    def get_augmented_df(self, df) :
        
        # df 복사
        df_aug = df.copy()
        
        # 부사 교체한 열을 만들어, 변한 것들만 남김
        df_aug['new_sentence'] = df_aug['sentence'].apply(self.adverb_gloss_replacement)
        df_aug = df_aug[(df_aug['sentence'] != df_aug['new_sentence'])]

        # sub, obj 재추적
        df_aug['new_subject_entity'] = df_aug.apply(self.find_new_subject_word, axis=1)
        df_aug['new_object_entity'] = df_aug.apply(self.find_new_object_word, axis=1)

        # col 정리
        df_aug = df_aug[['id', 'new_sentence', 'new_subject_entity', 'new_object_entity', 'label', "source"]]
        df_aug = df_aug.rename(columns={'new_sentence': 'sentence', 'new_subject_entity': 'subject_entity', 'new_object_entity': 'object_entity'})

        return df_aug
    

    def find_new_subject_word(self, row) :

        sub_word = row["subject_entity"][1:-1].split(',')[0].split(':')[1].replace("'", "").strip()
        start_idx = row["new_sentence"].find(sub_word)
        if start_idx == -1:
            return -1
        end_idx = start_idx + len(sub_word) - 1

        result = json.loads(row["subject_entity"].replace("'", "\""))
        result["start_idx"] = start_idx
        result["end_idx"] = end_idx

        return str(result)

    def find_new_object_word(self, row) :

        obj_word = row["object_entity"][1:-1].split(',')[0].split(':')[1].replace("'", "").strip()
        start_idx = row["new_sentence"].find(obj_word)
        if start_idx == -1:
            return -1
        end_idx = start_idx + len(obj_word) - 1

        result = json.loads(row["object_entity"].replace("'", "\""))
        result["start_idx"] = start_idx
        result["end_idx"] = end_idx

        return str(result)




class Bert_Augmentaion() :

    def __init__(self, data_path, save_path, wordnet_path):

        self.data_path = data_path
        self.save_path = save_path
        self.wordnet_path = wordnet_path
        self.df = pd.read_csv(data_path)
        self.kkma = Kkma()
        
        # 폴더 경로 생성
        self.create_folder(self.save_path)
    
    # 폴더 경로 생성 메서드
    def create_folder(self, save_path:str)->None:
        """지정된 경로에 폴더를 생성하는 메서드

        Args:
            save_path (str): 주어진 경로에 directory가 없을 경우 생성.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def save_data(self, df:pd.DataFrame, save_path:str, file_name:str)->None:
        """데이터 저장 메서드

        Args:
            df (pd.DataFrame): 저장할 데이터
            save_path (str): 저장할 경로
            file_name (str): 저장할 파일 명
        """
        df.to_csv(os.path.join(save_path, file_name), index=False)

    # train 데이터 전처리 및 증강 메서드
    def train_augmentationV2(self, file_name='train_augmentV2.csv', save=True) -> Union[pd.DataFrame, None]:
        """AugementV2 버전 데이터 생성
            - 원본 데이터 + 맞춤법 검사 데이터 + 동의어 교체 데이터
            - Swap sentence, Copied Sentence
            - 각 과정마다 데이터 불균형을 해소하기 위해 비율을 다르게 하여 증강
            
        Args:
            file_name (str, optional): 저장할 파일 이름. Defaults to 'train_augmentV2.csv'.
            save (bool, optional): 반환 or 저장 여부 선택. Defaults to True.

        Returns:
            Union[pd.DataFrame, None]: save=False일 경우 증강 데이터 반환
        """
        df_original = self.df.copy()
        df_spellcheck = self.train_preprocessing(save=False)
        df_sr = self.synonym_replacement(df = df_spellcheck, wordnet_path = self.wordnet_path, symmin=3.0, symmax=4.5, rng=0.5, ratio=2)
        df_concat = self.concat([df_original, df_spellcheck, df_sr])
        df_swap = self.swap_sentence(df_concat)
        df_concat_swap = self.concat([df_concat, df_swap])
        df_augment = self.copied_sentenceV2(df_concat_swap)

        if save:
            self.save_data(df_augment, self.save_path, file_name)
        else:
            return df_augment

    def train_augmentationV3(self, file_name='train_augmentV3.csv', save=True) -> Union[pd.DataFrame, None]:
        """AugmentationV3 버전 데이터 생성
            - AugmentationV2 데이터 + nnp masking 데이터 

        Args:
            file_name (str, optional): 저장할 파일 이름. Defaults to 'train_augmentV3.csv'.
            save (bool, optional): 반환 or 저장 여부 선택. Defaults to True.

        Returns:
            Union[pd.DataFrame, None]: save=False일 경우 증강 데이터 반환
        """        
        df_orignal = self.df.copy()
        if 'train_augmentV2.csv' in os.listdir('../data'):
            print('train_augmentV2.csv is already exist!! load....')
            df_augmentV2 = pd.read_csv('../data/train_augmentV2.csv')
        else:
            df_augmentV2 = self.train_augmentationV2(save=False)
        df_nnp_masking = self.nnp_masking(df_orignal)
        df_augment = self.concat([df_augmentV2, df_nnp_masking])


        if save:
            self.save_data(df_augment, self.save_path, file_name)
        else:
            return df_augment


    def train_preprocessing(self, file_name='train.csv', save=True):
        """TRAIN 데이터 전처리

        Args:
            file_name (str, optional): 저장할 파일 이름. Defaults to 'train.csv'.
            save (bool, optional): 반환 or 저장 여부 선택. Defaults to True.

        Returns:
            _type_: save=False일 경우 증강 데이터 반환
        """
        df_spellcheck = self.spelling_check(self.df)    
        if save:
            self.save_data(df_spellcheck, self.save_path, file_name)
        else:
            return df_spellcheck
    

    def val_preprocessing(self, file_name='dev.csv'):
        """DEV 데이터 전처리

        Args:
            file_name (str, optional): 저장할 파일 이름. Defaults to 'dev.csv'.
        """
        df_spellcheck = self.spelling_check(self.df)
        self.save_data(df_spellcheck, self.save_path, file_name)
    
    def test_preprocessing(self, file_name='test.csv'):
        """TEST 데이터 전처리

        Args:
            file_name (str, optional): 저장할 파일 이름. Defaults to 'test.csv'.
        """
        df_spellcheck = self.spelling_check(self.df)
        self.save_data(df_spellcheck, self.save_path, file_name)

    def spelling_check(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        데이터에 apply_hanspell 함수를 적용
        Args:
            df (pd.DataFrame): 맞춤법을 교정할 데이터

        Returns:
            data (pd.DataFrame): 맞춤법을 교정한 데이터
        """
        df_change = df.copy()
        tqdm.pandas()
        df_change["sentence_1"] = df_change["sentence_1"].progress_map(self.apply_hanspell)
        df_change["sentence_2"] = df_change["sentence_2"].progress_map(self.apply_hanspell)
        return df_change
    
    def apply_hanspell(self, text:str) -> str:
        """
        중복 감정 표현 및 표현 제거, 특수 문자 제거 후 hanspell 맞춤법 검사 적용
        Args:
            text (str): 교정할 문장

        Returns:
            spell_check_text (str): 교정한 문장 
        """
        text = emoticon_normalize(text, num_repeats=2)
        text = repeat_normalize(text, num_repeats=2)
        text = text.lower()
        text = re.sub("[^a-zr-ㅎ가-힣0-9 ]", "", text)
        text = text.strip()
        spell_check_text = spell_checker.check(text).checked
        return spell_check_text

    def synonym_replacement(self, df:pd.DataFrame, wordnet_path:str, rng:float=0.5, symmin:float=3.0, symmax:float=4.5, ratio:int=2) -> pd.DataFrame:
        """ 동의어 대체 및 조사 대체를 수행하는 메서드

        Args:
            df (pd.DataFrame): 데이터 증강을 진행할 DataFrame
            wordnet_path (str): 동의어 사전 경로(wordnet은 KAIST에서 만든 Korean WordNet(KWN) 을 사용함.)
            rng (float, optional): 데이터를 증강할 label 범위 (rng 이상). Defaults to 0.5.
            symmin (float, optional): 동의어 대체 비율을 조절하는 라벨의 범위의 최소값. Defaults to 3.0.
            symmax (float, optional): 동의어 대체 비율을 조절하는 라벨의 범위의 최대값. Defaults to 4.5.
            ratio (int, optional): 동의어에 대해서 대체할 비율. Defaults to 2.

        Returns:
            pd.DataFrame: 동의어가 대체된 DataFrame
        """
        df_change = df[df['label'] >=rng].reset_index(drop=True).copy()
        okt = Okt()
        with open(wordnet_path, "rb") as f:
            wordnet = pickle.load(f)
        
        n1, n2 = df_change['sentence_1'], df_change['sentence_2']
        sr_sentence = []

        for i in tqdm(range(len(n1)), desc="Synonym Replacement"):
            now_sent1 = n1[i]
            now_sent2 = n2[i]
            noun1 = okt.nouns(now_sent1)
            noun2 = okt.nouns(now_sent2)

            # 공통된 명사를 추출.
            same_nouns_set = set(noun1) & set(noun2)
            for same_noun in same_nouns_set:
                # 길이가 2 이상이고, wordnet에 있는지 확인.
                if len(same_nouns_set) > 1 and same_noun in wordnet and len(wordnet[same_noun]) >= 2:
                    sym_list = wordnet[same_noun][1:]
                    # label 별 비율을 맞춰주기 위해 3.0 <= label < 4.5 인 데이터 절반만 변환 
                    if symmin <= df_change['label'][i] < symmax:
                        sym_list = sym_list[:len(sym_list)//ratio+1]

                    for sym in sym_list:
                        s1 = okt.pos(now_sent1)
                        s2 = okt.pos(now_sent2)
                        sr_sentence.append(
                            [
                                df_change["id"][i],
                                df_change["source"][i],
                                self.make_sentence(s1, same_noun, sym),
                                self.make_sentence(s2, same_noun, sym),
                                df_change['label'][i],
                                df_change["binary-label"][i],
                            ]
                        ) 
        sr_sentence = pd.DataFrame(
            sr_sentence,
            columns=["id", "source", "sentence_1", "sentence_2", "label", "binary-label"],
        )

        return sr_sentence
    
    def make_sentence(self, sentence: list, compare: str, sym: str) -> str:
        """
        sentence_1, sentence_2에 모두 등장하는 명사를 교체하고 조사를 교정
            Args :
                sentence (list): 형태소 분석한 문장
                compare  (str): 문장에서 바꿀 명사
                sym      (str): 문장 삽입되는 동의어
            Returns :
                replace_sentence (str): 동의어로 교체한 문장
        """
        replace_sentence = []
        check = set(["이", "가", "을", "를", "과", "와"])
        for j in range(len(sentence)):
            # 문장에서 동의어를 추가한다.
            if sentence[j][0] == compare:
                replace_sentence.append(sym)
                # 뒷말이 조사면 조사를 확인하고 바꾼다.
                if (
                    j + 1 < len(sentence)
                    and sentence[j + 1][1] == "Josa"
                    and sentence[j + 1][0] in check
                ):
                    # 바뀐 명사 마지막 받침 확인 후 조사 변경
                    sentence[j + 1] = (
                        self.change_josa(replace_sentence[-1][0], sentence[j + 1][0]),
                        "Josa",
                    )
            else:
                replace_sentence.append(sentence[j][0])

        # hanspell로 띄어쓰기 교정.
        replace_sentence = "".join(replace_sentence)
        replace_sentence = spell_checker.check(replace_sentence).checked
        return replace_sentence
    
    def check_end(self, noun: str) -> bool:
        """
        한글의 유니코드가 28로 나누어 떨어지면 받침이 없음을 판단
            Args :
                noun (str): 받침 유무를 판단할 명사
            Returns :
                False (bool) : 받침이 없음
                True  (bool) :  받침이 있음
        """
        if (ord(noun[-1]) - ord("가")) % 28 == 0:
            return False
        else:
            return True

    def change_josa(self, noun: str, josa: str) -> str:
        """
        명사의 끝음절 받침 여부에 따라서 조사 교체
            Args :
                none (str): 끝음절의 받침 확인할 명사
                josa (str): 교정할 조사
            Returns :
                josa (str): 교정한 조사
        """
        if josa == "이" or josa == "가":
            return "이" if self.check_end(noun) else "가"
        elif josa == "은" or josa == "는":
            return "은" if self.check_end(noun) else "는"
        elif josa == "을" or josa == "를":
            return "을" if self.check_end(noun) else "를"
        elif josa == "과" or josa == "와":
            return "과" if self.check_end(noun) else "와"
        else:
            return josa
        
    def swap_sentence(self, df:pd.DataFrame) -> pd.DataFrame:
        """Sentence1과 Sentence2를 Swap(0.5 <= label <3.5, 4.5<= label < 5 인 라벨에 대해서만 sentence swap)

        Args:
            df (pd.DataFrame): 데이터를 증강할 DataFrame

        Returns:
            pd.DataFrame: Sentence1과 Sentence2가 Swap된 DataFrame
        """
        df_swapped = df.copy()
        df_swapped['sentence_1'] = df['sentence_2']
        df_swapped['sentence_2'] = df['sentence_1']
        df_swapped = df_swapped[((df_swapped['label'] >= 0.5) & (df_swapped['label'] <3.5)) | ((df_swapped['label'] >= 4.5) & (df_swapped['label'] <5)) ]
        return df_swapped
    
    def copied_sentenceV2(self, df:pd.DataFrame) -> pd.DataFrame:
        """(Setence1, Sentence2, label) --> (Sentence1 , Sentence2, 4.9)로 대체.
            많이 분포하는 라벨을 줄이고 적게 분포하는 라벨을 증강하기 위한 시도.
        Args:
            df (pd.DataFrame): 증강할 DataFrame

        Returns:
            pd.DataFrame: 증강된 DataFrame
        """
        df_change = df.reset_index(drop=True).copy()
        sample_0005 = df_change[(df_change['label'] >= 0.0) &(df_change['label'] < 0.5)][-1779:]
        sample_3035 = df_change[(df_change['label'] >= 3.0) &(df_change['label'] < 3.5)][-508:]
        sample_0445 = df_change[(df_change['label'] >= 1.0) &(df_change['label'] < 1.5)][-414:]

        for index in [sample_3035.index, sample_0005.index, sample_0445.index] :
            df_change.iloc[index, 3] = df_change.iloc[index, 2]
            df_change.iloc[index, 4] = 4.9
            df_change.iloc[index, 5] = 1
        
        df_change.drop_duplicates(subset=['sentence_1', 'sentence_2'], inplace=True)
        df_change.reset_index(drop=True, inplace=True)
        return df_change
    
    def mask_nnp(self, sentence:str) -> str:
        """고유명사, 외래어 masking 

        Args:
            sentence (str): masking될 문장

        Returns:
            str: masking된 문장
        """
        try:
            words = self.kkma.pos(sentence)
            for word, tag in words:
                # 태그  'NNG': '일반 명사', 'NNP': '고유 명사','SL' : '외래어'
                if tag in ['NNP', 'SL']:
                    sentence = sentence.replace(word, '<MASK>')
        except:
            return sentence
        return sentence

    def nnp_masking(self, df:pd.DataFrame) -> pd.DataFrame:
        """고유명사, 외래어를 마스킹하여 증강하는 메서드

        Args:
            df (pd.DataFrame): masking할 DataFrame

        Returns:
            pd.DataFrame: masking된 DataFrame
        """
        df_change = df.copy()
        df_change['sentence_1'] = df_change['sentence_1'].apply(self.mask_nnp)
        df_change['sentence_2'] = df_change['sentence_2'].apply(self.mask_nnp)
        return df_change



class KorEDAAugmentaion() :

    def __init__(self, data_path, save_path, wordnet_path):

        self.data_path = data_path
        self.save_path = save_path
        self.wordnet_path = wordnet_path
        self.df = pd.read_csv(data_path)
        



class Path(Enum):
    TRAIN = '../dataset/train/train.csv'
    DEV = '../dataset/dev/dev.csv'
    TEST = '../dataset/test/test.csv'
    WORDNET = '../dataset/wordnet.pickle'
    SAVE = '../dataset_aug/'


if __name__ == '__main__':
    
    """
        Train 데이터 증강       : Augmentation
        Dev, Test 데이터 전처리 : 특수문자 처리, 중복 표현 제거, hanspell(맞춤법 검사) 
    """
    
    train_augment = Adverb_Augmentation(data_path=Path.TRAIN.value, save_path=Path.SAVE.value)
    train_augment.augmentation()
    
    # dev_augment = BertAugmentaion(data_path=Path.DEV.value,
    #                             save_path=Path.SAVE.value,
    #                             wordnet_path=Path.WORDNET.value)
    
    # test_augment = BertAugmentaion(data_path=Path.TEST.value,
    #                             save_path=Path.SAVE.value,
    #                             wordnet_path=Path.WORDNET.value)
    
    # TRAIN 데이터 증강
    # train_augment.train_augmentationV1(file_name='train_augmentV1.csv', save=True)
    # train_augment.train_augmentationV2(file_name='train_augmentV2.csv', save=True)
    # train_augment.train_augmentationV3(file_name='train_augmentV3.csv', save=True)
    
    # DEV, TEST 전처리(맞춤법 검사, 특수문자 처리)
    # dev_augment.val_preprocessing(file_name='dev_spellcheck.csv')
    # test_augment.val_preprocessing(file_name='test_spellcheck.csv')