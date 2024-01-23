import torch
import pytorch_lightning as pl
import argparse
import pandas as pd
import re
import json
import transformers
import pickle
import torch.nn.functional as F

from pl_train import *

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

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../dataset/train/train.csv')
    parser.add_argument('--dev_path', default='../dataset/train/dev.csv')
    parser.add_argument('--test_path', default='../dataset/train/dev.csv')
    parser.add_argument('--predict_path', default='../dataset/test/test_data.csv')
    parser.add_argument('--model_path', default='../model/model_special_token_5.pt')
    args = parser.parse_args()

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load(args.model_path)


    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = torch.cat(predictions)

    prob = F.softmax(predictions, dim=-1).detach().cpu().numpy().tolist()
    logits = predictions.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    label = num_to_label(result)

    test_id = pd.read_csv(args.predict_path)['id']

    output = pd.DataFrame({'id': test_id, 'pred_label': label, 'probs': prob})
    output.to_csv('./prediction/submission.csv', index=False)