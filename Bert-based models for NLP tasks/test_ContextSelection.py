import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from tqdm import trange,tqdm
from dataset import ContextSelectionForTestDataset

from transformers import AutoTokenizer, AutoModelForMultipleChoice

from torch.utils.data import DataLoader
import pandas as pd
import os


def main(args):
    # model prepare
   
    # PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"  # 指定繁簡中文 BERT-BASE 預訓練模型
    
    # # tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME,truncation_side='right')
    # tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    
    # define your own tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/ContextSelection/', use_fast=True)

    # model = AutoModelForMultipleChoice.from_pretrained(PRETRAINED_MODEL_NAME).cuda()
    # define your own model
    model = torch.load('./model/ContextSelection/CS_bestWholeModel.pt').cuda()
    
   
    os.makedirs(args.reproduce_dir, exist_ok=True)

  
    
    
    # load data
    
    contextdata = json.loads(args.context_dir.read_text(encoding="utf-8"))

    testdata = json.loads(args.test_dir.read_text(encoding="utf-8"))
    
    
    encoding_test = ContextSelectionForTestDataset(testdata , contextdata , tokenizer = tokenizer)
    
    
   
    CSVName = "CSprediction.csv"
    JSONName = "CSprediction.json"
    

    print(model.eval())
    

    Test_id = []
    Test_ans = []
   
    
    for n, input_data in zip(testdata ,tqdm(encoding_test)):
        
        
        
        model.eval()
            
        with torch.no_grad():
            outputs = model( **{k: v.unsqueeze(0).cuda() for k, v in input_data[0].items()})
            
            prediction = torch.argmax(outputs.logits, dim=1)
            
            Test_ans.append(int(input_data[1].loc[prediction.item(),'Paragraph']))

            Test_id.append(input_data[2])
            n["relevant"] = int(input_data[1].loc[prediction.item(),'Paragraph'])
           
            
    # save prediction file for QA
        
    with open( args.reproduce_dir / JSONName, "w", encoding='utf8') as outfile:
        json.dump(testdata, outfile,ensure_ascii=False)  
    print('Test json file is saved!')     
            
             
    predictpath = args.reproduce_dir / JSONName
    predictdata = json.loads(predictpath.read_text(encoding="utf-8"))
    
    ID =[]
    question = []
    relevant = []
    
    for i in predictdata:
        ID.append(i['id'])
        question.append(i['question'])
        relevant.append(contextdata[i['relevant']])
        
    df_test = pd.DataFrame(
    {'context': relevant,
     'question': question,
     'id': ID,
    })
    df_test['answer_start']=""
    df_test['text']=""
    df_test.to_csv(args.reproduce_dir / CSVName , index = False, encoding='utf-8-sig')
    print('Prediction csv file is saved!')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    
    parser.add_argument(
        "--reproduce_dir",
        type=Path,
        help="Directory to the dataset.",
        default='./reproducefiles/',
    )
    parser.add_argument(
        "--context_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/context.json",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Path to the test file.",
        default="./data/test.json",
    )

    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/ContextSelection/"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")


    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
