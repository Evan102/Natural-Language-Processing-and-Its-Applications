import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import os
import sys
import time

import pandas as pd

import torch
torch.cuda.current_device()

from dataset import ContextSelectionForTrainDataset

from torch.utils.data import DataLoader

from tqdm import trange,tqdm



import numpy as np


from transformers import AutoTokenizer, AutoModelForMultipleChoice




def main(args):
   
   
   # model prepare
   
    PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 指定繁簡中文 BERT-BASE 預訓練模型
    
    # tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME,truncation_side='right')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    model = AutoModelForMultipleChoice.from_pretrained(PRETRAINED_MODEL_NAME).cuda()
    
    model.eval()

    print(model.eval())
    
    # load data
    
    contextdata = json.loads(args.context_dir.read_text(encoding="utf-8"))
    
    
    traindata = json.loads(args.train_dir.read_text(encoding="utf-8"))
 
    
    validdata = json.loads(args.valid_dir.read_text(encoding="utf-8"))

    
    
    #JsontoCSV
    os.makedirs(args.reproduce_dir, exist_ok=True)
    
    ID =[]
    question = []
    relevant = []
    answerstart = []
    answertext = []
    
    for i in traindata:
        ID.append(i['id'])
        question.append(i['question'])
        relevant.append(contextdata[i['relevant']])
        answerstart.append(int(i['answer']['start']))
        answertext.append(i['answer']['text'])
        
    df_train = pd.DataFrame(
    {'context': relevant,
     'question': question,
     'id': ID,
     'answer_start': answerstart,
     'text': answertext,
    })
    
    df_train.to_csv(args.reproduce_dir / 'train.csv' , index = False, encoding='utf-8-sig')
    print('Training csv file is saved!')
    
    
    ID =[]
    question = []
    relevant = []
    answerstart = []
    answertext = []
    
    for i in validdata:
        ID.append(i['id'])
        question.append(i['question'])
        relevant.append(contextdata[i['relevant']])
        answerstart.append(int(i['answer']['start']))
        answertext.append(i['answer']['text'])
        
    df_valid = pd.DataFrame(
    {'context': relevant,
     'question': question,
     'id': ID,
     'answer_start': answerstart,
     'text': answertext,
    })
    
    df_valid.to_csv(args.reproduce_dir / 'valid.csv' , index = False, encoding='utf-8-sig')
    print('validation csv file is saved!')
    
    
    #Define filename
    Type = 'Test_hfl_chinese-roberta-wwm-ext'
    Name = Type+'_Ebs64Epoch5AdamWLr3e-5'
    CSVName = Name + '_lossRecord.csv'
    # ModelProcess = Name+'_process.pt'
    ModelBest = Name+'_best.pt'

    if not os.path.isdir(args.ckpt_dir / Type / Name):
        os.makedirs(args.ckpt_dir / Type / Name)
    
    #record loss
    df_loss=pd.DataFrame(columns=['epoch','Train_Acc','Train_Loss','Val_Acc','Val_Loss'])
    df_loss.to_csv(args.ckpt_dir / Type / Name / CSVName , index = False)
    
    # Dataset define
    encoding_train = ContextSelectionForTrainDataset(traindata , contextdata , tokenizer = tokenizer)
    encoding_valid = ContextSelectionForTrainDataset(validdata , contextdata , tokenizer = tokenizer)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    val_accuracybest = 0
    bestepoch = 0
    
    train_dataloader = DataLoader(
        encoding_train, shuffle=True, batch_size=args.batch_size
    )
    
    valid_dataloader = DataLoader(
        encoding_valid, shuffle=False, batch_size=args.batch_size
    )
    
    

    accum_iter = args.gradient_accumulation
    
   
    
    for epoch in epoch_pbar:
        
        
        train_total = 0
        train_correct = 0 
        train_loss = 0 
        
        print("***** Running training *****")
        print(f"  Num examples = {len(encoding_train)}")
        print(f"  Num Epochs = {args.num_epoch}")
        print(f"  Instantaneous batch size per device = {args.batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation}")
        print(f"  Total optimization steps = {args.batch_size*args.gradient_accumulation}")
        print(f"  Learning rate = {args.lr}")

        
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            
            model.train()
            # input_data = input_data.cuda()
            input_data = batch[0]
            for k in input_data:
                input_data[k]=input_data[k].to("cuda")
                
            # for k, v in input_data.items():
            #     print(v)
            #     print(v.unsqueeze(0))
                
            label_num = batch[1].squeeze(1)
            # print(input_data)
            # print(label_num)
            outputs = model( **input_data , labels = label_num)
        
            loss = outputs.loss
            
            loss = loss / accum_iter
            
            loss.backward()
            
            train_loss += loss.item()
            #print(loss.item())
            #print(outputs.logits)
            predict = torch.argmax(outputs.logits, dim=1)
    
            
            train_correct += (predict == label_num).sum().item()
            
            # print(len(label_num))
            
            train_total += len(label_num)
        
            # if predict == label_num:
            #     train_correct += 1
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

        train_accuracy = train_correct / train_total
        train_totalloss = train_loss / train_total
        
        # validation------------------
        print("***** Running Evaluation *****")
        print(f"  Num examples = {len(valid_dataloader)}")  
        
        val_total = 0
        val_correct = 0 
        val_loss = 0
        
            
        for batch in tqdm(valid_dataloader):
            
            model.eval()
            
            input_data = batch[0]
            label_num = batch[1].squeeze(1)
            
            for k in input_data:
                input_data[k]=input_data[k].to("cuda")
            
            with torch.no_grad():
                
                outputs = model( **input_data , labels = label_num)
            
                loss = outputs.loss
                # print(outputs)
                val_loss += loss.item()
                
                prediction = torch.argmax(outputs.logits, dim=1)
                
                
                val_total += len(label_num)
                val_correct += (prediction == label_num).sum().item()
                # if prediction == label_num:
                #     val_correct += 1
                    
                # print('correct: ',val_correct)
                # print('total: ',val_total)
            
        val_accuracy = val_correct / val_total
        val_totalloss = val_loss / val_total
        
        allloss={
            'epoch':[epoch],
            'Train_Acc':[train_accuracy],
            'Train_Loss':[train_totalloss],
            'Val_Acc':[val_accuracy],
            'Val_Loss':[val_totalloss],
        }
        # append data frame to CSV file
        pd.DataFrame(allloss).to_csv(args.ckpt_dir / Type / Name / CSVName, mode='a', index=False, header=False)
        # logger.info("Accuracy and Loss are recorded!")
        print('Accuracy and Loss are recorded!')
           
        
        if val_accuracy >= val_accuracybest :
            bestepoch = epoch
            val_accuracybest = val_accuracy 

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Val_Accuracy': val_accuracybest,
                
                }, args.ckpt_dir / Type / Name / ModelBest)
            # logger.info("save best epoch!")
            print("save best epoch!")
        #logger.info(f"  Current Best Validation Accuracy: {val_accuracybest}%")
        #logger.info(f"  Current Best Validation Accuracy Epoch: {bestepoch}")
        print("Current Best Validation Accuracy: {}%".format(val_accuracybest))
        print("Current Best Validation Accuracy Epoch: {}".format(bestepoch))
    
    
    
    
    
    sys.exit()



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
        "--train_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/train.json",
    )
    
    parser.add_argument(
        "--valid_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/valid.json",
    )
     
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./model/ContextSelection/",
    )


    # data
    parser.add_argument("--max_len", type=int, default=512)


    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)
    
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=32,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = "cuda"
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
