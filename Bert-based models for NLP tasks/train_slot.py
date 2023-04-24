import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import os
import sys
import time


from transformers import AutoTokenizer, AutoModelForTokenClassification

import pandas as pd

import torch
torch.cuda.current_device()

from torch.utils.data import DataLoader
from tqdm import trange,tqdm

from dataset import SlotForTrainDataset


import numpy as np




def main(args):
    
    
     # load data
    
    
    traindata = json.loads(args.train_dir.read_text(encoding="utf-8"))
 
    
    validdata = json.loads(args.valid_dir.read_text(encoding="utf-8"))
    
    
    label2id = json.loads(args.label2id_dir.read_text(encoding="utf-8"))
    
   
    
    # model prepare
   
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'  # 指定繁簡中文 BERT-BASE 預訓練模型
    
    # tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME,truncation_side='right')
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=len(label2id) ).cuda()

    print(model.eval())
    
    
    
    
    
    
    # for i,a in encoding_train:
    #     print(i)
    #     print(a)
    #     time.sleep(2)
    
    
    

    #Define filename
    Type = 'bert-base-uncased'
    Name = Type+'_Ebs64Epoch5AdamWLr3e-5'
    CSVName = Name + '_lossRecord.csv'
    # ModelProcess = Name+'_process.pt'
    ModelBest = Name+'_best.pt'

    if not os.path.isdir(args.ckpt_dir / Type / Name):
        os.makedirs(args.ckpt_dir / Type / Name)
    
    #record loss
    df_loss=pd.DataFrame(columns=['epoch','Train_AccToken','Train_AccJoint','Train_Loss','Val_AccToken','Val_AccJoint','Val_Loss'])
    df_loss.to_csv(args.ckpt_dir / Type / Name / CSVName , index = False)
    
    # print(traindata)
    # time.sleep(2)
    
    
    # Dataset define
    encoding_train = SlotForTrainDataset(traindata , label2id, tokenizer = tokenizer)
    encoding_valid = SlotForTrainDataset(validdata , label2id, tokenizer = tokenizer)
    
    # for i in encoding_train:
    #     print(i)
    #     time.sleep(2)
    
    # sys.exit()
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    val_accuracybest = 0
    bestepoch = 0
    
    
    
    train_dataloader = DataLoader(
        encoding_train, shuffle=True, batch_size=args.batch_size,  collate_fn=encoding_train.collate_fn
    )
    
    valid_dataloader = DataLoader(
        encoding_valid, shuffle=False, batch_size=args.batch_size,  collate_fn=encoding_valid.collate_fn
    )
    
    # for i in train_dataloader:
    #     print(i)
    #     time.sleep(2)
    
    
    # sys.exit()

    accum_iter = args.gradient_accumulation
    
    
    
   
    for epoch in epoch_pbar:
        
        
        train_total = 0
        train_correct = 0 
        train_loss = 0 
        
        train_correct_joint = 0
        train_total_joint = 0
        
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
            labelid = batch.pop('Tagsid')
            
            for i in batch:
                batch[i] = batch[i].squeeze(0).to("cuda")
    
            
            # print(batch)
            # print(labelid)
            
            outputs = model( **batch , labels = labelid)
        
            loss = outputs.loss
            
            loss = loss / accum_iter
            
            loss.backward()
            
            
            
            train_loss += loss.item()
            # print(loss.item())
            # print(outputs.logits)
            predict = torch.argmax(outputs.logits,dim=2)

            # print(predict)
            # print(labelid)
            # time.sleep(5)
            
            for i in range(labelid.shape[0]):
                
                y = labelid[i]
                x = predict[i]
                
                x = x[y!=-100]
                y = y[y!=-100]
                
                
                
                if x.equal(y):
                    train_correct_joint += 1
                    
                train_total_joint += 1
                for k in range(labelid.shape[1]):
                    if labelid[i][k] == -100:
                        continue
                    elif labelid[i][k] == predict[i][k]:
                        train_correct += 1
                        train_total += 1
                    else:
                        train_total += 1
            
            # train_correct += (predict == labelid ).sum().item()
            
            # print(predict.shape)
            # print(labelid.shape)
            
            # # print(train_correct)
            
            # # print(len(label_num))
            
            # train_total += len(labelid)
            # print(train_correct)
            # print(train_total)
            # time.sleep(2)
        
            # if predict == label_num:
            #     train_correct += 1
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

        train_accuracy = train_correct / train_total
        train_totalloss = train_loss / train_total
        train_jointacc = train_correct_joint / train_total_joint
        
        # validation------------------
        print("***** Running Evaluation *****")
        print(f"  Num examples = {len(encoding_valid)}")  
        
        val_total = 0
        val_correct = 0 
        val_loss = 0
        
        val_correct_joint = 0
        val_total_joint = 0
        
            
        for batch in tqdm(valid_dataloader):
            
            model.eval()
            
            labelid = batch.pop('Tagsid')
            
            
            with torch.no_grad():
                
                outputs = model( **batch , labels = labelid)
            
                loss = outputs.loss
                # print(outputs)
                val_loss += loss.item()
                
                predict = torch.argmax(outputs.logits, dim=2)
                
                
                for i in range(labelid.shape[0]):
                    
                    y = labelid[i]
                    x = predict[i]
                
                    x = x[y!=-100]
                    y = y[y!=-100]
                    
                    # train_correct_joint += 1
                    # if predict[i].equal(labelid[i]):
                    if x.equal(y):
                        val_correct_joint += 1
                    
                    val_total_joint += 1
                    
                    for k in range(labelid.shape[1]):
                        if labelid[i][k] == -100:
                            continue
                        elif labelid[i][k] == predict[i][k]:
                            val_correct += 1
                            val_total += 1
                        else:
                            val_total += 1
                
                # val_total += len(labelid)
                # val_correct += (prediction == labelid).sum().item()
                
                    
                # print('correct: ',val_correct)
                # print('total: ',val_total)
            
        val_accuracy = val_correct / val_total
        val_totalloss = val_loss / val_total
        val_jointacc = val_correct_joint / val_total_joint
        
        allloss={
            'epoch':[epoch],
            'Train_AccToken':[train_accuracy],
            'Train_AccJoint':[train_jointacc],
            'Train_Loss':[train_totalloss],
            'Val_AccToken':[val_accuracy],
            'Val_AccJoint':[val_jointacc],
            'Val_Loss':[val_totalloss],
        }
        # append data frame to CSV file
        pd.DataFrame(allloss).to_csv(args.ckpt_dir / Type / Name / CSVName, mode='a', index=False, header=False)
        # logger.info("Accuracy and Loss are recorded!")
        print('Accuracy and Loss are recorded!')
           
        
        if val_jointacc >= val_accuracybest :
            bestepoch = epoch
            val_accuracybest = val_jointacc 

            torch.save({
                'epoch': epoch,
                # 'model_state_dict': model.state_dict(),
                'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'Val_Accuracy': val_accuracybest,
                
                }, args.ckpt_dir / Type / Name / ModelBest)
            # logger.info("save best epoch!")
            print("save best epoch!")
        #logger.info(f"  Current Best Validation Accuracy: {val_accuracybest}%")
        #logger.info(f"  Current Best Validation Accuracy Epoch: {bestepoch}")
        print("Current Best Validation Accuracy: {}".format(val_accuracybest))
        print("Current Best Validation Accuracy Epoch: {}".format(bestepoch))
    
    
    sys.exit()
        


def parse_args() -> Namespace:
    

    parser = ArgumentParser()
    
    
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/train.json",
    )
    
    parser.add_argument(
        "--valid_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/eval.json",
    )
    
    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    
    parser.add_argument(
        "--label2id_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/tag2idx.json",
    )
     
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )


    # data
    parser.add_argument("--max_len", type=int, default=512)


    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)
    
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
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