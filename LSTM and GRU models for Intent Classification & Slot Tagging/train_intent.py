import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import os

import pandas as pd

import torch
torch.cuda.current_device()

from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

from intentmodel import SeqClassifierLSTM 
from intentmodel import SeqClassifierGRU

import numpy as np

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]




def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"  # 150 classes
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()  
    }


    for i in datasets['train']:
        # print(i['text'])
        i['text'] = vocab.encode(i['text'])
        # print(i['text'])
        i['intent'] =datasets['train'].label2idx(i['intent'])
       
        

    for i in datasets['eval']:
        i['text'] = vocab.encode(i['text'])       
        i['intent'] =datasets['eval'].label2idx(i['intent'])
    
    
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(dataset = datasets['train'], batch_size = args.batch_size, shuffle = True
    ,collate_fn = datasets['train'].collate_fn)

    # iter_loader = iter(train_loader)
    # batch1 = next(iter_loader)
    # print(batch1)

    val_loader = DataLoader(datasets['eval'], batch_size = args.batch_size, shuffle = False
    ,collate_fn = datasets['eval'].collate_fn)

    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    #print(embeddings.shape[0])
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifierGRU (embeddings, 
    args.hidden_size, 
    args.num_layers, 
    args.dropout, 
    args.bidirectional,
    len(intent2idx)).cuda()
    
    model.eval()

    print(model.eval())

    

    # TODO: init optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    loss_function = torch.nn.CrossEntropyLoss().cuda()

    def train_model(data_loader, model, loss_function, optimizer):

        total = 0
        correct = 0
        Total_loss = 0
        
        model.train()

        for Batch in data_loader:

            #print(i)
            #print(Batch)

            data = Batch['text']
            label = Batch['intent']


            #label_onehot = torch.nn.functional.one_hot(label, num_classes = 150)


            output = model(data)


            #print('output: ',output.shape)

            

            #print('label_onehot', label.shape)

            loss = loss_function(output, label)

            Total_loss += loss.item()

            #print('loss:' ,loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # n = 0
            # for i in output :
            #     print(n,i.shape)
            #     n += 1
            #     print()


            train_ans = torch.argmax(output, dim=1)
            #print('train_ans',train_ans.shape)



            total += label.size(0)
            #print('total',total)
            
            #print('label',label.shape)

            correct += (train_ans == label).sum().item()
            #print('correct',correct)


        train_acc = 100 * correct / total
        train_loss = Total_loss / total

        print("Training Accuracy: {}%".format(train_acc))
        print("Training Loss: {}".format(train_loss))
        return train_acc, train_loss
    
    def val_model(data_loader, model, loss_function):

        total = 0
        correct = 0
        Total_loss = 0

        model.eval()
        with torch.no_grad():
            for Batch in data_loader:

                data = Batch['text']
                label = Batch['intent']

                output = model(data)

                loss = loss_function(output, label)
                Total_loss += loss.item()

                val_ans = torch.argmax(output, dim=1)


                total += label.size(0)

                correct += (val_ans  == label).sum().item()
      
        val_acc = 100 * correct / total
        val_loss = Total_loss / total
        print("Validation Accuracy: {}%".format(val_acc))
        print("Validation Loss: {}".format(val_loss))

        return val_acc, val_loss
    #-------------------------------------------------------------------------#


    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    
    #Define filename
    Type = 'test/GRU'
    Name = '2GRU3BaNorHidden512Drop0.1Layer3_Ba128Ep200lr-3'
    CSVName = Name + '_lossRecord.csv'
    ModelProcess = Name+'_process.pt'
    ModelBest = Name+'_best.pt'

    if not os.path.isdir(args.ckpt_dir / Type / Name):
        os.makedirs(args.ckpt_dir / Type / Name)
    
    #record loss
    df_losscreat=pd.DataFrame(columns=['epoch','Train_Acc','Train_Loss','Val_Acc','Val_Loss'])
    df_losscreat.to_csv(args.ckpt_dir / Type / Name / CSVName , index = False) 

    print("Untrained validation: ")
    validation_accbest, validation_lossbest = val_model(val_loader, model, loss_function)
    # print("Untrained validation accuracy: {}%".format(validation_accbest))
    # print("Untrained validation loss: {}%".format(validation_lossbest))

    # #load model
    # lastpoint = torch.load(args.ckpt_dir / type / "GRUBanorHidden512Drop0.2Layer3_Ba128Ep400lr-4Adamax" / "GRUBanorHidden512Drop0.2Layer3_Ba128Ep400lr-4Adamax_best.pt")
    # lastepoch = lastpoint['epoch']
    # lastloss = lastpoint['Val_Loss']
    # lastValAcc = lastpoint['Val_Accuracy']
    # model.load_state_dict(lastpoint['model_state_dict'])
    # optimizer.load_state_dict(lastpoint['optimizer_state_dict'])

    # validation_accbest = lastValAcc
    # validation_lossbest = lastloss

    # print('last epoch: ',lastepoch)
    # print('last loss: ',lastloss) 
    # print("last Val Acc: {}%".format(lastValAcc)) 
    # print(model.eval())

    

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        # print(torch.cuda.is_available())
        #print('epoch: ',epoch)
        #print()

        training_acc, training_loss = train_model(train_loader, model, loss_function, optimizer)
        validation_acc, validation_loss = val_model(val_loader, model, loss_function)



        #save each epoch loss
        allloss={
            'epoch':[epoch],
            'Train_Acc':[training_acc],
            'Train_Loss':[training_loss],
            'Val_Acc':[validation_acc],
            'Val_Loss':[validation_loss]
        }
        # append data frame to CSV file
        pd.DataFrame(allloss).to_csv(args.ckpt_dir / Type / Name / CSVName, mode='a', index=False, header=False)
        print('Accuracy and Loss are recorded!')


        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Val_Accuracy': validation_acc,
                'Val_Loss': validation_loss,
                
                }, args.ckpt_dir / Type / Name / ModelProcess )
        print("save each epoch")

        if validation_acc >= validation_accbest :
            validation_accbest = validation_acc 

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Val_Accuracy': validation_accbest,
                'Val_Loss': validation_loss,
                
                }, args.ckpt_dir / Type / Name / ModelBest)
            print("save best epoch")
        print("Current Best Validation Accuracy: {}%".format(validation_accbest))
        

    # TODO: Inference on test set


def parse_args() -> Namespace:
    

    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )


    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    #parser.add_argument("--wd", type=float, default=0)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=200)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = "cuda"
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
