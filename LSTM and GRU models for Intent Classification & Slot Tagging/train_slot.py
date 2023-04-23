import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from slotmodel import SeqTaggerLSTM, SeqTaggerGRU
from utils import Vocab

import pandas as pd
import os

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    #raise NotImplementedError

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"  # 9 classes
    iag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, iag2idx, args.max_len)
        for split, split_data in data.items()  
    }
    # print(len(iag2idx))


    for i in datasets['train']:
        #print(i['tokens'])
        i['tokens'] = vocab.encode(i['tokens'])
        # print(vocab.encode(i['tokens']))
        # print(vocab.encode_batch(i['tokens']))
        # print("i['tokens']" ,i['tokens'])
        # print()
        # print(i['tags'])
        i['tags'] =[datasets['train'].label2idx(d) for d in i['tags']]
        #print(i['tags'])
       
        

    for i in datasets['eval']:
        i['tokens'] = vocab.encode(i['tokens'])       
        i['tags'] =[datasets['eval'].label2idx(d) for d in i['tags']]
    
    
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(dataset = datasets['train'], batch_size = args.batch_size, shuffle = True
    ,collate_fn = datasets['train'].collate_fn_tags)

    # iter_loader = iter(train_loader)
    # batch1 = next(iter_loader)
    # print(batch1)

    val_loader = DataLoader(datasets['eval'], batch_size = args.batch_size, shuffle = False
    ,collate_fn = datasets['eval'].collate_fn_tags)

    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    #print(embeddings.shape[0])
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTaggerGRU (embeddings, 
    args.hidden_size, 
    args.num_layers, 
    args.dropout, 
    args.bidirectional,
    len(iag2idx)).cuda()

    print(model.eval())


    # TODO: init optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    loss_function = torch.nn.CrossEntropyLoss(ignore_index = -100).cuda()

    def train_model(data_loader, model, loss_function, optimizer):

        # token and joint accuracy
        total_joint = 0
        total_token = 0
        correct_joint = 0
        correct_token = 0

        Total_loss = 0
        
        model.train()

        for Batch in data_loader:

            #print(i)
            #print(Batch)

            data = Batch['tokens']
            label = Batch['tags']


            #label_onehot = torch.nn.functional.one_hot(label, num_classes = 150)


            output = model(data)
            
            output1 = output.view(-1, 9) 
            label1 = label.view(-1)
            



            #print('output: ',output.shape)

            

            #print('label', label.shape)

            loss = loss_function(output1, label1)

            

            Total_loss += loss.item()

            #print('loss:' ,loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


           
            # calculate accuracy
            for i in range(output.shape[0]):
               
                train_ans = torch.argmax(output[i], dim=1)

                # print('train_ans',train_ans)
                # print('label[i]0 ',label[i])
                #print('data[i].shape',data[i])

                if len((data[i] == 0).nonzero()) != 0 :
                    
                    Ans_length = (data[i] == 0).nonzero().flatten()[0].item()
                    #print('label[i]1',label[i])
                    label_real = label[i][:Ans_length]
                    train_real = train_ans[:Ans_length]
                    #print('label_real',label_real)
                else:
                    label_real = label[i]
                    train_real = train_ans


                correct_token += (train_real == label_real).sum().item()

                total_token += len(train_real)
                # print('total_token',total_token)

                if train_real.equal(label_real):
                    correct_joint += 1
                
                total_joint += 1

                # print('total_joint',total_joint)
                # print('correct_token',correct_token)
                # print('correct_joint',correct_joint)
                # print()

            
            
            #print('label',label.shape)

            
            #print('correct',correct)


        train_acc_joint = 100 * correct_joint / total_joint
        train_acc_token = 100 * correct_token / total_token
        train_loss = Total_loss / total_token

        print("Training Joint Accuracy: {}%".format(train_acc_joint))
        print("Training Token Accuracy: {}%".format(train_acc_token))
        print("Training Loss: {}".format(train_loss))
        return train_acc_joint, train_acc_token, train_loss
    
    def val_model(data_loader, model, loss_function):

        # token and joint accuracy
        total_joint = 0
        total_token = 0
        correct_joint = 0
        correct_token = 0

        Total_loss = 0

        model.eval()
        with torch.no_grad():
            for Batch in data_loader:

                data = Batch['tokens']
                label = Batch['tags']

                output = model(data)

                output1 = output.view(-1, 9) 
                label1 = label.view(-1)

                loss = loss_function(output1, label1)
                Total_loss += loss.item()

                # calculate accuracy
                for i in range(output.shape[0]):
                    val_ans = torch.argmax(output[i], dim=1)

                    if len((data[i] == 0).nonzero()) != 0 :
                    
                        Ans_length = (data[i] == 0).nonzero().flatten()[0].item()
                    
                        label_real = label[i][:Ans_length]
                        val_real = val_ans[:Ans_length]
                    
                    else:
                        label_real = label[i]
                        val_real = val_ans

                    correct_token += (val_real == label_real).sum().item()

                    total_token += len(val_real)

                    if val_real.equal(label_real):
                        correct_joint += 1
                
                    total_joint += 1
        val_acc_joint = 100 * correct_joint / total_joint
        val_acc_token = 100 * correct_token / total_token
        val_loss = Total_loss / total_token

        print("Validation Joint Accuracy: {}%".format(val_acc_joint))
        print("Validation Token Accuracy: {}%".format(val_acc_token))
        print("Validation Loss: {}".format(val_loss))

        return val_acc_joint, val_acc_token, val_loss
    #-------------------------------------------------------------------------#


    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    
    #Define filename
    Type = 'GRU'
    Name = 'GRUBanorHidden512Drop0.5Layer2_Ba128Ep200lr-3'
    CSVName = Name + '_lossRecord.csv'
    ModelProcess = Name+'_process.pt'
    ModelBest = Name+'_best.pt'

    if not os.path.isdir(args.ckpt_dir / Type / Name):
        os.makedirs(args.ckpt_dir / Type / Name)

    
    #record loss
    df_losscreat=pd.DataFrame(columns=['epoch','Train_Acc_Joint'
    ,'Train_Acc_Token','Train_Loss'
    ,'Val_Acc_Joint','Val_Acc_Token', 'Val_Loss'])

    df_losscreat.to_csv(args.ckpt_dir / Type / Name / CSVName , index = False)

    print("Untrained validation: ")
    validation_accjointbest, validation_acctokenbest, validation_lossbest = val_model(val_loader, model, loss_function)

    # #load model
    # lastpoint = torch.load(args.ckpt_dir / "GRUBanorHidden512Drop0.1Layer4_Ba64Ep600lr-4" / "GRUBanorHidden512Drop0.1Layer4_Ba64Ep600lr-4_best.pt")
    # lastepoch = lastpoint['epoch']
    # lastloss = lastpoint['Val_Loss']
    # lastValAccJoint = lastpoint['Val_Accuracy_Joint']
    # lastValAccToken = lastpoint['Val_Accuracy_Token']
    # model.load_state_dict(lastpoint['model_state_dict'])
    # optimizer.load_state_dict(lastpoint['optimizer_state_dict'])

    # validation_accjointbest = lastValAccJoint
    # validation_acctokenbest = lastValAccToken
    # validation_lossbest = lastloss

    # print('last epoch: ',lastepoch)
    # print('last loss: ',lastloss) 
    # print('last Val Acc Joint: ',lastValAccJoint)
    # print('last Val Acc Token: ',lastValAccToken)  
    # print(model.eval())

    

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        # print(torch.cuda.is_available())
        #print('epoch: ',epoch)
        #print()

        training_acc_joint, training_acc_token, training_loss = train_model(train_loader, model, loss_function, optimizer)
        validation_acc_joint, validation_acc_token, validation_loss = val_model(val_loader, model, loss_function)



        #save each epoch loss
        allloss={
            'epoch':[epoch],
            'Train_Acc_Joint':[training_acc_joint],
            'Train_Acc_Token':[training_acc_token],
            'Train_Loss':[training_loss],
            'Val_Acc_Joint':[validation_acc_joint],
            'Val_Acc_Token':[validation_acc_token],
            'Val_Loss':[validation_loss]
        }
        # append data frame to CSV file
        pd.DataFrame(allloss).to_csv(args.ckpt_dir / Type / Name / CSVName, mode='a', index=False, header=False)
        print('Accuracy and Loss are recorded!')


        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Val_Accuracy_Joint': validation_acc_joint,
                'Val_Accuracy_Token': validation_acc_token,
                'Val_Loss': validation_loss,
                
                }, args.ckpt_dir / Type  / Name / ModelProcess )
        print("save each epoch")

        if validation_acc_joint>= validation_accjointbest :
            validation_accjointbest = validation_acc_joint 

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Val_Accuracy_Joint': validation_acc_joint,
                'Val_Accuracy_Token': validation_acc_token,
                'Val_Loss': validation_loss,
                
                }, args.ckpt_dir / Type / Name / ModelBest)
            print("save best epoch")
        print("Current Best Validation Joint Accuracy: {}%".format(validation_accjointbest))
        


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

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