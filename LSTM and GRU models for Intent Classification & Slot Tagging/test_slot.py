import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from slotmodel import SeqTaggerGRU, SeqTaggerLSTM
from utils import Vocab

from torch.utils.data import DataLoader
import pandas as pd


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"  # 9 classes
    iag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, iag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    for i in dataset:
        i['tokens'] = vocab.encode(i['tokens'])     

    test_loader = DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = False
    ,collate_fn = dataset.collate_fn_tags_test)  
        

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTaggerGRU (
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    ).cuda()

    model.eval()


    #Model filename
    Type = 'GRU'
    Name = 'GRUBanorHidden512Drop0.5Layer2_Ba128Ep200lr-3'
    ModelBest = Name+'_best.pt'
    
    modelfile = args.ckpt_dir / Type / Name / ModelBest

    bestpoint = torch.load(modelfile)
    # load weights into model
    bestepoch = bestpoint['epoch']
    bestloss = bestpoint['Val_Loss']
    bestValAccJoint = bestpoint['Val_Accuracy_Joint']
    bestValAccToken = bestpoint['Val_Accuracy_Token']
    model.load_state_dict(bestpoint['model_state_dict'])
    #optimizer.load_state_dict(bestpoint['optimizer_state_dict'])

    validation_accjointbest = bestValAccJoint
    validation_acctokenbest = bestValAccToken
    validation_lossbest = bestloss

    print('Best Epoch: ',bestepoch)
    print('Best Loss: ',bestloss) 
    print('Best Joint Val Acc: ',bestValAccJoint)
    print('Best Token Val Acc: ',bestValAccToken)  
    print(model.eval())
    model.eval()

    # TODO: predict dataset
    Test_id = []
    Test_ans = []
    for Batch in test_loader:

        data = Batch['tokens']
        ID = Batch['id']
        
        Test_id.extend(ID)

        output = model(data)

        for i in range(output.shape[0]):
            test_ans = torch.argmax(output[i], dim=1) 
            
            if len((data[i] == 0).nonzero()) != 0 :
                    
                Ans_length = (data[i] == 0).nonzero().flatten()[0].item()
                      
                test_real = test_ans[:Ans_length]
                    
            else:

                test_real = test_ans     

            Test_ans.append(test_real.tolist())

            # print(Test_ans)
            # print()


    # TODO: write prediction to file (args.pred_file)
    Test_label = []
    for ans in Test_ans :

        
        ans_label = []

        for token in ans:
            ans_label.append(dataset.idx2label(token))
                 
        ans_label = " ".join(ans_label)
        Test_label.append(ans_label)

    #print(Test_label)

    df_test = pd.DataFrame(
    {'id': Test_id,
     'tags': Test_label,
    })
    df_test.to_csv(args.ckpt_dir / Type / Name / args.pred_file, index = False)
    print('Test csv file is saved!')

    #raise NotImplementedError


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)