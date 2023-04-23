import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from intentmodel import SeqClassifierLSTM
from intentmodel import SeqClassifierGRU
from utils import Vocab

from torch.utils.data import DataLoader
import pandas as pd


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    for i in dataset:
        i['text'] = vocab.encode(i['text'])     

    test_loader = DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = False
    ,collate_fn = dataset.collate_fn_test)  
        

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifierGRU(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    ).cuda()

    model.eval()
    
    # model path
    Type = 'test/GRU'
    Name = '2GRU3BaNorHidden512Drop0.1Layer3_Ba128Ep200lr-3'
    ModelBest = Name+'_best.pt'

    modelfile = args.ckpt_path / Type / Name / ModelBest

    bestpoint = torch.load(modelfile)
    # load weights into model
    bestepoch = bestpoint['epoch']
    bestValloss = bestpoint['Val_Loss']
    bestValAcc = bestpoint['Val_Accuracy']
    model.load_state_dict(bestpoint['model_state_dict'])

    print('best epoch: ',bestepoch)
    print('best loss: ',bestValloss) 
    print('best Val Acc: ',bestValAcc) 
    print(model.eval())

    # TODO: predict dataset
    Test_id = []
    Test_ans = []
    for Batch in test_loader:

        data = Batch['text']
        ID = Batch['id']
        
        Test_id.extend(ID)

        output = model(data)
        
        train_ans = torch.argmax(output, dim=1)

        Test_ans.extend(train_ans.tolist())

    #print(Teat_ans)

    # TODO: write prediction to file (args.pred_file)
    Test_label = []
    for ans in Test_ans : 
        Test_label.append(dataset.idx2label(ans))
    #print(Test_label)

    df_test = pd.DataFrame(
    {'id': Test_id,
     'intent': Test_label,
    })
    df_test.to_csv(args.ckpt_path / Type / Name / args.pred_file , index = False)
    print('Test csv file is saved!')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
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
