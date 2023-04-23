from typing import List, Dict

from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from utils import Vocab

import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, data: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        inputs = [torch.tensor(d['text']).cuda() for d in data] #(3)
        labels = [d['intent'] for d in data]
        inputs = pad_sequence(inputs, batch_first=True) #(4)
        labels = torch.tensor(labels).cuda() #(5)
        return { #(6)
        'text': inputs, 
        'intent': labels
        }

    def collate_fn_test(self, data: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        inputs = [torch.tensor(d['text']).cuda() for d in data] #(3)
        IDs = [d['id'] for d in data]
        #print(IDs)
        inputs = pad_sequence(inputs, batch_first=True) #(4)
        #IDs = torch.tensor(IDs).cuda() #(5)
        return { #(6)
        'text': inputs, 
        'id': IDs
        }
        # raise NotImplementedError



    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn_tags(self, samples):
        
        # TODO: implement collate_fn
        # raise NotImplementedError
        # for d in samples:
        #     print('d',d)
        inputs = [torch.tensor(d['tokens']).cuda() for d in samples] #(3)
        labels = [torch.tensor(d['tags']).cuda() for d in samples]
        
        # for d in inputs:
        #     print('d',d)
        #     print('d',d.shape)
        #     print()

        # first:
        # inputs = pad_sequence(inputs, batch_first=True) #(4)
        # labels = pad_sequence(labels, batch_first=True)  #(5)

        #print('inputs',inputs)

        inputs = pad_sequence(inputs, batch_first=True) #(4)
        labels = pad_sequence(labels, batch_first=True, padding_value = -100)  #(5)
        
        
        #print('labels',labels)
        return { #(6)
        'tokens': inputs, 
        'tags': labels
        }

    def collate_fn_tags_test(self, samples):
        # TODO: implement collate_fn
        # raise NotImplementedError
        # for d in samples:
        #     print('d',d)
        inputs = [torch.tensor(d['tokens']).cuda() for d in samples] #(3)
        IDs = [d['id'] for d in samples]
        
        
        inputs = pad_sequence(inputs, batch_first=True ) #(4)
        
        return { #(6)
        'tokens': inputs, 
        'id': IDs
        }
