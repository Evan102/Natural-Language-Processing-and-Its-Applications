from typing import List, Dict

from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import time

import torch


class ContextSelectionForTrainDataset(Dataset):
     # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self , data: List[Dict] , context: List[Dict] , tokenizer):
    
        
        # 大數據你會需要用 iterator=True
        self.data = data
        self.len = len(self.data)
        
        self.context = context
        self.question = [ sub['question'] for sub in data ]
        self.paragraphs = [ sub['paragraphs'] for sub in data ]
        self.answerstart = [ sub['answer']['start'] for sub in data ]
        
        self.answertext = [ sub['answer']['text'] for sub in data ]
        
        self.label_map = [ sub['relevant'] for sub in data ]
        
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        
        
        eachquestion = self.question[idx]
        eachparagraphs = self.paragraphs[idx]
        # 將 label 文字也轉換成索引方便轉換成 tensor
        label_id = self.label_map[idx]
            
            
        for i in range(len(eachparagraphs)):
            if label_id == eachparagraphs[i]:
                Ans_Num = i

        label_num_tensor = torch.tensor(Ans_Num).unsqueeze(0).cuda()
        label_paragraph_tensor = torch.tensor(label_id).unsqueeze(0).cuda()
            
        # 建立第一個問題的 BERT tokens 並加入分隔符號 [SEP]
        # word_pieces = ["[CLS]"]
        
        # tokens_question = self.tokenizer.tokenize(eachquestion)
        
        #tokens_question = eachquestion
        paragraphelen = int(512-len(eachquestion))-3
        # word_pieces += tokens_question + ["[SEP]"]
        # len_question = len(word_pieces)
        
        # 第一個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[0]]):
            tokens_p0 = self.context[eachparagraphs[0]][:paragraphelen]
        else:
            tokens_p0 = self.context[eachparagraphs[0]]
        # tokens_p0 = self.tokenizer.tokenize(self.context[eachparagraphs[0]])
        # word_pieces += tokens_p0 + ["[SEP]"]
        # len_p0 = len(word_pieces) - len_question
        
        # 第二個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[1]]):
            tokens_p1 = self.context[eachparagraphs[1]][:paragraphelen]
        else:
            tokens_p1 = self.context[eachparagraphs[1]]
        
        # tokens_p1 = self.tokenizer.tokenize(self.context[eachparagraphs[1]])
        # word_pieces += tokens_p1 + ["[SEP]"]
        # len_p1 = len(word_pieces) - len_question - len_p0
        
        # 第三個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[2]]):
            tokens_p2 = self.context[eachparagraphs[2]][:paragraphelen]
        else:
            tokens_p2 = self.context[eachparagraphs[2]]
        
        # tokens_p2 = self.tokenizer.tokenize(self.context[eachparagraphs[2]])
        # word_pieces += tokens_p2 + ["[SEP]"]
        # len_p2 = len(word_pieces) - len_question - len_p0 - len_p1
        
        # 第四個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[3]]):
            tokens_p3 = self.context[eachparagraphs[3]][:paragraphelen]
        else:
            tokens_p3 = self.context[eachparagraphs[3]]
        
        # tokens_p3 = self.tokenizer.tokenize(self.context[eachparagraphs[3]])
        # word_pieces += tokens_p3 + ["[SEP]"]
        # len_p3 = len(word_pieces) - len_question - len_p0 - len_p1 - len_p2
        
        # 將整個 token 序列轉換成索引序列
        # ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        # tokens_tensor = torch.tensor(ids)
        
        # encoding = self.tokenizer([eachquestion, eachquestion, eachquestion, eachquestion], [tokens_p0, tokens_p1, tokens_p2, tokens_p3], return_tensors = "pt", padding=True, max_length=512, truncation='longest_first')
        encoding = self.tokenizer([eachquestion, eachquestion, eachquestion, eachquestion], [tokens_p0, tokens_p1, tokens_p2, tokens_p3], return_tensors = "pt", max_length=512, padding='max_length')
        # print(encoding['input_ids'].shape)
        # print(encoding)
        # time.sleep(5)
        # 
        # segments_tensor = torch.tensor([0] * len_question + [1] *len_p0 + [2] *len_p1 + [3] *len_p2 + [4] *len_p3,  
        #                                 dtype=torch.long)
        
        # return (tokens_tensor, segments_tensor, label_num_tensor, label_paragraph_tensor)
        return encoding, label_num_tensor, label_paragraph_tensor
    
    def __len__(self):
        return self.len
    
    # def collate_fn(self, samples):
    #     # TODO: implement collate_fn
    #     tokens_tensors = [s[0] for s in samples]
    #     segments_tensors = [s[1] for s in samples]
        
    #     # 測試集有 labels
    #     if samples[0][2] is not None:
    #         label_ids = torch.stack([s[2] for s in samples])
    #     else:
    #         label_ids = None
            
    #     # zero pad 到同一序列長度
    #     tokens_tensors = pad_sequence(tokens_tensors, 
    #                               batch_first=True)
    #     segments_tensors = pad_sequence(segments_tensors, 
    #                                 batch_first=True)
        
        
    #     # attention masks，將 tokens_tensors 裡頭不為 zero padding
    #     # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    #     masks_tensors = torch.zeros(tokens_tensors.shape, 
    #                             dtype=torch.long)
    #     masks_tensors = masks_tensors.masked_fill(
    #                             tokens_tensors != 0, 1)
    
    #     return tokens_tensors, segments_tensors, masks_tensors, label_ids
class ContextSelectionForTestDataset(Dataset):
     # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self , data: List[Dict] , context: List[Dict] , tokenizer):
        

        # 大數據你會需要用 iterator=True
        self.data = data
        self.len = len(self.data)
        
        self.context = context
        self.question = [ sub['question'] for sub in data ]
        self.paragraphs = [ sub['paragraphs'] for sub in data ]
       
        
        
        self.ID = [ sub['id'] for sub in data ]
        
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        
        eachquestion = self.question[idx]
        eachparagraphs = self.paragraphs[idx]
        
        Data_ID = self.ID[idx]
        
            
            
        df_eachparagraphs = pd.DataFrame()
        for i in range(len(eachparagraphs)):
            df_eachparagraphs.loc[i,'Paragraph'] = eachparagraphs[i]

        
            
        # 建立第一個問題的 BERT tokens 並加入分隔符號 [SEP]
        # word_pieces = ["[CLS]"]
        
        # tokens_question = self.tokenizer.tokenize(eachquestion)
        
        #tokens_question = eachquestion
        paragraphelen = int(512-len(eachquestion))-3
        # word_pieces += tokens_question + ["[SEP]"]
        # len_question = len(word_pieces)
        
        # 第一個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[0]]):
            tokens_p0 = self.context[eachparagraphs[0]][:paragraphelen]
        else:
            tokens_p0 = self.context[eachparagraphs[0]]
        # tokens_p0 = self.tokenizer.tokenize(self.context[eachparagraphs[0]])
        # word_pieces += tokens_p0 + ["[SEP]"]
        # len_p0 = len(word_pieces) - len_question
        
        # 第二個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[1]]):
            tokens_p1 = self.context[eachparagraphs[1]][:paragraphelen]
        else:
            tokens_p1 = self.context[eachparagraphs[1]]
        
        # tokens_p1 = self.tokenizer.tokenize(self.context[eachparagraphs[1]])
        # word_pieces += tokens_p1 + ["[SEP]"]
        # len_p1 = len(word_pieces) - len_question - len_p0
        
        # 第三個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[2]]):
            tokens_p2 = self.context[eachparagraphs[2]][:paragraphelen]
        else:
            tokens_p2 = self.context[eachparagraphs[2]]
        
        # tokens_p2 = self.tokenizer.tokenize(self.context[eachparagraphs[2]])
        # word_pieces += tokens_p2 + ["[SEP]"]
        # len_p2 = len(word_pieces) - len_question - len_p0 - len_p1
        
        # 第四個段落的 BERT tokens
        if paragraphelen < len(self.context[eachparagraphs[3]]):
            tokens_p3 = self.context[eachparagraphs[3]][:paragraphelen]
        else:
            tokens_p3 = self.context[eachparagraphs[3]]
        
        # tokens_p3 = self.tokenizer.tokenize(self.context[eachparagraphs[3]])
        # word_pieces += tokens_p3 + ["[SEP]"]
        # len_p3 = len(word_pieces) - len_question - len_p0 - len_p1 - len_p2
        
        # 將整個 token 序列轉換成索引序列
        # ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        # tokens_tensor = torch.tensor(ids)
        
        # encoding = self.tokenizer([eachquestion, eachquestion, eachquestion, eachquestion], [tokens_p0, tokens_p1, tokens_p2, tokens_p3], return_tensors = "pt", padding=True, max_length=512, truncation='longest_first')
        encoding = self.tokenizer([eachquestion, eachquestion, eachquestion, eachquestion], [tokens_p0, tokens_p1, tokens_p2, tokens_p3], return_tensors = "pt", padding=True)
        
        # 
        # segments_tensor = torch.tensor([0] * len_question + [1] *len_p0 + [2] *len_p1 + [3] *len_p2 + [4] *len_p3,  
        #                                 dtype=torch.long)
        
        # return (tokens_tensor, segments_tensor, label_num_tensor, label_paragraph_tensor)
        return encoding, df_eachparagraphs, Data_ID
    
    def __len__(self):
        return self.len
    
    
    
class IntentForTrainDataset(Dataset):
     # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self , data: List[Dict] , label2id: List[Dict] , tokenizer):
        
        
        # 大數據你會需要用 iterator=True
        self.data = data
        self.len = len(self.data)
        
        
        self.intent = [ sub['intent'] for sub in data ]
        
        self.text = [ sub['text'] for sub in data]
        
        self.id = [ sub['id'] for sub in data]
        
        self.label2id = label2id
        
        
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        
        
        eachtext = self.text[idx]
        eachintentid = torch.tensor([self.label2id[self.intent[idx]]]).cuda()
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        
        
        
        tokenized_examples = self.tokenizer(eachtext, return_tensors="pt")
        
        tokenized_examples['Intentid'] = eachintentid
        
        # print('tokenized_examples',tokenized_examples)
        
        # print('eachintentid',eachintentid)
                
        return tokenized_examples
    
    def collate_fn(self, samples):
        # print(samples)
        # TODO: implement collate_fn
        input_ids_tensors = [torch.flatten(s['input_ids']).cuda() for s in samples]
        token_type_ids_tensors = [torch.flatten(s['token_type_ids']).cuda() for s in samples]
        attention_mask_tensors = [torch.flatten(s['attention_mask']).cuda() for s in samples] 
        Intentid_tensors = [s['Intentid'] for s in samples]  
        
        # zero pad 到同一序列長度
        

        input_ids_tensors = pad_sequence(input_ids_tensors, 
                                  batch_first=True)
        token_type_ids_tensors = pad_sequence(token_type_ids_tensors, 
                                    batch_first=True, padding_value=1)
        attention_mask_tensors = pad_sequence(attention_mask_tensors, 
                                    batch_first=True)
        Intentid_tensors = pad_sequence(Intentid_tensors, 
                                    batch_first=True)
        
    
        return { 
        'input_ids': input_ids_tensors, 
        'token_type_ids': token_type_ids_tensors,
        'attention_mask': attention_mask_tensors,
        'Intentid':Intentid_tensors   
        }

        
    
    def __len__(self):
        return self.len
    
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
        

    
    
class SlotForTrainDataset(Dataset):
     # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self , data: List[Dict] , label2id: List[Dict] , tokenizer):
        
        
        # 大數據你會需要用 iterator=True
        self.data = data
        self.len = len(self.data)
        
        
        self.tokens = [ sub['tokens'] for sub in data ]
        
        self.tags = [ sub['tags'] for sub in data]
        
        self.id = [ sub['id'] for sub in data]
        
        self.label2id = label2id
        
        
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        
        
        eachtokens = self.tokens[idx]
        
        # neweachtokens=""
        # for i in eachtokens:
        #     neweachtokens += i
        #     neweachtokens += " "
        
        # print(neweachtokens)
        # time.sleep(2)
        
        tags = []
        for i in self.tags[idx]:
            tags.append(self.label2id[i])
            
        # eachtagsid = torch.LongTensor(tags).cuda()
        
        # 將 label 文字也轉換成索引方便轉換成 tensor
        
        
        
        tokenized_examples = self.tokenizer(text_target=eachtokens, add_special_tokens=False, return_tensors="pt", is_split_into_words=True )
        # print('tokenized_examples',tokenized_examples.word_ids())
        
        labels = []
        
        # print(tags)
        
        word_ids = tokenized_examples.word_ids()
        # print(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        # print(label_ids)
        # time.sleep(5)
        
        # print(word_ids)
            
            
        
        tokenized_examples['Tagsid'] = torch.LongTensor(label_ids).cuda()
        
        # print('tokenized_examples',tokenized_examples)
        
        # print('eachintentid',eachintentid)
                
        return tokenized_examples
    
    def collate_fn(self, samples):
        
        # TODO: implement collate_fn
        input_ids_tensors = [torch.flatten(s['input_ids']).cuda() for s in samples]
        token_type_ids_tensors = [torch.flatten(s['token_type_ids']).cuda() for s in samples]
        attention_mask_tensors = [torch.flatten(s['attention_mask']).cuda() for s in samples] 
        Tagsid_tensors = [s['Tagsid'] for s in samples]  
        
        # zero pad 到同一序列長度
        

        input_ids_tensors = pad_sequence(input_ids_tensors, 
                                  batch_first=True)
        token_type_ids_tensors = pad_sequence(token_type_ids_tensors, 
                                    batch_first=True, padding_value=1)
        attention_mask_tensors = pad_sequence(attention_mask_tensors, 
                                    batch_first=True)
        Tagsid_tensors = pad_sequence(Tagsid_tensors, 
                                    batch_first=True, padding_value=-100)
        
    
        return { 
        'input_ids': input_ids_tensors, 
        'token_type_ids': token_type_ids_tensors,
        'attention_mask': attention_mask_tensors,
        'Tagsid':Tagsid_tensors   
        }

        
    
    
    def __len__(self):
        return self.len
    
    
