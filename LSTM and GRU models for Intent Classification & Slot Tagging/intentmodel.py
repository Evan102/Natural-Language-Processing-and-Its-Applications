from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifierLSTM(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifierLSTM, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)       
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        self.BatchNor = torch.nn.BatchNorm1d(300,affine=False)
        
        

        # TODO: model architecture
        
        self.lstm1 = torch.nn.LSTM(
            input_size = self.embed.weight.shape[1],
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.num_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional

        )
        
       
        self.ReLuFun= torch.nn.ReLU(inplace=False) 
        

        self.linear1 = torch.nn.Linear(in_features = int(2*self.hidden_size*1.5) , out_features = int(1.5*self.hidden_size))


        self.linear2 = torch.nn.Linear(in_features = int(hidden_size), out_features = num_class)

        self.linear2_1 = torch.nn.Linear(in_features = int(2*self.hidden_size) , out_features = num_class)





    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.num_class
        #raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
     

        h1 = torch.zeros(2*self.num_layers, len(batch), self.hidden_size, device=batch.device , requires_grad=True)
        c1 = torch.zeros(2*self.num_layers, len(batch), self.hidden_size, device=batch.device  , requires_grad=True)
        h2 = torch.zeros(2*self.num_layers, len(batch), int(2*self.hidden_size*1.5), device=batch.device  , requires_grad=True)
        c2 = torch.zeros(2*self.num_layers, len(batch), int(2*self.hidden_size*1.5), device=batch.device  , requires_grad=True)

        x = self.embed(batch) 
      

        x = x.permute(0, 2, 1)
        x = self.BatchNor(x)
        x = x.permute(0, 2, 1)

        out1, (h1, c1) = self.lstm1(x, (h1, c1))

        out1_1 = self.ReLuFun(out1)

        
        out2, (h2, c2) = self.lstm2(out1_1, (h2, c2))


        out2_1 = self.ReLuFun(h2[-1])


        out3 = self.linear1(h2[-1])  # First dim of Hn is num_layers, which is set to 1 above.

        out3_1 = self.ReLuFun(out3)

        out3_2 = self.linear1_1(out3_1)

        out3_3 = self.ReLuFun(out3_2)

        out4 = self.linear2(out3_3)

        return out4



class SeqTaggerLSTM(SeqClassifierLSTM):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        h1 = torch.zeros(2*self.num_layers, len(batch), self.hidden_size, device=batch.device , requires_grad=True)
        c1 = torch.zeros(2*self.num_layers, len(batch), self.hidden_size, device=batch.device  , requires_grad=True)
       

        
        x = self.embed(batch) 


        x = x.permute(0, 2, 1)
        x = self.BatchNor(x)
        x = x.permute(0, 2, 1)

        out1, (h1, c1) = self.lstm1(x, (h1, c1))


        out1_1 = self.ReLuFun(out1)


        out4 = self.linear2_1(out1_1)

        return out4

        #raise NotImplementedError


class SeqClassifierGRU(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifierGRU, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)       
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        self.BatchNor = torch.nn.BatchNorm1d(300)
        
     


        # TODO: model architecture
        # self.num_layers = 2
        #print(';;;', self.embed.weight.shape[1])
        self.GRU1 = torch.nn.GRU(
            input_size = self.embed.weight.shape[1],
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.num_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional

        )

        self.dropFun = torch.nn.Dropout(0)
        
        self.ReLuFun = torch.nn.ReLU(inplace=False) 
        
        self.BatchNor1 = torch.nn.BatchNorm1d(int(0.5*self.hidden_size))
        
        

        


    
        self.GRU2 = torch.nn.GRU(
            input_size = 2*self.hidden_size,
            hidden_size = int(self.hidden_size*0.5) ,
            batch_first = True,
            num_layers = 1,
            dropout = 0,
            bidirectional = self.bidirectional
        )
             


        self.linear2 = torch.nn.Linear(in_features = int(hidden_size*0.5), out_features = num_class)

       




    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.num_class
        #raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
       

        h1 = torch.zeros(2*self.num_layers, len(batch), self.hidden_size, device=batch.device , requires_grad=True)
        
        h1_1 = torch.zeros(2, len(batch), int(self.hidden_size*0.5), device=batch.device  , requires_grad=True)
    

        
        x = self.embed(batch) 
  

        x = x.permute(0, 2, 1)
        x = self.BatchNor(x)
        x = x.permute(0, 2, 1)
        


        out1, h1 = self.GRU1(x, h1)
        
      
        
        out1 = self.dropFun(out1)
        
        out1 = self.ReLuFun(out1)
        
        

        out1, h1_1 = self.GRU2(out1, h1_1)
        

        
        out1_1  = self.BatchNor1(h1_1[-1]) # Use_Output_To_predict
        
        
        out4 = self.linear2(out1_1)
        
        return out4


class SeqTaggerGRU(SeqClassifierGRU):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        h1 = torch.zeros(2*self.num_layers, len(batch), self.hidden_size, device=batch.device , requires_grad=True)
      


        x = self.embed(batch) 


        x = x.permute(0, 2, 1)
        x = self.BatchNor(x)
        x = x.permute(0, 2, 1)

        out1, h1 = self.GRU1(x, h1)


        out1_1 = self.ReLuFun(out1)


        out4 = self.linear2_1(out1_1)
  
        return out4

        #raise NotImplementedError

