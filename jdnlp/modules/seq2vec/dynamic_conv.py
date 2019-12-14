"""
Sourced from the following but with heavy modifications.
https://github.com/kuldeep7688/dynamic_convolutional_neural_network
"""

from typing import *

import torch
import torch.nn as nn
import math

from overrides import overrides
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed

# from fairseq.modules.dynamic_convolution import DynamicConv1dTBC

import torchsnooper
import logging
logger = logging.getLogger(__name__)

class DCNNCell(nn.Module):
    def __init__(
        self,
        cell_number=1,
        sent_length=7,
        conv_kernel_size=(3, 1),
        conv_input_channels=1,
        conv_output_channels=2,
        conv_stride=(1, 1),
        k_max_number=5,
        folding_kernel_size=(1, 2),
        folding_stride=(1,1)
    ):
        super().__init__()
        self.cell_number=cell_number 
        self.sent_length=sent_length
        self.conv_kernel_size=conv_kernel_size
        self.conv_input_channels=conv_input_channels
        self.conv_output_channels=conv_output_channels
        self.conv_stride=conv_stride
        self.k_max_number=k_max_number
        self.folding_kernel_size=folding_kernel_size
        self.folding_stride=folding_stride
        
        # calculating padding size
        self.pad_0_direction = math.ceil(self.conv_kernel_size[0]  - 1)
        self.pad_1_direction = math.ceil(self.conv_kernel_size[1] - 1)
        
        # 2d convolution
        self.conv_layer = nn.Conv2d(
            in_channels=self.conv_input_channels,
            out_channels=self.conv_output_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=(self.pad_0_direction, self.pad_1_direction)
        )
        
        # if cell is last then initialising folding
        if cell_number == -1:
            self.fold = nn.AvgPool2d(kernel_size=self.folding_kernel_size, stride=self.folding_stride)
            
    def forward(self, inp):
        
        # [batch_size, input_channels, sent_length_in, embedding_dim]
        conved = self.conv_layer(inp)
        
        # [batch_size, out_channels, sent_length_out, embedding_dim]
        if self.cell_number == -1:
            conved = self.fold(conved)
        
        # [batch_size, out_channels, sent_length, embedding_dim/2]
        k_maxed = torch.tanh(torch.topk(conved, min(self.k_max_number, conved.size(-2)), dim=2, largest=True)[0])
        
        # [batch_size, out_channels, k_maxed_number, embedding_dim/2]
        return k_maxed
    
 
@Seq2SeqEncoder.register("dynamic_conv")
class DCNN(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cells: List = None,
        dropout_rate: int = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        
        # """
        self.cells = nn.Sequential(*(DCNNCell(**c_params) for c_params in cells))
        
        num_averages = sum([c['folding_stride'][-1] for c in cells if c.get('cell_number') == -1])
        
        self.fc_layer_input = cells[-1]['k_max_number'] *\
            cells[-1]['conv_output_channels'] *\
            math.ceil((input_dim-cells[-1]['folding_stride'][0])/cells[-1]['folding_stride'][-1])
            
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.fc_layer_input, output_dim)
        # """
        #self.dcn = DynamicConv1dTBC(input_dim)

    def _forward(self, embedded, *args, **kwargs):
        return self.dcn(embedded.permute(1,0,2).contiguous(), unfold=True).permute(1, 0, 2).contiguous()
    
    @overrides
    def forward(self, embedded, *args, **kwargs): # [batch_size, sent_length, embedding_dim]
        embedded = embedded.unsqueeze(1)
        # [batch_size, 1(initial_input_channel), sent_length, embedding_dim]
        out = self.cells(embedded)
        # out = self.dcnn_first_cell(embedded) # [batch, c1_output_channels, c1_k_maxed_number, emb_dim]
        # out = self.dcnn_last_cell(out) # [batch, c2_output_channels, c2_k_maxed_number, emb_dim/2]
        out = out.view(out.size(0), -1)
        out = self.dropout(out) # [batch, c2_output_channels * c2_k_maxed_number * emb_dim/2]
        out = self.fc(out)
        return out
    
    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim
    
    @overrides
    def is_bidirectional(self) -> int:
        return False

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim
    
   