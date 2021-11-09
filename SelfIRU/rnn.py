import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from .infinity import *
from fairseq.modules import MultiheadAttention


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last,
                    rnn_type='GREAT', args=None):
        super().__init__()
        self.rnns = []
        self.num_units = num_units
        self.rnn_type = rnn_type
        self.args = args
        _input_size = -1
        self.pre_extra_layers = self.args.extra.split('_')

        if('SA' in self.args.extra):
            # print("Using MHSA before")
            self.self_attn = MultiheadAttention(
                            input_size, self.args.num_heads,
                            dropout=dropout)

        if('CONV' in self.args.extra):
            kernel_size = self.args.ksize
            dilation_size=1
            padding=(kernel_size-1) * dilation_size
            dilation=dilation_size
            print(kernel_size)
            self.conv = TemporalBlock(input_size, input_size, kernel_size, stride=1,
                                dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size,
                                dropout=dropout)


        for i in range(nlayers):
            # print(i)
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
                if(rnn_type=='INFINITY' and bidir):
                    input_size_ = int(input_size_ // 2)

            if(rnn_type=='GRU'):
                self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
            elif(rnn_type=='LSTM'):
                self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True, bias=True))
            elif(rnn_type=='INFINITY'):
                print("===============")
                print(input_size_)
                print(output_size_)
                self.rnns.append(InfinityRNN(input_size_, output_size_, 1, bidirectional=bidir, args=args))
            # self.rnns.append(LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
            # self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        # self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.rnn_type = rnn_type
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last
        # self.reset_parameters()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if('LSTM' not in self.rnn_type):
            return [Variable(weight.new(1, bsz, self.num_units).zero_()) for l in range(self.nlayers)]
        else:
            return [(Variable(weight.new(1, bsz, self.num_units).zero_()),
                     Variable(weight.new(1, bsz, self.num_units).zero_()))
                    for l in range(self.nlayers)]

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.01)
                    # torch.nn.init.orthogonal_(p.data)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        hidden = self.init_hidden(bsz)
        return hidden[i]

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        if(len(self.pre_extra_layers)>0):
            for extra in self.pre_extra_layers:
                if(extra=='SA'):
                    # print("using SA")
                    input = torch.transpose(input, 1, 0)
                    G = self.self_attn(query=input, key=input, value=input,
                                    attn_mask=self.buffered_future_mask(input))
                    try:
                        input, _ = G
                    except:
                        input, _ = G
                    input = torch.transpose(input, 0, 1)
                if(extra=='CONV'):
                    input = self.conv(torch.transpose(input, 2, 1))
                    input = torch.transpose(input, 2, 1)

                    # print("Using CONV")

        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        # if(self.rnn_type=='INFINITY'):
        #     output = torch.transpose(output, 0, 1)

        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            # output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            pre_output = output
            output, hidden = self.rnns[i](output, hidden)
            if('RES' in self.pre_extra_layers):
                if(output.size(-1)==pre_output.size(-1)):
                    # print("Adding residual")
                    output += pre_output
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        # if(self.rnn_type=='INFINITY'):
        #     output = torch.transpose(output, 0, 1)
        return outputs[-1]
