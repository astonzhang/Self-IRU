import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNBase, LSTMCell
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import math
import numpy as np
import csv
from .forget_mult import *

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    categorical_dim = logits.size(-1)
    latent_dim = 1
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = Variable(torch.LongTensor(ind).transpose(0, 1))
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

class InfinityRNNLayer(nn.Module):

    """
    An implementation of the Self-IRU layer
    """

    def __init__(self, input_size, hidden_size, bias=False, dropout=0.0,
                    bidirectional=True, batch_first=True, gumbel=-1,
                    depth=1, base_model='linear', args=None):
        super(InfinityRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.depth = depth
        self.prevX = None
        self.max_depth = args.max_depth
        print("InfinityRNN layer depth={}".format(depth))
        print("bidirectional={}".format(bidirectional))
        self.base_model = args.base_model
        print("Base model={}".format(self.base_model))
        self.viz = []

        if(args is not None):
            self.max_depth = args.max_depth
            print("Max_depth={}".format(self.max_depth))
        # self.base_model = base_model
        rnn_size = hidden_size
        if(bidirectional):
            rnn_size = int(rnn_size //2)

        self.bidirectional = bidirectional

        if(depth>self.max_depth):
            if(self.base_model=='linear'):
                self.proj = nn.Linear(input_size, hidden_size * 3, bias=bias)
            elif(self.base_model=='LSTM'):
                self.proj = nn.LSTM(input_size, rnn_size * 3, dropout=dropout, bidirectional=bidirectional)
            elif(self.base_model=='GRU'):
                self.proj = nn.GRU(input_size, rnn_size * 3, dropout=dropout, bidirectional=bidirectional)
        else:
            self.ornn = InfinityRNNLayer(input_size, hidden_size, bias=bias,
                                    dropout=dropout, depth=depth+1, args=args, bidirectional=bidirectional)

            self.frnn = InfinityRNNLayer(input_size, hidden_size, bias=bias,
                                    dropout=dropout, depth=depth+1, args=args, bidirectional=bidirectional)
            # self.proj = nn.Linear(input_size, hidden_size * 3)
            if(self.base_model=='linear'):
                self.proj = nn.Linear(input_size, hidden_size * 3, bias=bias)
            elif(self.base_model=='LSTM'):
                self.proj = nn.LSTM(input_size, rnn_size * 3, dropout=dropout, bidirectional=bidirectional)
            elif(self.base_model=='GRU'):
                self.proj = nn.GRU(input_size, rnn_size * 3, dropout=dropout, bidirectional=bidirectional)

        self.T = nn.Linear(input_size, 2)

        # self.reset_parameters()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, bsz, self.hidden_size).zero_())

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):

        # std = 1.0 / math.sqrt(self.hidden_size)
        std = 0.1
        for w in self.parameters():
             w.data.uniform_(-std, std)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        # needed for truncated BPTT, called at every batch forward pass
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def forward(self, X, hidden=None, input_lengths=None):
        """ inputs should be sq x bsz x d
        """

        def write_to_log(out, fp):
            with open(fp,'w+') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(out)

        # print(input_lengths)
        seq_len, batch_size, _ = X.size()
        # input_lengths = torch.from_numpy(np.array(input_lengths).cpu()).to('cuda:0')

        # print("Infinity running..")
        cancel_prev = False
        terminate = self.T(X)   # bsz x seq_len x 1
        # terminate = torch.sum(terminate, dim=0)
        # terminate = torch.nn.Parameter(torch.ones(2)).cuda()
        terminate = torch.sigmoid(terminate)
        # print(terminate.size())
        # self.viz.append(terminate)
        t1, t2 = torch.chunk(terminate, 2, dim=-1)
        # print("===============================================")
        #print('LEFT tree | depth={}'.format(self.depth))
        _t1 = t1.view((batch_size,-1))

        #print(t1.view((-1)))
        #print(_t1[0].view(-1))
        #print(_t1[1].view(-1))
        # print(_t1.max(-1)[0])
        # print(_t1.min(-1))
        #print(_t1.mean())
        write_to_log(_t1[0].cpu().detach().numpy().tolist(), 'tmp_seq_left{}.txt'.format(self.depth))

        # print("===============================================")
        #print('RIGHT tree | depth={}'.format(self.depth))
        _t2 = t2.view((batch_size,-1))
        #print(_t2.mean())
        write_to_log(_t2[0].cpu().detach().numpy().tolist(), 'tmp_seq_right{}.txt'.format(self.depth))
        # print(t2.view((batch_size, -1)))
        if(self.depth>self.max_depth):
            if(self.base_model=='linear'):
                V = self.proj(X)
            else:
                self.proj.flatten_parameters()
                V, _ = self.proj(X)
            V = V.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = torch.chunk(V, 3, dim=-1)
        else:
            F2, _ = self.frnn(X, hidden, input_lengths=input_lengths)
            O2, _ = self.ornn(X, hidden, input_lengths=input_lengths)
            if(self.base_model=='linear'):
                Z = self.proj(X)
            else:
                self.proj.flatten_parameters()
                Z, _ = self.proj(X)
            Z = Z.view(seq_len, batch_size, 3 * self.hidden_size)
            # print(self.hidden_size)
            if(self.bidirectional):
                # Z1, Z2 = torch.chunk(Z, 2, dim=-1)
                # print('Calling bidir')
                Z = Z.view(seq_len, batch_size, 2, 3 * (self.hidden_size//2))
                Z, F, O = torch.chunk(Z, 3, dim=-1)
                Z=Z.contiguous()
                O=O.contiguous()
                F=F.contiguous()
                Z = Z.view(seq_len, batch_size, self.hidden_size)
                O = O.view(seq_len, batch_size, self.hidden_size)
                F = F.view(seq_len, batch_size, self.hidden_size)
            else:
                Z, F, O = torch.chunk(Z, 3, dim=-1)
                Z = Z.view(seq_len, batch_size, self.hidden_size)

            F = (F2 * t1) + ((1-t1) * F)
            O = (O2 * t2) + ((1-t2) * O)
        Z = torch.relu(Z)
        F = torch.sigmoid(F)

        Z=Z.contiguous()
        F=F.contiguous()
        # C = F * Z
        C = ForgetMult()(F, Z, hidden_init=None, use_cuda=True)
        H = torch.sigmoid(O) * C

        if(input_lengths is not None):
            input_lengths = [x-1 for x in input_lengths]
            input_lengths = torch.tensor(input_lengths, requires_grad=False, dtype=torch.long).cuda()
            bsz = C.size(1)
            count = torch.arange(bsz)
            C = H[input_lengths,count,:]
            #print(C.size())
            C = C.unsqueeze(0)
        else:
            C = torch.max(H, 0, keepdim=True)[0]
        # C = torch.mean(C,0,keepdim=True)
        if(H.size(-1)==X.size(-1)):
            H += X
        return H, C

class InfinityRNNDecoder(nn.Module):

    """
    An implementation of the Self-IRU decoder
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                    bidirectional=False, batch_first=True, gumbel=-1,
                    depth=1, base_model='linear', args=None):
        super(InfinityRNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.depth = depth
        self.prevX = None
        self.max_depth = 3
        print("InfinityRNN layer depth={}".format(depth))
        if(args is not None):
            self.max_depth = args.max_depth
            print("Max_depth={}".format(self.max_depth))
        self.base_model = base_model
        self.T = nn.Linear(input_size, 2)

        if(depth>self.max_depth):
            if(base_model=='linear'):
                self.proj = nn.Linear(input_size, hidden_size * 3, bias=bias)
            elif(base_model=='LSTM'):
                self.proj = nn.LSTMCell(input_size, hidden_size * 3)
        else:
            self.ornn = InfinityRNNDecoder(input_size, hidden_size, bias=bias,
                                    dropout=dropout, depth=depth+1, args=args)

            self.frnn = InfinityRNNDecoder(input_size, hidden_size, bias=bias,
                                    dropout=dropout, depth=depth+1, args=args)
            # self.proj = nn.Linear(input_size, hidden_size * 3)
            if(base_model=='linear'):
                self.proj = nn.Linear(input_size, hidden_size * 3, bias=bias)
            elif(base_model=='LSTM'):
                self.proj = nn.LSTMCell(input_size, hidden_size * 3)

        # self.reset_parameters()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, bsz, self.hidden_size).zero_())

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):

        # std = 1.0 / math.sqrt(self.hidden_size)
        std = 0.1
        for w in self.parameters():
             w.data.uniform_(-std, std)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        # needed for truncated BPTT, called at every batch forward pass
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, X, hidden, input_lengths=None):
        """ inputs should be sq x d
        """

        # hidden = hidden[0]
        batch_size, _ = X.size()

        terminate = self.T(X)   # bsz x seq_len x 1
        # terminate = torch.sum(terminate, dim=0)
        terminate = torch.sigmoid(terminate)
        t1, t2 = torch.chunk(terminate, 2, dim=-1)

        cancel_prev = False
        if(self.depth>self.max_depth):
            if(self.base_model=='linear'):
                V = self.proj(X)
            elif(self.base_model=='LSTM'):
                V, _ = self.proj(X)
            V = V.view(batch_size, 3 * self.hidden_size)
            Z, F, O = torch.chunk(V, 3, dim=-1)
        else:
            F2, _ = self.frnn(X, hidden)
            O2, _ = self.ornn(X, hidden)
            if(self.base_model=='linear'):
                Z = self.proj(X)
            elif(self.base_model=='LSTM'):
                Z, _ = self.proj(X)
            Z.view(batch_size, 3 * self.hidden_size)
            Z, F, O = torch.chunk(Z, 3, dim=-1)
            F += (t1 * F2)
            O += (t2 * O2)
            Z = Z.view(batch_size, self.hidden_size)

        Z = torch.tanh(Z)
        F = torch.sigmoid(F)

        Z=Z.contiguous()
        F=F.contiguous()
        if(type(hidden) is tuple):
            hidden = hidden[0]
        C = (F * Z) + ((1-F) * hidden)
        # C = ForgetMult()(F, Z, hidden, use_cuda=True)
        H = torch.sigmoid(O) * C
        # H = torch.nn.functional.dropout(H, p=self.dropout, training=self.training, inplace=False)
        return H, C[-1]

class InfinityRNN(torch.nn.Module):
    r"""Applies a Self-IRU cell to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        num_layers: The number of Self-IRU layers to produce.
        layers: List of preconstructed Self-IRU layers to use for the Self-IRU module (optional).
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the Self-IRU values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs Self-IRU-fo (applying an output gate to the output). If False, performs Self-IRU-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (layers, batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (layers, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, args=None, **kwargs):
        #assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
        #assert batch_first == False, 'Batch first mode is not yet supported'
        #assert bias == True, 'Removing underlying bias is not yet supported'

        super(InfinityRNN, self).__init__()
        print("bidirectional={}".format(bidirectional))
        if(bidirectional):
            hidden_size_2 = hidden_size
        else:
            hidden_size_2 = hidden_size
        self.layers = torch.nn.ModuleList(layers if layers else [InfinityRNNLayer(input_size if l == 0 else hidden_size_2, hidden_size,
                                        args=args, bidirectional=bidirectional, **kwargs) for l in range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        print(self.layers)

    def reset(self):
        r'''If your convolutional window is greater than 1, you must reset at the beginning of each new sequence'''
        [layer.reset() for layer in self.layers]

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, bsz, self.hidden_size).zero_())

    def forward(self, input, hidden=None):
        next_hidden = []
        input = input.transpose(0, 1)
        # print("=================================")
        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])
            # print(input.size())
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])
        # next_hidden = torch.cat(next_hidden, 0)
        input = input.transpose(0,1)
        return input, next_hidden
