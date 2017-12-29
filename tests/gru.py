'''
' @file      gru.py
' @author    zhangshu(shu.zhang@intel.com)
' @date      2017-12-18 10:18:13
' @brief
'''
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import json
import numpy as np
import sys


def tensor_to_string(x):
    tmp = x.data.numpy()
    tmpfile="../tmp/tmp"
    np.savetxt(tmpfile, tmp.reshape(tmp.size), fmt="%.6f")
    with open(tmpfile, "r") as f:
        data = f.read()
    return data.replace('\n', ' ').strip()

def gru_model(T, N, D, H, nl, bid):
    print("seq_length = %d, batch_size = %d, input_size = %d, " \
          "hidden_size = %d, num_layer = %d, bidirectional = %s"
            % (T, N, D, H, nl, bid))
    nd = 2 if bid else 1
    gru_model = nn.GRU(D, H, nl, bias = True, bidirectional = bid)
    input = Variable(torch.randn(T, N, D))
    target = Variable(torch.randn(T, N, nd*H))
    h0 = Variable(torch.randn(nl * nd, N, H))
    output, hn = gru_model(input, h0)
    # get inputs and params
    param_list = []
    for param in gru_model.parameters():
        # weights layout is different with c code, so need transpose
        if len(param.data.shape) == 2:
            y = torch.transpose(param,0,1)
            param_list.append(tensor_to_string(y))
        else:
            param_list.append(tensor_to_string(param))
    weights = ' '.join(param_list) 

    outdict = { 
        "num_layer": str(nl),
        "num_direction": str(nd),
        "seq_length": str(T),
        "batch_size": str(N),
        "input_size": str(D),
        "hidden_size": str(H),
        "weights": weights, 
        "x": tensor_to_string(input), 
        "hx": tensor_to_string(h0), 
        "y": tensor_to_string(output),
        "hy": tensor_to_string(hn)
    }
    outstr = json.dumps(outdict, indent=4) 
    print(outstr, file=open("../tmp/gru_forward.json", "w"))
    tmp = weights.split(' ')

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Error args")
        sys.exit(-1)
    _, T, N, I, H, nl, bidirectional = sys.argv
    bid = bidirectional == str(True)
    gru_model(int(T), int(N), int(I), int(H), int(nl), bid) 
