import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

import argparse
import yaml

parser = argparse.ArgumentParser(description="PyTorch Semantic-Line Training")
# arguments from command line
parser.add_argument("--config", default="./config.yml", help="path to config file")
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.safe_load(open(args.config))

if CONFIGS["DATA"]["PLATFORM"] == "Colab":
    import deep_hough as dh

    class C_dht_Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, feat, numangle, numrho):
            N, C, _, _ = feat.size()
            out = torch.zeros(N, C, numangle, numrho).type_as(feat).cuda()
            out = dh.forward(feat, out, numangle, numrho)
            outputs = out[0]
            ctx.save_for_backward(feat)
            ctx.numangle = numangle
            ctx.numrho = numrho
            return outputs

        @staticmethod
        def backward(ctx, grad_output):
            feat = ctx.saved_tensors[0]
            numangle = ctx.numangle
            numrho = ctx.numrho
            out = torch.zeros_like(feat).type_as(feat).cuda()
            out = dh.backward(grad_output.contiguous(), out, feat, numangle, numrho)
            grad_in = out[0]
            return grad_in, None, None

    class C_dht(torch.nn.Module):
        def __init__(self, numAngle, numRho):
            super(C_dht, self).__init__()
            self.numAngle = numAngle
            self.numRho = numRho

        def forward(self, feat):
            return C_dht_Function.apply(feat, self.numAngle, self.numRho)

elif CONFIGS["DATA"]["PLATFORM"] == "CPU":
    import torch
    from . import dht_cpu as dh

    class C_dht(torch.nn.Module):
        def __init__(self, numAngle, numRho):
            super(C_dht, self).__init__()
            self.numAngle = numAngle
            self.numRho = numRho

        def forward(self, feat):
            return dh.forward(feat, self.numAngle, self.numRho)[0]
