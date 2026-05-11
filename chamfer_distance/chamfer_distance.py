import os
import torch
from torch.utils.cpp_extension import load

cd = load(
    name="cd",
    sources=[
        "chamfer_distance/chamfer_distance.cpp",
        "chamfer_distance/chamfer_distance.cu",
    ],
)

import argparse
import yaml

parser = argparse.ArgumentParser(description="PyTorch Semantic-Line Training")
# arguments from command line
parser.add_argument("--config", default="./config.yml", help="path to config file")
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.safe_load(open(args.config))

if CONFIGS["DATA"]["PLATFORM"] == "Colab":

    class ChamferDistanceFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xyz1, xyz2):
            batchsize, n, _ = xyz1.size()
            _, m, _ = xyz2.size()
            xyz1 = xyz1.contiguous()
            xyz2 = xyz2.contiguous()
            dist1 = torch.zeros(batchsize, n)
            dist2 = torch.zeros(batchsize, m)

            idx1 = torch.zeros(batchsize, n, dtype=torch.int)
            idx2 = torch.zeros(batchsize, m, dtype=torch.int)

            if not xyz1.is_cuda:
                cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
            else:
                dist1 = dist1.cuda()
                dist2 = dist2.cuda()
                idx1 = idx1.cuda()
                idx2 = idx2.cuda()
                cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

            ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

            return dist1, dist2

        @staticmethod
        def backward(ctx, graddist1, graddist2):
            xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

            graddist1 = graddist1.contiguous()
            graddist2 = graddist2.contiguous()

            gradxyz1 = torch.zeros(xyz1.size())
            gradxyz2 = torch.zeros(xyz2.size())

            if not graddist1.is_cuda:
                cd.backward(
                    xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
                )
            else:
                gradxyz1 = gradxyz1.cuda()
                gradxyz2 = gradxyz2.cuda()
                cd.backward_cuda(
                    xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
                )

            return gradxyz1, gradxyz2

    class ChamferDistance(torch.nn.Module):
        def forward(self, xyz1, xyz2):
            return ChamferDistanceFunction.apply(xyz1, xyz2)

elif CONFIGS["DATA"]["PLATFORM"] == "CPU":

    class ChamferDistance(torch.nn.Module):
        def forward(self, xyz1, xyz2):
            """
            xyz1: [B, N, 3]
            xyz2: [B, M, 3]
            """

            # [B, N, M, 3]
            diff = xyz1.unsqueeze(2) - xyz2.unsqueeze(1)

            # [B, N, M]
            dist = (diff**2).sum(-1)

            # minimalne odległości
            dist1, _ = dist.min(dim=2)  # dla xyz1
            dist2, _ = dist.min(dim=1)  # dla xyz2

            return dist1, dist2
