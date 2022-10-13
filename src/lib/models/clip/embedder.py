import torch.nn as nn
import torch
from torchinfo import summary


class Embedder(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt=opt
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(3))
        layers.append(nn.MaxPool2d(kernel_size=16,
                                   stride=4,
                                   ))

        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,

            ))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(3))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(2187, 512))

        self.model = nn.Sequential(*layers)
        #self.print_details()

    def forward(self, heatmap_output,dets):
        dets = dets.reshape((heatmap_output.shape[0], self.opt.clip_topk, dets.shape[1]))
        masked_heatmap = torch.zeros((dets.shape[0],dets.shape[1], 1, self.opt.output_res, self.opt.output_res),device="cuda")

        for batch_index in range(dets.shape[0]):
            heatmap_batch_index = heatmap_output[batch_index]

            for topk_index in range(dets.shape[1]):
                mask_topk_index=torch.zeros_like(heatmap_batch_index)
                bbox=dets[batch_index,topk_index,0:4]
                (left, upper, right, lower) = (
                    int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                mask_topk_index[:,upper:lower,left:right]=1
                masked_heatmap[batch_index,topk_index,:,:,:]=heatmap_batch_index*mask_topk_index
        masked_heatmap=masked_heatmap.reshape((dets.shape[0]*dets.shape[1], 1, self.opt.output_res, self.opt.output_res))
        return self.model(masked_heatmap)

    def print_details(self):
        summary(self.model, input_size=(400, 1, 128, 128))
