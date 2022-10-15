import torch.nn as nn
import torch
from torchinfo import summary
from torchvision.models import ResNet18_Weights,ResNet50_Weights


class Embedder(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt=opt

        """
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet18", weights=ResNet18_Weights.DEFAULT
        )
        self.model.fc=nn.Flatten()
        """
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet50", weights=ResNet50_Weights.DEFAULT
        )
        
        self.model.fc = nn.Linear(2048, 512)

        self.model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)


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

                #heatmap_index=heatmap_output[batch_index,0,upper:lower,left:right]
                #heatmap_index=heatmap_index.unsqueeze(0).unsqueeze(0)
                #upsampled_heatmap_index=torch.nn.functional.interpolate(heatmap_index,(self.opt.output_res, self.opt.output_res),mode="bilinear")[0]

                mask_topk_index[0,upper:lower,left:right]=1
                masked_heatmap[batch_index,topk_index,:,:,:]=heatmap_batch_index*mask_topk_index
        masked_heatmap=masked_heatmap.reshape((dets.shape[0]*dets.shape[1], 1, self.opt.output_res, self.opt.output_res))
        model_encoding= self.model(masked_heatmap)

        model_encoding_norm = model_encoding/ model_encoding.norm(dim=-1, keepdim=True)
        return model_encoding_norm

    def print_details(self):
        summary(self.model, input_size=(400, 1, 128, 128))
