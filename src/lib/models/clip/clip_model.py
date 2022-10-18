import torch.nn as nn
import torch
from torchinfo import summary
import clip
from PIL import Image
import os
import torchvision.transforms as T
from models.clip.clip_utils import make_detections_valid
from torchvision import transforms

class CLIPModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt =opt
        self.clip_model =torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.clip_model.fc = nn.Flatten()
        self.clip_preprocess = transforms.Compose([
                                        transforms.Resize(224),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        self.clip_model = self.clip_model.cuda().eval()

    def forward(self, batch, dets):
        self.clip_model = self.clip_model.cuda().eval()
        dets=make_detections_valid(self.opt.output_res,dets)

        dets = dets.reshape((batch["input"].shape[0] , self.opt.clip_topk, dets.shape[1]))
        clip_encodings = torch.zeros((dets.shape[0],dets.shape[1], 2048),device="cuda")
        for batch_index in range(dets.shape[0]):
            image=batch["input"][batch_index,:,:,:]
            transform = T.ToPILImage()
            image = transform(image)

            for topk_index in range(dets.shape[1]):
                    bbox = dets[batch_index,topk_index,0:4]
                    (left, upper, right, lower) = (
                        int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    image_cropped = image.crop((left, upper, right, lower))

                    image_cropped_clip = self.clip_preprocess(image_cropped).unsqueeze(0)

                    image_clip_embedding = self.clip_model(image_cropped_clip.cuda())

                    clip_encodings[batch_index,topk_index,:] = image_clip_embedding

        clip_encodings=clip_encodings.reshape((batch["input"].shape[0] * self.opt.clip_topk,2048))
        clip_encodings /= clip_encodings.norm(dim=-1, keepdim=True)

        dets = dets.reshape((dets.shape[0]*dets.shape[1], dets.shape[2]))
        return clip_encodings,dets

    def print_details(self):
        pass
