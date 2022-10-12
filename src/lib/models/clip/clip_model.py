import torch.nn as nn
import torch
from torchinfo import summary
import clip
from PIL import Image
import os
import torchvision.transforms as T
from models.clip.clip_utils import make_detections_valid

class CLIPModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt =opt
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device="cuda")
        self.clip_model = self.clip_model.cuda().eval()

    def forward(self, batch, dets):

        dets=make_detections_valid(self.opt.output_res,dets)

        clip_encodings = torch.zeros((dets.shape[0], dets.shape[1],512))
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

                    image_clip_embedding = self.clip_model.encode_image(image_cropped_clip.cuda())

                    clip_encodings[batch_index,topk_index,:] = image_clip_embedding

        clip_encodings /= clip_encodings.norm(dim=-1, keepdim=True)
        return clip_encodings

    def print_details(self):
        pass
