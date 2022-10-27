from tqdm import tqdm
import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import sys
from PIL import Image

from datetime import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import cv2
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from yaml.loader import SafeLoader

import copy
def generate_clip_embedding(clip_embedding_root,class_id_list,class_name_list):
    clip_embeddings = np.zeros((len(class_id_list), 512))

    for idx,class_name in enumerate(class_name_list):
        class_embedding=np.load(os.path.join(clip_embedding_root,class_name+".npy"))
        clip_embeddings[idx,:]=class_embedding

    return clip_embeddings

def get_class_embeddings(dataset,clip_embedding_root):
    class_name_list = []
    class_id_list = []
    for index in tqdm(range(len(dataset.cats))):
        cat = dataset.cats[index]
        class_id_list.append(cat["id"])
        class_name_list.append(cat["name"])
    clip_embedding = generate_clip_embedding(clip_embedding_root,  class_id_list, class_name_list)
    return clip_embedding,class_id_list,class_name_list

def assign_classes(clip_encodings, predictions):
    embeddings = torch.tensor(predictions)
    clip_encodings = torch.tensor(clip_encodings)
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    clip_encodings /= clip_encodings.norm(dim=-1, keepdim=True)
    scores = torch.matmul(embeddings.float(), clip_encodings.T.float())
    top_probs, top_labels = scores.cpu().topk(1, dim=-1)
    return top_labels

def proces_dataset_class(dataset,gt_clip_embedding,class_id_list,class_name_list):
    print()
    for ann_index in dataset.anns:
        ann=dataset.anns[ann_index]
        predicted_clip_encoding=ann["clip_encoding"]
        top_labels=assign_classes(gt_clip_embedding,predicted_clip_encoding)
        dataset.anns[ann_index]["category_id"]=class_id_list[top_labels]
    return dataset

def proces_dataset_class(dataset,gt_clip_embedding,class_id_list,class_name_list):
    print()
    for ann_index in dataset.anns:
        ann=dataset.anns[ann_index]
        predicted_clip_encoding=ann["clip_encoding"]
        top_labels=assign_classes(gt_clip_embedding,predicted_clip_encoding)
        dataset.anns[ann_index]["category_id"]=class_id_list[top_labels]
    return dataset


def get_groundtruths(dataset, is_gt=False):
    print(dataset.getImgIds())
    gt = np.empty((0, 7))
    for index in dataset.getImgIds():
        #index=5
        #image_id = dataset.ids[index]
        ann_id=dataset.getAnnIds(imgIds=index)
        image, anns = dataset.loadImgs(index),dataset.loadAnns(ann_id)
        image = np.array(image)
        bounding_box_list = []
        class_list = []
        scores_list= []
        for ann in anns:
            bounding_box_list.append(ann['bbox'])
            class_list.append(ann['category_id'])
            if not is_gt:
                scores_list.append(ann['score'])

        image_id = np.array(index)
        bounding_box_list = np.array(bounding_box_list)
        image_id_list = np.ones((len(class_list), 1)) * image_id
        if (is_gt):
            scores_list = np.ones((len(class_list), 1))

        class_list = np.array(class_list).reshape((len(class_list), 1))
        scores_list = np.array(scores_list).reshape((len(scores_list), 1))
        # ["image_id", "bbox_y", "bbox_x", "w", "h", "score", "class_label"]
        if (len(bounding_box_list != 0)):
            gt_idx = np.hstack((image_id_list, bounding_box_list, scores_list, class_list))
            gt = np.vstack((gt, gt_idx))
    return gt

def visualise_bbox(annFile_root,dataset, id, gt=None, pred=None, draw_gt=True, draw_pred=True,
                  checkpoint_dir=None):
    img = dataset.loadImgs(id)[0]
    img = img["file_name"]
    img = os.path.join(annFile_root, "coco", "data", img)
    #print(img)
    img = Image.open(img)
    image = np.asarray(img)

    fig, ax = plt.subplots()
    ax.imshow(image)
    #plt.show()

    if draw_pred:
        predictions_image = pred[pred[:, 0] == int(id)]

        for i in range(predictions_image.shape[0]):
            bbox_i = predictions_image[i, 1:5]
            rect = patches.Rectangle(
                (bbox_i[0], bbox_i[1]), bbox_i[2],
                bbox_i[3], linewidth=2, edgecolor='r',
                facecolor='none')
            ax.add_patch(rect)
        #plt.show()
    if draw_gt:
        gt_image = gt[gt[:, 0] == int(id)]

        for i in range(gt_image.shape[0]):
            bbox_i = gt_image[i, 1:5]
            rect = patches.Rectangle(
                (bbox_i[0], bbox_i[1]), bbox_i[2],
                bbox_i[3], linewidth=2, edgecolor='b',
                facecolor='none')
            ax.add_patch(rect)
    #plt.show()
    #print(id," | Number of Predictions | ", predictions_image.shape[0]," | Number of GroundTruth Objects | ", gt_image.shape[0])
    #plt.show()
    os.makedirs(os.path.join(checkpoint_dir), exist_ok=True)
    plt.axis('off')
    plt.savefig(os.path.join(checkpoint_dir,str(id)+".png"),bbox_inches='tight')
    plt.close("all")

def filter_dataset_score(dataset,score_threshold):
    print()
    ann_ids=dataset.getAnnIds()
    for ann_index in ann_ids:
        ann=dataset.anns[ann_index]
        img = dataset.loadImgs(ann["image_id"])

        if(ann["score"]<score_threshold):
            del dataset.anns[ann_index]

    return dataset