from tqdm import tqdm
import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import sys
from datetime import datetime
from post_process.coco_evaluation import calculate_coco_result
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import cv2
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from yaml.loader import SafeLoader
from post_process.torchmetric_evaluation import calculate_torchmetrics_mAP
from post_process.nms import perform_nms
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



def get_groundtruths(dataset, show_image=False):
    print(dataset.getImgIds())
    gt = np.empty((0, 7))
    for index in dataset.getImgIds():
        #image_id = dataset.ids[index]
        image, anns = dataset.loadImgs(index),dataset.loadAnns(index)
        image = np.array(image)
        bounding_box_list = []
        class_list = []
        for ann in anns:
            bounding_box_list.append(ann['bbox'])
            class_list.append(ann['category_id'])

        if (show_image):
            bbox = bounding_box_list[0]
            bbox = [int(x) for x in bbox]
            print(bbox)
            image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
            plt.imshow(image)
            plt.show()
            break
        image_id = np.array(index)
        bounding_box_list = np.array(bounding_box_list)
        image_id_list = np.ones((len(class_list), 1)) * image_id
        scores_list = np.ones((len(class_list), 1))
        class_list = np.array(class_list).reshape((len(class_list), 1))
        # ["image_id", "bbox_y", "bbox_x", "w", "h", "score", "class_label"]
        if (len(bounding_box_list != 0)):
            gt_idx = np.hstack((image_id_list, bounding_box_list, scores_list, class_list))
            gt = np.vstack((gt, gt_idx))
    return gt

def visualise_bbox(cfg, dataset, id, gt=None, pred=None, draw_gt=True, draw_pred=True,
                   resize_image_to_output_shape=False,checkpoint_dir=None):
    image = dataset._load_image(id)

    image = np.array(image)
    if (resize_image_to_output_shape):
        image = cv2.resize(image,
                           (cfg["post_processing"]["model_output_shape"], cfg["post_processing"]["model_output_shape"]))

    fig, ax = plt.subplots()
    ax.imshow(image)

    if draw_pred:
        predictions_image = pred[pred[:, 0] == int(id)]

        for i in range(predictions_image.shape[0]):
            bbox_i = predictions_image[i, 1:5]
            rect = patches.Rectangle(
                (bbox_i[0], bbox_i[1]), bbox_i[2],
                bbox_i[3], linewidth=2, edgecolor='r',
                facecolor='none')
            ax.add_patch(rect)

    if draw_gt:
        gt_image = gt[gt[:, 0] == int(id)]

        for i in range(gt_image.shape[0]):
            bbox_i = gt_image[i, 1:5]
            rect = patches.Rectangle(
                (bbox_i[0], bbox_i[1]), bbox_i[2],
                bbox_i[3], linewidth=2, edgecolor='b',
                facecolor='none')
            ax.add_patch(rect)
    print(id," | Number of Predictions | ", predictions_image.shape[0]," | Number of GroundTruth Objects | ", gt_image.shape[0])
    #plt.show()
    os.makedirs(os.path.join(checkpoint_dir,"images"), exist_ok=True)
    plt.savefig(os.path.join(checkpoint_dir,"images",str(id)+".png"))
