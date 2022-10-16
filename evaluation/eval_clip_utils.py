from tqdm import tqdm
import os.path
import numpy as np
import torch
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



