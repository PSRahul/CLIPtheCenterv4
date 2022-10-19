import os
annFile_root="/home/psrahul/MasterThesis/datasets/centernet/coco/PASCAL_15_5_1500/base_classes/train"
annFile = os.path.join(annFile_root,"coco/labels.json")
resFile = "/home/psrahul/MasterThesis/Experiments/PASCAL_15_5_1500_CA_clip_res_18_1510/2.json"
clip_embedding_root= "/home/psrahul/MasterThesis/datasets/BBoxGroundtruths/PASCAL_15_5/train/"
score_threshold= 0.5

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from evaluation.eval_clip_utils import get_class_embeddings,proces_dataset_class,get_groundtruths,visualise_bbox

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here

cocoGt=COCO(annFile)
cocoDt=cocoGt.loadRes(resFile)

gt_clip_embeddings,class_id_list,class_name_list=get_class_embeddings(clip_embedding_root=clip_embedding_root,dataset=cocoGt)
cocoDt=proces_dataset_class(cocoDt,gt_clip_embeddings,class_id_list,class_name_list)

class_name=[]
class_id=[]

groundtruth_matrix=get_groundtruths(dataset=cocoGt,is_gt=True)
detection_matrix=get_groundtruths(dataset=cocoDt)

detection_matrix=detection_matrix[detection_matrix[:,5]>score_threshold]
for index in cocoGt.getImgIds():
    visualise_bbox(
        annFile_root=annFile_root,dataset=cocoGt, id=index,
               gt=groundtruth_matrix,
               pred=detection_matrix,
               draw_gt=True,
               draw_pred=True,
               checkpoint_dir="/home/psrahul/MasterThesis/Experiments/test/")

print("Classes : No classes")
cocoEval = COCOeval(cocoGt, cocoDt, annType)
#cocoEval.params.iouThrs = [0.5]
cocoEval.params.useCats = 0
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


print("Classes : All classes")
cocoEval = COCOeval(cocoGt, cocoDt, annType)
#cocoEval.params.iouThrs = [0.5]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
for index in range(len(cocoGt.cats)):
    cat = cocoGt.cats[index]
    class_name.append(cat["name"])
    class_id.append(cat["id"])

for i,cat_id in enumerate(class_id,0):
    print("Classes : ",class_name[i])

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.catIds=cat_id
    #cocoEval.params.iouThrs = [0.5]

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()