annFile = "/home/psrahul/MasterThesis/datasets/PASCAL_3_2_10_train/base_classes/train/coco/labels.json"
resFile = "/home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_3_2_10_res_18/results.json"


import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here

cocoGt=COCO(annFile)
cocoDt=cocoGt.loadRes(resFile)

class_name=[]
class_id=[]
print("Classes : All classes")
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.iouThrs = [0.5]
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
    cocoEval.params.iouThrs = [0.5]

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()