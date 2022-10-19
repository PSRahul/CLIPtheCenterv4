annFile = "/home/psrahul/MasterThesis/datasets/centernet/coco/PASCAL_15_5_1500/novel_classes/test/coco/labels.json"
resFile = "/home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_15_5_1500_CA_clip_res_18_1510/results.json"
clip_embedding_root= "/home/psrahul/MasterThesis/datasets/BBoxGroundtruths/PASCAL_15_5/train/"


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

groundtruth_matrix=get_groundtruths(dataset=cocoGt)
detection_matrix=get_groundtruths(dataset=cocoDt)
visualise_bbox(cfg=cfg, dataset=dataset, id=id,
               gt=gt,
               pred=prediction_with_nms_resized,
               draw_gt=True,
               draw_pred=True,
               resize_image_to_output_shape=False,
               checkpoint_dir=checkpoint_dir)

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