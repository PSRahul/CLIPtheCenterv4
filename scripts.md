## PASCAL_15_5 Without CLIP

 CUDA_VISIBLE_DEVICES=3 python main.py ctdet --exp_id PASCAL_15_5_res_101_1210 --arch res_101 --batch_size 64  --lr 2.5e-4 --gpus 3 --num_workers 32 --num_epochs 800 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15

python test.py ctdet --exp_id coco_resdcn18 --arch res_18 --keep_res --resume --num_epochs 400

## PASCAL_15_5 Without CLIP Local Test

python test.py ctdet --exp_id PASCAL_15_5_res_50 --arch res_50 --keep_res --resume --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15 --test_split train --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_15_5_res_50/model_best.pth
python test.py ctdet --exp_id PASCAL_15_5_res_50 --arch res_50 --keep_res --resume --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15 --test_split val --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_15_5_res_50/model_best.pth
python test.py ctdet --exp_id PASCAL_15_5_res_50 --arch res_50 --keep_res --resume --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15 --test_split test --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_15_5_res_50/model_best.pth

## PASCAL_3_2_10 With CLIP Local Test

ctdet --exp_id PASCAL_3_2_10_CA_res_18_cliptest_4 --arch res_18 --batch_size 4 --lr 0.1314e-4 --gpus 0 --num_workers 8 --num_epochs 200 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_3_2_10_CA/base_classes/ --num_classes 1 --clip_encoder --clip_topk 100

ctdet --exp_id PASCAL_3_2_10_CA_res_18_cliptest_4 --arch res_18 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_3_2_10_CA/base_classes/ --num_classes 1 --test_split train --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_3_2_10_CA_res_18_cliptest_4/model_last.pth --nms --clip_encoder --clip_topk 100

## PASCAL_15_5 With CLIP

 CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ctdet --exp_id PASCAL_15_5_CA_res_50_1510_clip --arch res_50 --batch_size 128  --lr 5e-4 --gpus 0,1,2,3 --num_workers 100 --num_epochs 100 --data_root_dir /home/psrahul/MasterThesis/datasets/centernet/coco/PASCAL_15_5_CA/base_classes/ --num_classes 1 --clip_encoder --clip_topk 100

python test.py ctdet --exp_id coco_resdcn18 --arch res_18 --keep_res --resume --num_epochs 400