## PASCAL_15_5 Without CLIP

 CUDA_VISIBLE_DEVICES=3 python main.py ctdet --exp_id PASCAL_15_5_res_101_1210 --arch res_101 --batch_size 64  --lr 2.5e-4 --gpus 3 --num_workers 32 --num_epochs 800 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15

python test.py ctdet --exp_id coco_resdcn18 --arch res_18 --keep_res --resume --num_epochs 400

## PASCAL_15_5 Without CLIP Local Test

python test.py ctdet --exp_id PASCAL_15_5_res_50 --arch res_50 --keep_res --resume --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15 --test_split train --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_15_5_res_50/model_best.pth
python test.py ctdet --exp_id PASCAL_15_5_res_50 --arch res_50 --keep_res --resume --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15 --test_split val --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_15_5_res_50/model_best.pth
python test.py ctdet --exp_id PASCAL_15_5_res_50 --arch res_50 --keep_res --resume --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_15_5/base_classes/ --num_classes 15 --test_split test --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_15_5_res_50/model_best.pth

## PASCAL_3_2_10 With CLIP Local Test

ctdet --exp_id PASCAL_3_2_10_CA_res_18_cliptest_6 --arch res_18 --batch_size 4 --lr 0.1314e-4 --gpus 0 --num_workers 4 --num_epochs 200 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_3_2_10_CA/base_classes/ --num_classes 1 --clip_encoder --clip_topk 25 --K 25

ctdet --exp_id PASCAL_2012_ZETA_CA_res_18_cliptest --arch res_18 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_2012_ZETA_CA/base_classes/ --num_classes 1 --test_split train --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_2012_ZETA_CA_res_18_cliptest/model_last.pth --nms

## PASCAL_2012_ZETA With CLIP Local Test


ctdet --exp_id PASCAL_2012_ZETA_res_18_cliptest --arch res_18 --batch_size 3 --lr 1e-4 --gpus 0 --num_workers 8 --num_epochs 10000 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_2012_ZETA/base_classes/ --num_classes 1 --clip_encoder --clip_topk 1 --lr_step 5000 --lr 0.15625e-4 --no_color_aug --flip 0 --not_rand_crop

ctdet --exp_id PASCAL_2012_ZETA_CA_res_18_cliptest --arch res_18 --batch_size 3 --lr 1e-4 --gpus 0 --num_workers 8 --num_epochs 200 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_2012_ZETA_CA/base_classes/ --num_classes 1 

ctdet --exp_id PASCAL_2012_ZETA_CA_res_18_cliptest --arch res_18 --data_root_dir /home/psrahul/MasterThesis/datasets/PASCAL_2012_ZETA_CA/base_classes/ --num_classes 1 --test_split train --load_model /home/psrahul/MasterThesis/repo/Phase7/CenterCLIP_Outputs/exp/ctdet/PASCAL_2012_ZETA_CA_res_18_cliptest/model_last.pth --nms 

## PASCAL_15_5 With CLIP

 CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ctdet --exp_id PASCAL_15_5_CA_res_18_1510_clip --arch res_18 --batch_size 256  --lr 1e-3 --gpus 0,1,2,3 --num_epochs 100 --lr_step 45,60 --data_root_dir /home/psrahul/MasterThesis/datasets/centernet/coco/PASCAL_15_5_CA/base_classes/ --num_classes 1 --clip_encoder --clip_topk 25 --num_workers 96 

 CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ctdet --exp_id PASCAL_15_5_CA_res_18_1510_clip --arch res_18 --batch_size 512  --lr 1e-3 --gpus 0,1,2,3 --num_epochs 100 --lr_step 45,60 --data_root_dir /home/psrahul/MasterThesis/datasets/centernet/coco/PASCAL_15_5_CA/base_classes/ --num_classes 1 --clip_encoder --clip_topk 25 --num_workers 96

python test.py ctdet --exp_id coco_resdcn18 --arch res_18 --keep_res --resume --num_epochs 400


## PASCAL_15_5_1500 Without CLIP

PASCAL_15_5_1500_res_18_1510

 CUDA_VISIBLE_DEVICES=3 python main.py ctdet --exp_id PASCAL_15_5_1500_res_18_1510 --arch res_18 --batch_size 128  --lr 5e-4 --gpus 3 --num_workers 64 --data_root_dir /home/psrahul/MasterThesis/datasets/centernet/coco/PASCAL_15_5_1500/base_classes/ --num_classes 15 --K 25 --save_all --num_epochs 200

ctdet --exp_id PASCAL_15_5_1500_res_18_1510 --arch res_18 --data_root_dir /home/psrahul/MasterThesis/datasets/centernet/coco/PASCAL_15_5_1500/base_classes/ --num_classes 15   --K 25 --nms  --load_model /home/psrahul/MasterThesis/repo/Phase7/Server_Outputs/PASCAL_15_5_1500_res_18_1510/model_50.pth --test_split train
