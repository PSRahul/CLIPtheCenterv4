import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys

class_split_list=["base_classes","base_classes","novel_classes","novel_classes"]
split_list=["train","test","train","test"]

for i in range(len(class_split_list)):
    class_split = class_split_list[i]
    split = split_list[i]

    centernet_root = os.path.join(
        "/home/psrahul/MasterThesis/Experiments/sample_images/centernet/", class_split, split)
    clipthecenter_root = os.path.join(
        "/home/psrahul/MasterThesis/Experiments/sample_images/clipthecenter/", class_split, split)

    for image_name in os.listdir(centernet_root):
        centernet_img = Image.open(os.path.join(centernet_root, image_name))
        clipthecenter_img=Image.open(os.path.join(clipthecenter_root, image_name))
        centernet_img=np.asarray(centernet_img)
        clipthecenter_img=np.asarray(clipthecenter_img)


        fig, (ax1, ax2)=plt.subplots(1, 2)

        #fig.set_size_inches(19.20,10.80)
        #fig.set_size_inches(3.60, 3.6)
        plt.subplots_adjust(wspace=0, hspace=0)
        ax1.imshow(centernet_img)
        ax1.axis('off')
        ax2.imshow(clipthecenter_img)
        ax2.axis('off')
        os.makedirs(os.path.join("/home/psrahul/MasterThesis/Experiments/sample_images/combined_results/",class_split,split), exist_ok=True)
        plt.axis('off')
        plt.savefig(os.path.join(os.path.join("/home/psrahul/MasterThesis/Experiments/sample_images/combined_results/",class_split,split),
                                 str(image_name) ), bbox_inches='tight', dpi=600)
        plt.close("all")
