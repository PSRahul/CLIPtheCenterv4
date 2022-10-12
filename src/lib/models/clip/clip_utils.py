import copy
def make_detections_valid(output_res, detections):
    detections_valid = copy.deepcopy(detections)
    detections_valid=detections_valid.reshape((detections_valid.shape[0]*detections_valid.shape[1],detections_valid.shape[2]))
    detections_valid[:,2]=detections_valid[:,2]-detections_valid[:,0]
    detections_valid[:,3] = detections_valid[:,  3] - detections_valid[:, 1]
    detections_valid[detections_valid[:, 0] <= 0, 0] = 0
    detections_valid[detections_valid[:, 1] <= 0, 1] = 0
    detections_valid[detections_valid[:, 2] <= 40, 2] = 40
    detections_valid[detections_valid[:, 3] <= 40, 3] = 40
    detections_valid[detections_valid[:, 2] >= output_res - 1, 2] = output_res - 1
    detections_valid[detections_valid[:, 3] >= output_res - 1, 3] = output_res - 1
    for i in range(detections.shape[0]):
        x = detections_valid[i, 0] + detections_valid[i, 2]
        y = detections_valid[i, 1] + detections_valid[i, 3]
        if (x > output_res - 1):
            detections_valid[i, 0] = output_res - 1 - detections_valid[i, 1]
        if (y > output_res - 1):
            detections_valid[i, 1] = output_res - 1 - detections_valid[i, 3]
    detections_valid[detections_valid[:, 0] <= 0, 0] = 0
    detections_valid[detections_valid[:, 1] <= 0, 1] = 0
    detections_valid = detections_valid.reshape(detections.shape)
    return detections_valid
