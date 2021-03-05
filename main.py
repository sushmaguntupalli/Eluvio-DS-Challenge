'''
Created by  - Sushma G.
Created on - 03-03-2021
'''

## Import the required libraries
import os
import sys
import glob
import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import average_precision_score


## Defined function for calculating the Average Precision (AP)
def calc_ap(gt_dict, pr_dict):
    """Average Precision (AP) for scene transitions.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
    Returns:
        AP, mean AP, and a dict of AP for each movie.
    """
    assert gt_dict.keys() == pr_dict.keys()

    AP_dict = dict()
    gt = list()
    pr = list()
    for imdb_id in gt_dict.keys():
        AP_dict[imdb_id] = average_precision_score(gt_dict[imdb_id], pr_dict[imdb_id])
        gt.append(gt_dict[imdb_id])
        pr.append(pr_dict[imdb_id])

    mAP = sum(AP_dict.values()) / len(AP_dict)

    gt = np.concatenate(gt)
    pr = np.concatenate(pr)
    AP = average_precision_score(gt, pr)

    return AP, mAP, AP_dict

## Defined function for calculating the Maximum IoU (Miou) for scene segmentation
def calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict, threshold=0.5):
    """Maximum IoU (Miou) for scene segmentation.
    Miou measures how well the predicted scenes and ground-truth scenes overlap. The descriptions can be found in
    https://arxiv.org/pdf/1510.08893.pdf. Note the length of intersection or union is measured by the number of frames.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
        shot_to_end_frame_dict: End frame index for each shot.
        threshold: A threshold to filter the predictions.
    Returns:
        Mean MIoU, and a dict of MIoU for each movie.
    """
    def iou(x, y):
        s0, e0 = x
        s1, e1 = y
        smin, smax = (s0, s1) if s1 > s0 else (s1, s0)
        emin, emax = (e0, e1) if e1 > e0 else (e1, e0)
        return (emin - smax + 1) / (emax - smin + 1)

    def scene_frame_ranges(scene_transitions, shot_to_end_frame):
        end_shots = np.where(scene_transitions)[0]
        scenes = np.zeros((len(end_shots) + 1, 2), dtype=end_shots.dtype)
        scenes[:-1, 1] = shot_to_end_frame[end_shots]
        scenes[-1, 1] = shot_to_end_frame[len(scene_transitions)]
        scenes[1:, 0] = scenes[:-1, 1] + 1
        return scenes

    def miou(gt_array, pr_array, shot_to_end_frame):
        gt_scenes = scene_frame_ranges(gt_array, shot_to_end_frame)
        pr_scenes = scene_frame_ranges(pr_array >= threshold, shot_to_end_frame)
        assert gt_scenes[-1, -1] == pr_scenes[-1, -1]

        m = gt_scenes.shape[0]
        n = pr_scenes.shape[0]

        # IoU for (gt_scene, pr_scene) pairs
        iou_table = np.zeros((m, n))

        j = 0
        for i in range(m):
            # j start prior to i end
            while pr_scenes[j, 0] <= gt_scenes[i, 1]:
                iou_table[i, j] = iou(gt_scenes[i], pr_scenes[j])
                if j < n - 1:
                    j += 1
                else:
                    break
            # j end prior to (i + 1) start
            if pr_scenes[j, 1] < gt_scenes[i, 1] + 1:
                break
            # j start later than (i + 1) start
            if pr_scenes[j, 0] > gt_scenes[i, 1] + 1:
                j -= 1
        assert np.isnan(iou_table).sum() == 0
        assert iou_table.min() >= 0

        # Miou
        return (iou_table.max(axis=0).mean() + iou_table.max(axis=1).mean()) / 2

    assert gt_dict.keys() == pr_dict.keys()

    miou_dict = dict()

    for imdb_id in gt_dict.keys():
        miou_dict[imdb_id] = miou(gt_dict[imdb_id], pr_dict[imdb_id], shot_to_end_frame_dict[imdb_id])
    mean_miou = sum(miou_dict.values()) / len(miou_dict)

    return mean_miou, miou_dict

## Defined function for calculating Precision, Recall and F1 for scene transitions at a given threshold
def calc_precision_recall(gt_dict, pr_dict, threshold=0.5):
    """Precision, Recall and F1 for scene transitions at a given threshold.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
        threshold: A threshold to filter the predictions.
    Returns:
        Mean Precision, Recall, and F1, per IMDB ID Precisions, Recalls, and F1 scores.
    """
    def precision_recall(gt_array, pr_array):
        tp_fn = gt_array == 1
        tp_fp = pr_array >= threshold

        tps = (tp_fn & tp_fp).sum()

        precision = tps / tp_fp.sum()
        recall = tps / tp_fn.sum()

        return np.nan_to_num(precision), np.nan_to_num(recall)

    assert gt_dict.keys() == pr_dict.keys()

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()

    for imdb_id in gt_dict.keys():
        p, r = precision_recall(gt_dict[imdb_id], pr_dict[imdb_id])
        precision_dict[imdb_id] = p
        recall_dict[imdb_id] = r
        fscore_dict[imdb_id] = 2 * p * r / (p + r)

    n = len(gt_dict)
    mean_precision = sum(precision_dict.values()) / n
    mean_recall = sum(recall_dict.values()) / n
    mean_fscore = sum(fscore_dict.values()) / n

    return mean_precision, mean_recall, mean_fscore, precision_dict, recall_dict, fscore_dict


if __name__ == '__main__':
    ground_truth = {}
    prediction = {}
    shot_end_frame = {}
    scores = {}
    filename = '/content/drive/MyDrive/Eluvio/data/data'
    print("Total Files : ", len(os.listdir('/content/drive/MyDrive/Eluvio/data/data')))
    print("\n")
    if not os.path.exists('/content/drive/MyDrive/Eluvio/saved_outputs'):
        os.makedirs('/content/drive/MyDrive/Eluvio/saved_outputs')

    output_location = '/content/drive/MyDrive/Eluvio/saved_outputs'
    # Loading the pickle files from the data folder
    for file in os.listdir('/content/drive/MyDrive/Eluvio/data/data'):
        # Loading pickle
        x = pickle.load(open(os.path.join(filename,file), "rb"))
        # Extract ground truth, prediction and shot end frame
        ground_truth[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"].numpy()
        prediction[x["imdb_id"]] = x["scene_transition_boundary_prediction"].numpy()
        shot_end_frame[x["imdb_id"]] = x["shot_end_frame"].numpy()
        # Generate Input dataframe
        place = pd.DataFrame(x["place"]).astype("float")
        cast = pd.DataFrame(x["cast"]).astype("float")
        action = pd.DataFrame(x["action"]).astype("float")
        audio = pd.DataFrame(x["audio"]).astype("float")
        scene_transition_boundary_ground_truth = pd.DataFrame(x["scene_transition_boundary_ground_truth"]).astype("float")
        # Merge the dataframe
        mergedDF = pd.concat([place, 
                        cast, 
                        action, 
                        audio, 
                        scene_transition_boundary_ground_truth],axis=1)
        mergedDF = mergedDF.head(-1)
        X = mergedDF.iloc[ : , :-1].values
        Y = mergedDF.iloc[: , -1].values
        # fit the xgboost model
        model = XGBRegressor(objective ='reg:squarederror', 
                              colsample_bytree = 0.3, 
                              learning_rate = 0.1, 
                              max_depth = 5, 
                              n_estimators = 50)
        model.fit(X,Y)
        Y_pred = model.predict(X)
        Y_pred = pd.DataFrame(Y_pred)
        prediction[x["imdb_id"]] = Y_pred
        # Save output
        Y_pred.to_csv(os.path.join(output_location,(file+".csv")),sep=',',index = False)
        scores["AP"], scores["mAP"], _ = calc_ap(ground_truth, prediction)
        scores["Miou"], _ = calc_miou(ground_truth, prediction, shot_end_frame)
        result = pd.json_normalize(scores).T
        result.columns = [str(file).strip('.pkl')]
        print(result.T)