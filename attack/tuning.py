import numpy as np
from yolox_model import YoloxModel
import pandas as pd
from tqdm import tqdm
from Inference import detect_sign
from fast_attacks import fgsm, it_fgsm
from typing import Union, Callable
from pathlib import Path
from itertools import product
from math import ceil

def obtain_statistics(df):
    sucess_df = df.query('benign_prediction != non_benign_prediction')
    sucess_precentage = sucess_df.shape[0] / df.shape[0]
    benign_score_mean = df['benign_score'].mean()
    non_benign_score_mean = df['non_benign_score'].mean()
    sucess_score_mean = sucess_df[sucess_df['non_benign_score']
                                  != None]['non_benign_score'].mean()
    return sucess_precentage, sucess_score_mean, benign_score_mean, non_benign_score_mean

def get_model_results(imgs, onnx_model, ratio):
    predictions = []
    scores = []
    for img in imgs:
        result = detect_sign(img, onnx_model, ratio)
        if result is not None:
            predictions.append(result[0])
            scores.append(result[1])
        else:
            predictions.append(None)
            scores.append(None)
    return predictions, scores

#TODO find a better way and better names for my sanity

def tune_hyperparameters(
    fun: Callable,
    model: Union[str, Path],
    imgs: np.ndarray,
    original_imgs_df:pd.DataFrame,
    ratio: float,
    cuda:bool=False):
    obj_threshold_values = np.arange(0.05, 1, 0.05)
    cls_threshold_values = np.arange(0.5, 1, 0.05)

    if cuda:
        model = model.cuda()
    df = original_imgs_df.copy()

    statistics = {
        "obj_threshold": [],
        "cls_threshold": [],
        "success_rate": [],
        "success_score_mean": [],
        "benign_score_mean": [],
        "non_benign_score_mean": []
    }

    for obj_threshold, cls_threshold in product(obj_threshold_values,
                                                cls_threshold_values):
        torch_model = YoloxModel(
            model,
            obj_threshold=obj_threshold,
            cls_threshold=cls_threshold,
        )
        perturbed_imgs = fun(torch_model, imgs, cuda=cuda)
        predictions, scores = get_model_results(perturbed_imgs, model, ratio)
        df['non_benign_prediction'] = predictions
        df['non_benign_score'] = scores

        s = obtain_statistics(df)
        statistics['obj_threshold'].append(obj_threshold)
        statistics['cls_threshold'].append(cls_threshold)
        statistics['success_rate'].append(s[0])
        statistics['success_score_mean'].append(s[0])
        statistics['benign_score_mean'].append(s[2])
        statistics['non_benign_score_mean'].append(s[3])
    return statistics




def run_fgsm(
    torch_model,
    imgs:np.ndarray,
    cuda:bool=False):
    
    n = len(imgs)
    perturbed_imgs = []
    for i in tqdm(range(n)):
        img = np.asarray(imgs[i])
        if cuda:
            perturbed_img = fgsm(
                torch_model, img, eps=4, cuda=True).cpu().numpy()
        else:
            perturbed_img = fgsm(torch_model, img, eps=4).numpy()
        perturbed_imgs.append(perturbed_img)
    return perturbed_imgs