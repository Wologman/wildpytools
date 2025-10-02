from sklearn.metrics import (accuracy_score,
                             f1_score,
                             balanced_accuracy_score,
                             average_precision_score,
                             roc_auc_score)
from .data import remove_rare_classes
import numpy as np


class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def get_map_score(target_df, pred_df, average='macro'):
    target_df, pred_df = remove_rare_classes(target_df, pred_df, 1)
    col_sums = target_df.sum()
    mask = col_sums >= 1 #keeping this in to avoid division by 0
    targs_arr = target_df.loc[:,mask].copy().values
    preds_arr = pred_df.loc[:,mask].copy().values
    if average is None:
        scores_vals = average_precision_score(targs_arr,preds_arr, average=None)
        if isinstance(scores_vals, float): #handle the situation where only one species is present
            scores_vals = [scores_vals]  
        scores_keys = target_df.columns[mask].tolist()
        scores_dict = {k:v for (k,v) in zip(scores_keys, scores_vals)}
    else:
        scores_dict = {'mean': average_precision_score(targs_arr,preds_arr, average=average)}   
    return scores_dict


def get_macro_auc_score(target_df, pred_df, average='macro'):
    target_df, pred_df = remove_rare_classes(target_df, pred_df, 1)
    col_sums = target_df.sum()
    mask = col_sums >= 1  # avoid division by 0 if class never appears
    targs_arr = target_df.loc[:, mask].copy().values
    preds_arr = pred_df.loc[:, mask].copy().values
    
    if average is None:
        # per-class AUCs
        scores_vals = roc_auc_score(targs_arr, preds_arr, average=None)
        if isinstance(scores_vals, float):  # handle situation where only one species is present
            scores_vals = [scores_vals]
        scores_keys = target_df.columns[mask].tolist()
        scores_dict = {k: v for (k, v) in zip(scores_keys, scores_vals)}
    else:
        # averaged AUC (macro, micro, weighted)
        scores_dict = {'mean': roc_auc_score(targs_arr, preds_arr, average=average)}
    return scores_dict


def get_separation_score(df_target, df_pred, average=None, min_samples=5):
    df_target, df_pred = remove_rare_classes(df_target, df_pred, min_samples)
    col_names = list(df_target.columns)
    targ_vals = df_target.values
    pred_vals = df_pred.values
    tp_scores = (pred_vals * targ_vals)
    tn_scores = (pred_vals * (1-targ_vals))
    
    targ_sums = targ_vals.sum(axis=1)
    negative_means = tn_scores.sum(axis=1) / (len(df_target) - targ_sums)
    positive_means = tp_scores.sum(axis=1) / targ_sums
    difference = positive_means - negative_means
    scores_dict = {col_names[idx]: difference[idx] for idx in range(len(col_names))}
    flat_tp_scores = tp_scores.flatten()
    flat_tn_scores = tn_scores.flatten()
    mean_tp_scores = flat_tp_scores[flat_tp_scores != 0].mean()
    mean_tn_scores= flat_tn_scores[flat_tn_scores != 0].mean()
    print(Colour.S + 'The mean prediction value for true positives is: ' + Colour.E + f'{mean_tp_scores:.2}')
    print(Colour.S + 'The mean prediction value for true negatives is: ' + Colour.E + f'{mean_tn_scores:.2}')
    return scores_dict


def get_ba_score(targets_df, predictions_df):
    targets_df, predictions_df = remove_rare_classes(targets_df, predictions_df, 1)
    target_arr = np.argmax(targets_df.values, axis=1)
    preds_arr = np.argmax(predictions_df.values, axis=1)
    return balanced_accuracy_score(target_arr, preds_arr)


def get_accuracy_score(targets_df, preds_df, normalize=True):
    targets_df, preds_df = remove_rare_classes(targets_df, preds_df, 1)
    target_arr = np.argmax(targets_df.values, axis=1)
    preds_arr = np.argmax(preds_df.values, axis=1)
    return accuracy_score(target_arr,  preds_arr, normalize=normalize, sample_weight=None)


def get_f1_score(targets_df, predictions_df):
    targets_df, predictions_df = remove_rare_classes(targets_df, predictions_df, 1)
    target_arr = np.argmax(targets_df.values, axis=1)
    preds_arr = np.argmax(predictions_df.values, axis=1)
    return f1_score(target_arr, preds_arr, pos_label=1, average='macro', zero_division=np.nan)


def get_metrics(targets, predictions, min_freq=5):
    targets, predictions = remove_rare_classes(targets, predictions, min_freq)
    balanced_accuracy = get_ba_score(targets, predictions)
    overall_accuracy = get_accuracy_score(targets, predictions)
    map_scores = get_map_score(targets, predictions)['mean']  #for some reason, this can only run once
    f1_score = get_f1_score(targets, predictions)

    print(Colour.S + f'Overall accuracy score for classes with more than {min_freq} samples is: ' + Colour.E + f'{overall_accuracy:.3f}')
    print(Colour.S + f'Balanced accuracy score for classes with more than {min_freq} samples is: ' + Colour.E + f'{balanced_accuracy:.3f}')
    print(Colour.S + f'macro Average Precision Score for classes with more than {min_freq} samples is: ' + Colour.E + f'{map_scores:.3f}')
    print(Colour.S + f'macro F1 Score for classes with more than {min_freq} samples is: ' + Colour.E + f'{f1_score:.3f}')
    return overall_accuracy, balanced_accuracy, map_scores, f1_score