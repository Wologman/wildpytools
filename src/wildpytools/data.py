import pandas as pd
from pathlib import Path
import re
import json
from typing import Optional, Union, Literal


def remove_rare_classes(df_true: pd.DataFrame,
                        df_pred: pd.DataFrame,
                        rare_threshold: int = 10,
                        verbose: bool = False
                        ):
    """A helper function to remove rare classes prior to calculating metrics
       or plotting. 

    Args:
        df_true (pd.DataFrame): One hot encoded dataframe of targets
        df_pred (pd.DataFrame): Binary predictions
        rare_threshold (int, optional): Minimum number of samples. Defaults to 10.

    Returns:
        New dataframes where all classes have at least rare_threshold instances
    """

    col_sums = df_true.sum(axis=0)
    original_width = df_true.shape[1]
    mask = col_sums >= rare_threshold

    remove_list=[]
    total_removed = 0
    for column_name, col_sum in col_sums.items():
        if col_sum < rare_threshold:
            remove_list.append((column_name,col_sum))
            total_removed +=col_sum
    
    remove_cols = original_width - mask.sum()
    df_true = df_true.loc[:, mask]
    df_pred = df_pred.loc[:, mask]  

    rows_to_remove = df_true.index[df_true.sum(axis=1) == 0]
    df_true = df_true.drop(rows_to_remove)
    df_pred = df_pred.drop(rows_to_remove)

    rows_with_targets = df_true.any(axis=1)
    df_true = df_true[rows_with_targets]
    df_pred = df_pred[rows_with_targets]

    if verbose and remove_cols > 0:
        print(f'Removing {remove_cols} classes as they have less than {rare_threshold} samples')
        print(f'Also removing {len(rows_to_remove)} rows as they were from those classes')
    return df_true, df_pred


def convert_to_multiclass(df_true: pd.DataFrame,
                          df_pred: pd.DataFrame,
                          ):
    """Converts labels and predictions dataframe from a multilabel classifier into
       a pseudo-multiclass output, by dropping any rows with more than one target

    Args:
        target_df (pd.DataFrame): One hot encoded dataframe of targets
        pred_df (pd.DataFrame): Binary predictions

    Returns:
        _type_: _description_
    """
    mask_single = df_true.sum(axis=1) == 1
    df_true_single = df_true[mask_single].copy()
    df_pred_single = df_pred[mask_single].copy()
    df_targets = df_true_single.idxmax(axis=1).to_frame(name='Targets').reset_index(drop=True)
    df_predictions = df_pred_single.reset_index(drop=True)
    df_combined = pd.concat([df_targets.reset_index(drop=True), df_predictions.reset_index(drop=True)], axis=1)
    return df_combined


def rename_predictions(rename_dict: dict,
                       df: pd.DataFrame, 
                       merge_by: Literal["mean", "max"] = 'max'
                       ):
    """Rename a labels dataframe where the labels are column names using a dictionary
       Allows for a many to one relationship between keys and values, with the merging
       logic specified by the merge_by argument

    Args:
        rename_dict (dict): {old_name: new_name}
        df (pd DataFrame): Labels with column names as classes, rows as instances
        merge_by (str): The merging option for many to one names

    Raises:
        ValueError: Error message if an invalid merger method is specified

    Returns:
        pd Dataframe: New dataframe with the new (potentially merged) classes
    """
    renamed = df.copy()
    renamed.rename(columns=rename_dict, inplace=True)
    renamed = renamed[sorted(renamed.columns)]

    if merge_by is not None:
        if merge_by == 'max':
            renamed = renamed.T.groupby(level=0).max().T
        elif merge_by == 'mean':
            renamed = renamed.T.groupby(level=0).mean().T
        else:
            raise ValueError("Merge method not recognised, use 'mean' or 'max'")
    return renamed