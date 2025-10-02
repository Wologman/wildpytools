import pandas as pd

def remove_rare_classes(target_df, pred_df, rare_threshold):
    col_sums = target_df.sum(axis=0)
    original_width = target_df.shape[1]
    mask = col_sums >= rare_threshold

    remove_list=[]
    total_removed = 0
    for column_name, col_sum in col_sums.items():
        if col_sum < rare_threshold:
            remove_list.append((column_name,col_sum))
            total_removed +=col_sum
    
    remove_cols = original_width - mask.sum()
    target_df = target_df.loc[:, mask]
    pred_df = pred_df.loc[:, mask]  

    rows_to_remove = target_df.index[target_df.sum(axis=1) == 0]
    target_df = target_df.drop(rows_to_remove)
    pred_df = pred_df.drop(rows_to_remove)

    rows_with_targets = target_df.any(axis=1)
    target_df = target_df[rows_with_targets]
    pred_df = pred_df[rows_with_targets]

    if remove_cols > 0:
        #print(f'From counting loop: Removing {len(remove_list)} species, {total_removed} instances')  #
        print(f'Removing {remove_cols} classes as they have less than {rare_threshold} samples')
        print(f'Also removing {len(rows_to_remove)} rows as they were from those classes')
    return target_df, pred_df


def convert_to_multiclass(df_true, df_pred):
    mask_single = df_true.sum(axis=1) == 1
    df_true_single = df_true[mask_single].copy()
    df_pred_single = df_pred[mask_single].copy()
    df_targets = df_true_single.idxmax(axis=1).to_frame(name='Targets').reset_index(drop=True)
    df_predictions = df_pred_single.reset_index(drop=True)
    df_combined = pd.concat([df_targets.reset_index(drop=True), df_predictions.reset_index(drop=True)], axis=1)
    return df_combined


def rename_predictions(rename_dict, df, merge_by=None):
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


###############################################################################
#####Audio format handling ####################################################
###############################################################################
'''The idea here is to have a standardised dataframe for soundscapes
   Col-names: filepath, ebird, start, stop, max f, min_f, id_by,  reviewed, certainty 

'''

def raven_to_df(label_paths: list,
                name_map: dict,
                ):
    """Converts raven tab delimited text files into a Pandas Dataframe

    Args:
        raven_txt (_type_): filepath as a text string or Path object
    """

    dfs = []
    for path in label_paths:
        df = pd.read_csv(
            path,
            sep='\t',            # Tab delimiter
            header=0,            # First line is header; use None if no header
            encoding='utf-8',    # Explicit encoding
            na_values=['', 'NA'],# Treat empty strings or 'NA' as NaN
            engine='python'      # Use Python engine for complex files (optional)
        )
        dfs.append(df)

    return pd.concat(dfs, index = 0)


def avenza_to_df(label_paths: list,
                 name_map: dict,
                 ):
    """Converts avenza .data label files, and returns a standardised dataframe

    Args:
        raven_txt (_type_): filepath as a text string or Path object
    """

    records = []
    for path in label_paths:
        file_dict = {}
        #open the file
        #read each item
        #create a dictionary
        #add the dictionary to a list
        records.append(file_dict)

    return pd.DataFrame(records)


def predictions_to_avenza(destn_path: str,
                          df: pd.DataFrame):
    """Takes a dataframe in the csv format used for BirdCLEF and 
       converts to the .json like format used by Avenza though with 
       temporal resolution quantised to 5 seconds.

    Args:
        destn_path (str): _description_
        df (pd.DataFrame): _description_
    """
    #group by filepath
    #create the json entries
    #json.dump or just write out the text 
    #save as a filename.wav.data  file