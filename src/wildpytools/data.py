import pandas as pd
from pathlib import Path
import re
import json
from typing import Optional, Union
import xml.etree.ElementTree as ET
from tqdm import tqdm

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
   Filepath | Start Time (s) | End Time (s)	| Low Freq (Hz)	| High Freq (Hz) | Label

   Convert Raven, Avianz and Freebird data into this format before further processing
   Reject any multi-bird labels over some default_length (default 5 seconds)
   Split any multi-bird labels under the default_length into multi-rows
   Centre then crop any long labels to the default_length, using waveform energy 
   For rare cases where long birdsong start/stop not found above, crop to some max_time

   Prior to splitting & training, crops are made and a new ML friendly data format is 
   created for the merged training dataset.  
   For this use the Kaggle Primary/secondary approach plus allowance for marked primary-call centre times

   Test statistics are to be run on soundscapes with the same format as zenodo
   For example:  https://zenodo.org/records/7525805
   Filename | Start Time (s) | End Time (s)	| Low Freq (Hz)	| High Freq (Hz) | Species eBird Code
'''

def pair_audio_labels(data_dir: Union[Path,str],
                      match_suffix: str,
                      label_dir: Optional[Union[Path,str]] = None
                      ):
    audio_exts = {".wav", ".ogg", ".flac"}

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    if label_dir:
        if not isinstance(label_dir, Path):
            label_dir = Path(label_dir)
    else:
        label_dir = data_dir

    paired = {}
    for audio_file in tqdm(data_dir.iterdir()):
        if audio_file.suffix.lower() not in audio_exts:
            continue

        base = audio_file.stem  # e.g. "120401_07"

        # Find all selection files that start with this base and end with .selections.txt
        sel_matches = list(label_dir.glob(f"{base}*{match_suffix}"))

        if sel_matches:
            paired[base] = {
                "audio": audio_file,
                "labels": sel_matches[0]
            }
    return paired


def combine_dfs(dfs, cols):
    combined = pd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]
    try:
        combined = combined[cols]
    except Exception as e:
        print(f'There was an exception {e} combining or filtering columns')
        print(f'There were {len(dfs)} dataframes')
        print(combined.head())
    return combined

def raven_to_df(data_dir: Union[str, Path],
                name_map: Optional[dict] = None):
    """Converts Raven .selections.txt annotation files into a Pandas DataFrame
    along with matched audio files.
    Only column change is the addtion of a Filepath

    Args:
        data_dir (Path | str): Directory containing audio + .selections.txt files
        name_map (dict): Column renaming map for the Raven tables
    """

    cols_to_keep = ['Filepath',	'Start Time (s)', 'End Time (s)',	
                    'Low Freq (Hz)',  'High Freq (Hz)',	'Label']

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    paired = pair_audio_labels(data_dir, '.selections.txt')

    dfs = []
    invalid_dfs = []
    for key in tqdm(paired):
        sel_path = paired[key]['labels']

        df = pd.read_csv(
            sel_path,
            sep='\t',            # Tab delimiter
            header=0,            # First line is header; use None if no header
            encoding='utf-8',    # Explicit encoding
            na_values=['', 'NA'],# Treat empty strings or 'NA' as NaN
            engine='python'      # Use Python engine for complex files (optional)
        )
        if 'Filepath' in df.columns:
            df = df[['Filepath'] + [c for c in df.columns if c != 'Filepath']]
        else:
            df.insert(0, 'Filepath', [sel_path] * len(df))
                #df = df[cols_to_keep]

        if "View" in df.columns:
            #remove any view rows, as they won't be needed
            df = df[df["View"].str.contains("Spectrogram", case=False, na=False)]

        rename_cols = {"Annotation": "Label", "Begin Time (s)": "Start Time (s)"}
        df.rename(columns=rename_cols, inplace=True)

        if name_map is not None:
            #Convert any names that are mapped, and replace the rest with NA
            df["Label"] = df["Label"].map(name_map)
            mapped_values = set(name_map.values())
            df["Label"] = df["Label"].where(df["Label"].isin(mapped_values), pd.NA)

        
        valid_1 = df['Label'].apply(lambda x: isinstance(x, str))
        valid_2 = df['Start Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))
        valid_3 = df['End Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))
        
        valids = df[valid_1 & valid_2 & valid_3]
        errors = df[~valid_1 | ~valid_2 | ~valid_3]

        dfs.append(valids)
        invalid_dfs.append(errors)

    valids = combine_dfs(dfs, cols=cols_to_keep)
    invalids = combine_dfs(invalid_dfs, cols=cols_to_keep)

    return valids, invalids


def validate_name(name: str,
                  name_map: Optional[dict] = None
                  ):
    """_summary_

    Args:
        name (str): The name found in the raven file
        name_map (Optional[dict], optional): mapping to ebird, Defaults to None.

    Returns:
        string: the e-bird name if original name matches a key, 
        or it will be left unchanged if found in e-bird values
    """
    
    if name_map is not None:
        name_map = {k.lower(): v for k, v in name_map.items()}
        mapped_values = set(name_map.values())
        if name not in mapped_values:
            name = name_map.get(name.lower())
    return name


def extract_bird_tags(path_pair: dict,
                      int_to_label: Optional[dict] = None,
                      ):
    audio_path = path_pair['audio']
    label_path = path_pair['labels']

    empty = [{
                'Filepath' : str(audio_path),
                'Start Time (s)': 0,
                'End Time (s)': 0,
                'Low Freq (Hz)': 0,
                'High Freq (Hz)': 0,
                'Label': 'empty'
            }]
    
    try:
        tree = ET.parse(label_path)
        root = tree.getroot()
        records = []
        
        for bird_tag in root.findall('BirdTag'):
            code = int(bird_tag.find('Code').text) if bird_tag.find('Code') is not None else None
            start = round(float(bird_tag.find('TimeSecond').text),1) if bird_tag.find('TimeSecond') is not None else None
            duration = round(float(bird_tag.find('Duration').text),1) if bird_tag.find('Duration') is not None else None
            freq_low = round(float(bird_tag.find('FreqLow').text),1) if bird_tag.find('FreqLow') is not None else None
            freq_high = round(float(bird_tag.find('FreqHigh').text),1) if bird_tag.find('FreqHigh') is not None else None

            end = None
            if start is not None and duration is not None:
                end =  start + duration

            label = None
            if int_to_label:
                label = int_to_label.get(code)
            if label is None:
                label = code

            records.append({
                'Filepath' : str(audio_path),
                'Start Time (s)': start,
                'End Time (s)': end,
                'Low Freq (Hz)': freq_low,
                'High Freq (Hz)': freq_high,
                'Label': label
                })
            
        df = pd.DataFrame(records)

        return None, df
    except:
        print(f'Unable to parse the xml file {label_path.name}')
        return label_path.name, pd.DataFrame


def freebird_to_df(data_dir: Union[str, Path],
                   name_map: Optional[dict] = None,
                   max_length = 5,
                   resample_with_rms = False,
                   ):
    
    cols_to_keep = ['Filepath',	'Start Time (s)', 'End Time (s)',	
                    'Low Freq (Hz)',  'High Freq (Hz)',	'Label']

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    paired = pair_audio_labels(data_dir,
                               match_suffix= '.tag',
                               label_dir=data_dir / '.session')

    #print(paired)

    dfs, invalid_dfs = [], []
    for value in tqdm(paired.values()):
        failures, df = extract_bird_tags(value, int_to_label = name_map)

        #print(df.head())
        if len(df) > 0:
            #valid_1 = df['Label'].apply(lambda x: isinstance(x, str))
            valid_2 = df['Start Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))
            valid_3 = df['End Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))

        
            valids = df #[valid_2 & valid_3]  #valid_1 & 
            errors = df[~valid_2 | ~valid_3] #~valid_1 | 

            dfs.append(valids)
            invalid_dfs.append(errors)

    valids = combine_dfs(dfs, cols=cols_to_keep)
    invalids = combine_dfs(invalid_dfs, cols=cols_to_keep)

    return valids, invalids


def avianz_to_df(data_dir: Union[str, Path],
                       name_map: Optional[dict] = None,
                       max_length = 5,
                       resample_with_rms = False,
                       ):
    """Converts avenza .data label files, and returns a standardised dataframe
       with the avenza bird names converted to e-bird names.

       Rejects any items > max_length with more than a single bird
       Otherwise trims the box to max_length with rms sampling to localise the calls

    Args:
        raven_txt (_type_): filepath as a text string or Path object
    """

    paired = pair_audio_labels(data_dir, '.data')

    for key in tqdm(paired):
        label_path = paired[key]['labels']
        audio_path = paired[key]['audio']  # not used here, but you could store it if needed
        with open(label_path, "r") as f:
            content = json.load(f)
        
        if len(content) <= 1:
            continue

        metadata = content[0]  # first element is metadata
        observations = content[1:]  # rest are observations
        
        records = []
        for obs in observations:
            record = {
                "Filepath": audio_path,
                "Start Time (s)": round(obs[0], 1),
                "End Time (s)": round(obs[1], 1),
                "Low Freq (Hz)": round(obs[2], 1),
                "High Freq (Hz)": round(obs[3], 1)
                }

            duration = obs[1] - obs[0]
            bird_list = obs[4]

            # Skip if multi-bird and too long
            if len(bird_list) > 1 and duration > max_length:
                continue

            # Skip if bird_list is empty
            if not bird_list:
                continue

            if duration <= max_length:
                # Multiple short observations allowed
                for bird in bird_list:
                    species = validate_name(bird.get('species'), name_map)
                    if species is not None:
                        record_copy = record.copy()
                        record_copy['Label'] = species
                        records.append(record_copy)
            else:
                # Single long observation
                bird = bird_list[0]
                species = validate_name(bird.get('species'), name_map)
                if species is not None:
                    record_copy = record.copy()
                    record_copy['Label'] = species
                    records.append(record_copy)

        # Create the DataFrame
    df = pd.DataFrame(records)
    return df, None


def predictions_to_annotations(df: pd.DataFrame,
                               name_map: Optional[dict] = None,
                               threshold: float = 0.5,
                               ):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        name_map (Optional[dict], optional): _description_. Defaults to None.
        threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: Another dataframe with aggregated and thresholded predictions
        using what ever naming scheme was stored in the name_map dictionary
    """
    #convert name columns with the name_map
    #extract start and stop time from the id column
    #apply binary threshold
    #split any rows with multiple birds
    #merge any adjacent rows

    return df


def predictions_to_avianz(destn_dir: Union[str, Path],
                          df: pd.DataFrame,
                          name_map: Optional[dict] = None,
                          threshold: float = 0.5,
                          ):
    """Takes a dataframe in the csv format used for BirdCLEF and 
       converts to the .json-like format used by Avenza though with 
       temporal resolution quantised to 5 seconds.

    Args:
        destn_path (str): _description_
        df (pd.DataFrame): _description_
    """
    annots = predictions_to_annotations(df,
                                        name_map=name_map,
                                        threshold=threshold)
    #create any empty columns for things like frequency & power density
    #json.dump to a .data file
    return


def predictions_to_raven(destn_dir: Union[str, Path],
                         df: pd.DataFrame,
                         name_map: Optional[dict] = None,
                         threshold: float = 0.5,
                         ):
    """Converts a dataframe in the format used for BirdCLEF and 
       to the tab seperated .selections.txt files of raven,
       with temporal resolution quantised to 5 seconds.

    Args:
        destn_path (str): a directory path for the .selections.txt files
        df (pd.DataFrame): dataframe with 5-second presence/absence values
    """
    annots = predictions_to_annotations(df,
                                        name_map=name_map,
                                        threshold=threshold)
    
    #create any empty columns for things like frequency & power density
    #write to a .selections.text file with tabs as the seperator
    return