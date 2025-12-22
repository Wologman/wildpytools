import pandas as pd
from pathlib import Path
import json
from typing import Optional, Union, Sequence, Literal, Tuple, Set
import xml.etree.ElementTree as ET
from tqdm import tqdm
import IPython.display as ipd
from IPython.display import display, HTML
import torch
import torchaudio
import numpy as np
import librosa
import plotly.express as px
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import ast
from scipy.signal import resample
from typing import Literal
from joblib import Parallel, delayed
import soundfile as sf
import logging
import re

# Set up logger for this module
logger = logging.getLogger(__name__)


###############################################################################
####################Audio format handling #####################################
###############################################################################
'''The idea here is to have a standardised dataframe for soundscapes
   1. Annotations
   Filename | Start Time (s) | End Time (s)	| Low Freq (Hz)	| High Freq (Hz) | Label
   Plus a convenient method to turn this to and from Raven .selections files

   2. Metadata
   Using BirdCLEF column names, as default but in principle allowing unlimited extra columns.
   As a minimum (even if some are empty):
   | primary_label | secondary_labels | type | filename | collection | rating | url | latitude
   | longitude | scientific_name | common_name | author | license | reviewed_by | reviewed_on

   Test (and potentially val) performance metrics can be conveniently be run directly
   on these these fixed-length soundscapes, split from training data by location, 
   or at least by date.

   For training new models, crops can be made and a more dataloader-friendly format. 
   The method proposed here uses the Kaggle Primary/secondary approach plus columns 
   for marked primary-call bounding boxes

   To get historical data into the above format we do the following:
   * Convert the various data formats: BirdCLEF, BirdClef-Zenodo, Raven, Freebird, Avianz
     into the unified format above before further processing
   * Run an object-detection model over the data to propose missing T-F bounding boxes
   * Reject any multi-bird labels over default_length (3 sec), from Avianz for example
   * Split any multi-bird labels under default_length (3 sec) into multi-rows 
   * Centre then crop any long labels to default_length, using a detection model
   * For rare cases where long birdsong start/stop not found above, crop to max_time
'''

#Helper functions
def tail_path(path, depth: int):
    p = Path(path)
    return Path(*p.parts[-depth:]).as_posix()



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


def combine_dfs(dfs: Sequence[pd.DataFrame], cols: Sequence[str]) -> pd.DataFrame:
    if not dfs:
        logger.warning('Attempted to combine empty dataframe list; returning empty dataframe')
        return pd.DataFrame(columns=cols)

    combined = pd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]
    try:
        combined = combined[cols]
    except Exception as e:  # ISSUE #2: Generic exception handling - should catch KeyError, IndexError specifically
        logger.error(f'Exception {e} combining or filtering columns from {len(dfs)} dataframes')
        logger.debug(f'Combined dataframe head:\n{combined.head()}')
    return combined


def load_from_raven(data_dir: Union[str, Path],
                    name_map: Optional[dict] = None,
                    metadata_path: Optional[Union[str, Path]] = None,
                    metadata_dict: Optional[dict] = None):
    
    """
    Converts Raven .selections.txt annotation files into a Pandas DataFrame
    along with matched audio files. The only column change is the addition 
    of a Filepath

    Args:
        data_dir (Path | str): Directory containing audio + .selections.txt files
        name_map (dict): Column renaming map for the Raven tables
        metadata_csv (Path | str): A CSV file with metadata and matching filenames
        metadata_dict (dict): Optional metadata that can update the csv data
    """

    annot_cols_to_keep = ['Filename',	'Start Time (s)', 'End Time (s)',	
                          'Low Freq (Hz)',  'High Freq (Hz)',	'Label', 
                          'Type', 'Rating', 'Delta Time (s)', 'Delta Freq (Hz)',
                          'Avg Power Density (dB FS/Hz)','Filepath']

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    print(f'The data dir is {data_dir}')
    paired = pair_audio_labels(data_dir, '.selections.txt')

    dfs = []
    for key in tqdm (paired):
        sel_path = paired[key]['labels']
        audio_path = paired[key]['audio']

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
            df.insert(0, 'Filepath', [audio_path] * len(df))

        cols_to_check = {'Type', 'Rating'}
        missing_cols = cols_to_check - set(df.columns)
        for col in missing_cols:
            df[col] = pd.NA


        base_dir = Path(data_dir).resolve()
        df["Filename"] = df["Filepath"].apply(
            lambda p: str(Path(p).resolve().relative_to(base_dir))
        )
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

        valid_1 = df['Label'].notna() & (df['Label'].apply(type) == str)
        is_numeric_col = pd.api.types.is_numeric_dtype(df['Start Time (s)'])
        valid_2 = df['Start Time (s)'].notna() & (is_numeric_col | df['Start Time (s)'].apply(type).isin([int, float]))
        is_numeric_col_end = pd.api.types.is_numeric_dtype(df['End Time (s)'])
        valid_3 = df['End Time (s)'].notna() & (is_numeric_col_end | df['End Time (s)'].apply(type).isin([int, float]))
        
        valids = df[valid_1 & valid_2 & valid_3]

        dfs.append(valids)

    valid_labels = combine_dfs(dfs, cols=annot_cols_to_keep)

    if metadata_path is not None:
        df_meta = pd.read_csv(metadata_path) #Placeholder
    else: 
        filepaths = sorted(list(set(valid_labels['Filepath'])))
        filenames = sorted(list(set(valid_labels['Filename'])))
        df_meta = pd.DataFrame([metadata_dict]*len(filenames))
        df_meta['filepath'] = filepaths
        df_meta['filename'] = filenames
        df_meta['source_filename'] = filenames

    return valid_labels, df_meta  #This isn't what we mean by invalids.  Use the metadata df


def find_relative_audio_filenames(root: Path) -> list[Path]:
    """
    Recursively find all audio files under `root`.

    Returns paths *relative to root*.
    """
    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a"}
    root = root.resolve()

    return [
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]





def validate_filenames(
    base_dir: Path,
    list_of_filenames: Sequence[str],
    purpose: str = "",
) -> Set[Path]:
    """
    Confirm that a list of filenames exist under base_dir.

    Returns:
        valid_paths: files present both on disk and in the list
        missing_paths: files referenced in the list but not found on disk
    """

    base_dir = base_dir.resolve()

    # Files found on disk (relative → absolute → resolved)
    existing_files = find_relative_audio_filenames(base_dir)
    existing_paths = {
        (base_dir / fp).resolve()
        for fp in existing_files
    }

    # Files referenced by the dataframe (relative → absolute → resolved)
    referenced_paths = {
        (base_dir / fn).resolve()
        for fn in list_of_filenames
    }

    missing_paths = referenced_paths - existing_paths
    extra_files = existing_paths - referenced_paths
    valid_paths = referenced_paths & existing_paths

    if missing_paths:
        print(
            f"[WARNING] {len(missing_paths)} {purpose} file(s) were referenced "
            f"but not found under {base_dir}. Examples:\n"
            f"{list(missing_paths)[:3]}\n"
        )

    if extra_files:
        print(
            f"[INFO] {len(extra_files)} file(s) exist under {base_dir} "
            f"but are not referenced by the {purpose} dataframe. Examples:\n"
            f"{list(extra_files)[:14]}\n"
        )

    return {str(p.relative_to(base_dir)) for p in valid_paths}


def load_from_anqa(dir_path: Path,
                   metadata_path: Path,
                   labels_path: Path,
                   name_map: Optional[dict] = None):

    try:
        df_labels = pd.read_csv(labels_path)
        if len(df_labels) < 1:
            logger.warning('Warning: The labels dataframe is empty')
    except Exception as e:
        logger.error("Unable to load labels dataframe")
        df_labels = pd.DataFrame()

    try:
        df_meta = pd.read_csv(metadata_path)
        if len(df_meta) < 1:
            logger.warning('Warning: The metadata dataframe is empty')
    except Exception as e:
        logger.error("Unable to metadata dataframe")
        df_meta = pd.DataFrame()

    labels_paths = set(df_labels['Filename'])
    metadata_paths = set(df_meta['filename'])
    valid_metadata_fns = validate_filenames(dir_path, metadata_paths, purpose='metadata')
    valid_label_fns = validate_filenames(dir_path, labels_paths, purpose = 'annotations')
    valid_names = valid_metadata_fns & valid_label_fns

    df_labels = df_labels[df_labels["Filename"].isin(valid_names)]
    df_meta = df_meta[df_meta["filename"].isin(valid_names)]

    if name_map is not None:
        df_labels["Label"] = df_labels["Label"].replace(name_map)

    return df_labels, df_meta


def add_missing_label_columns(df):
    float_cols = [
        'Start Time (s)', 'End Time (s)',
        'Low Freq (Hz)', 'High Freq (Hz)'
    ]
    for col in float_cols:
        if col not in df.columns:
            df[col] = pd.Series(index=df.index, dtype='float64')

    datetime_cols = ['recorded_on', 'reviewed_on']
    for col in datetime_cols:
        if col not in df.columns:
            df[col] = pd.Series(index=df.index, dtype='datetime64[ns]')

    text_cols = ['reviewed_by']
    for col in text_cols:
        if col not in df.columns:
            df[col] = pd.Series(index=df.index, dtype='string')

    if 'models_used' not in df.columns:
        df['models_used'] = ['[]' for _ in range(len(df))]

    return df


def load_from_birdclef(dir_path: Path,
                       metadata_path: Path,
                       name_map: Optional[dict] = None):
    """Loads a birdclef style train.csv file and formats into anqa format with seperate
       dataframes for annotations and metadata.

    Args:
        dir_path (Path): Path to lowest directory containing all the audio files of interest
        metadata_path (Path): Path to the BirdCLEF format metadata csv
        name_map (Optional[dict], optional): {e-bird/inat code: common name}. Defaults to None.
        cols_to_keep (Optional[list], optional): Columns to display and save. 
                     Defaults to ['filename', primary_label, 'secondary_labels']
    """

    def rename_list(lst):
        if isinstance(lst, str):
            try:
                lst = ast.literal_eval(lst)
            except (ValueError, SyntaxError):
                pass  # not a valid list string
        if not isinstance(lst, list):
            return lst
        return [name_map.get(item, item) for item in lst]

    dir_path = Path(dir_path)
    metadata_path = Path(metadata_path)
    
    if metadata_path.suffix == '.parquet':
        df = pd.read_parquet(metadata_path, engine="pyarrow")
        logger.debug("Loading parquet file")
    elif metadata_path.suffix == '.csv':
        df = pd.read_csv(metadata_path)
    else:
        raise TypeError("Metadata file must be a .csv or a .parquet file type")

    if name_map is not None:
        df["primary_label"] = df["primary_label"].map(name_map).fillna(df["primary_label"])
        df["secondary_labels"] = df["secondary_labels"].apply(rename_list)

    df = df.rename(columns={
        'primary_label': 'Label',
        'type': 'Type',
        'rating': 'Rating',
    })

    df = add_missing_label_columns(df)

    # Add a filepath column so we can test existence
    df['filepath'] = df['filename'].apply(lambda p: dir_path / p)
    #mask_exists = df['filepath'].apply(lambda p: p.exists())

    # Now format the source_filename and filename columns
    #df['filename'] = #df['Filepath'].apply(lambda x: tail_path(x, depth=2))
    df['source_filename'] = df['filename'] #df['Filepath'].apply(lambda x: tail_path(x, depth=2))
    

    filenames = set(df['filename'])
    valid_fns = validate_filenames(dir_path, filenames, purpose='metadata')

    df = df[df["filename"].isin(valid_fns)].copy()

    cols_to_move = ['Label', 'Start Time (s)', 'End Time (s)',
                    'Low Freq (Hz)', 'High Freq (Hz)', 'Type', 'Rating']
    df_labels = pd.DataFrame({c: df.pop(c) for c in cols_to_move})

    df_labels['Filename'] = df['filename']
    df_labels['Filepath'] = df['filepath']  # already computed
    df_labels = df_labels[['Filename'] + cols_to_move + ['Filepath']]

    return df_labels, df


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
    except:  # ISSUE #2: Bare except clause - should catch specific exceptions (ET.ParseError, FileNotFoundError, etc.)
        logger.warning(f'Unable to parse the xml file {label_path.name}')
        return label_path.name, pd.DataFrame  # ISSUE #6: Returns pd.DataFrame (class) instead of pd.DataFrame() (empty instance)


def load_from_bc_zenodo(data_dir: Union[str, Path],
                        labels_path: Union[str, Path],
                        metadata_path: Optional[Union[str, Path]] = None,
                        name_map: Optional[dict] = None
                        ):
    
    cols_to_keep = ['Filepath',	'Start Time (s)', 'End Time (s)',	
                    'Low Freq (Hz)',  'High Freq (Hz)',	'Label']

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    df = pd.read_csv(labels_path)

    df = df.rename(columns={'Start Time (s)': 'Begin Time (s)',
                            'Species eBird Code': 'Label'})

    if label_cols_to_keep is None:
        label_cols_to_keep = ['filename', 'primary_label', 'secondary_labels', 'Reviewed By',
                              'Reviewed On', 'Filepath', 'Start Time (s)', 'End Time (s)', 
                              'Low Freq (Hz)','High Freq (Hz)', 'Annotation']

    dfs, invalid_dfs = [], []
    #for value in tqdm(paired.values()):
    #    failures, df = extract_bird_tags(value, int_to_label = name_map)

        #print(df.head())  # ISSUE #4: Commented out code - remove if not needed
    #    if len(df) > 0:
    #        #valid_1 = df['Label'].apply(lambda x: isinstance(x, str))  # ISSUE #4: Commented out code - remove if not needed
    #        valid_2 = df['Start Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))
    #        valid_3 = df['End Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))

        
    #        valids = df #[valid_2 & valid_3]  #valid_1 &   # ISSUE #4: Commented out validation - should validate or remove comment
    #        errors = df[~valid_2 | ~valid_3] #~valid_1 |  # ISSUE #4: Commented out validation - should validate or remove comment 

    #        dfs.append(valids)
     #       invalid_dfs.append(errors)

    #valids = combine_dfs(dfs, cols=cols_to_keep)
    #invalids = combine_dfs(invalid_dfs, cols=cols_to_keep)

    return df, pd.DataFrame(), pd.DataFrame()



def load_from_freebird(data_dir: Union[str, Path],
                   name_map: Optional[dict] = None
                   ):
    
    cols_to_keep = ['Filepath',	'Start Time (s)', 'End Time (s)',	
                    'Low Freq (Hz)',  'High Freq (Hz)',	'Label']

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    paired = pair_audio_labels(data_dir,
                               match_suffix= '.tag',
                               label_dir=data_dir / '.session')

    dfs, invalid_dfs = [], []
    for value in tqdm(paired.values()):
        failures, df = extract_bird_tags(value, int_to_label = name_map)

        if len(df) > 0:
            #valid_1 = df['Label'].apply(lambda x: isinstance(x, str))  # ISSUE #4: Commented out code - remove if not needed
            valid_2 = df['Start Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))
            valid_3 = df['End Time (s)'].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))

        
            valids = df #[valid_2 & valid_3]  #valid_1 &   # ISSUE #4: Commented out validation - should validate or remove comment
            errors = df[~valid_2 | ~valid_3] #~valid_1 |  # ISSUE #4: Commented out validation - should validate or remove comment

            dfs.append(valids)
            invalid_dfs.append(errors)

    valids = combine_dfs(dfs, cols=cols_to_keep)
    invalids = combine_dfs(invalid_dfs, cols=cols_to_keep)

    return valids, invalids


def load_from_avianz(data_dir: Union[str, Path],
                       name_map: Optional[dict] = None,
                       max_multibird_length: float = 5,
                       keep_multibird_labels: bool = True
                       ):
    """Converts avenza .data label files, and returns a standardised dataframe
       with the avenza bird names converted to e-bird names.

       Rejects any items > max_length with more than a single bird
       Otherwise trims the box to max_length with rms sampling to localise the calls

    Args:
        raven_txt (_type_): filepath as a text string or Path object
    """

    paired = pair_audio_labels(data_dir, '.data')
    all_records = []

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
            if len(bird_list) > 1 and duration > max_multibird_length:
                continue

            # Skip if bird_list is empty
            if not bird_list:
                continue

            # Skip if rejecting all multi-bird labels
            if len(bird_list) > 1 and not keep_multibird_labels:
                continue

            if duration <= max_multibird_length:
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

        all_records.extend(records)

    df = pd.DataFrame(all_records)
    return df, None



def load_audio(path: Path) -> Optional[Tuple[np.ndarray, int]]:
    try:
        wav, sr = librosa.load(path, sr=None)  # returns mono NumPy array
    except Exception as e:
        logger.error(f"The file '{path.name}' failed to load: {e}")
        return None

    if wav.size == 0:
        logger.warning(f"The file '{path.name}' is empty and will be skipped.")
        return None

    if sr is None or sr <= 0:
        logger.warning(f"The file '{path.name}' has an invalid sampling rate ({sr}) and will be skipped.")
        return None
    return wav, sr


def compute_spectrogram(waveform: np.ndarray | torch.Tensor,
                        sr: int,
                        n_fft: int = 1024,
                        hop_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a power spectrogram from waveform (NumPy or torch).
    Returns: (power, freqs, times)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform.unsqueeze(0)

    spec = torch.stft(waveform, 
                      n_fft=n_fft,
                      hop_length=hop_length,
                      window= torch.hann_window(n_fft),
                      return_complex=True)
    power = spec.abs() ** 2

    freqs = torch.fft.rfftfreq(n_fft, 1 / sr)
    times = torch.arange(spec.shape[-1]) * hop_length / sr

    return power, freqs, times


def avg_power_density(wav: np.ndarray,
                      sr: int,
                      t_start: float,
                      t_end: float,
                      f_low: float,
                      f_high: float) -> float:
    """
    Compute Avg Power Density (dB FS/Hz) from a precomputed spectrogram.
    """
    power, freqs, times = compute_spectrogram(wav, sr, n_fft=1024, hop_length=256)
    freqs = freqs.numpy()
    times = times.numpy()

    t_mask = (times >= t_start) & (times <= t_end)
    f_mask = (freqs >= f_low) & (freqs <= f_high)

    if not t_mask.any() or not f_mask.any():
        return np.nan

    roi = power[..., f_mask, :][..., t_mask]
    mean_power = roi.mean().item()

    bandwidth = f_high - f_low
    if bandwidth <= 0:
        return np.nan

    power_density = mean_power / bandwidth
    return round(10 * np.log10(power_density + 1e-12), 1)


def anqa_to_raven_selections(df: pd.DataFrame,
                             destn_dir: Path):
        '''Creates .selections.txt tab seperated annotations
        for visualisation of spectrogram labels in Raven
        (https://www.ravensoundsoftware.com/)
        '''
        destn_dir = Path(destn_dir)
        rename_cols = {"Label": "Annotation", "Start Time (s)": "Begin Time (s)"}
        df.rename(columns=rename_cols, inplace=True)

        cols_to_keep = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                        'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',
                        'Delta Time (s)', 'Delta Freq (Hz)',
                        'Avg Power Density (dB FS/Hz)', 'Annotation']

        df['View'] = 1
        df['Channel'] = 1
        df['Delta Time (s)'] = df['End Time (s)'] - df['Begin Time (s)']
        df['Delta Freq (Hz)'] = df['High Freq (Hz)'] - df['Low Freq (Hz)']
        
        for fn, group in df.groupby('Filename'):
            # Create Selection 1..N
            group['Selection'] = np.arange(1, len(group) + 1, dtype=int)
            file_stem = Path(fn).stem
            selections_destn = destn_dir / f'{file_stem}.Table.1.selections.txt'
            group = group[cols_to_keep]
            group.to_csv(selections_destn, sep='\t', index=False)


class ToAnqa:
    def __init__(self,
                 source_dir: Union[str, Path],
                 destn_dir: Union[str, Path],
                 name_map: Optional[dict] = None,
                 save_audio: bool = False,
                 max_seconds: int = 60,
                 max_hz: int = 1600,
                 min_hz: int = 0,
                 end_padding: Literal['pad', 'noise', 'zero', None] = None,
                 crop_method: Literal['keep_all'] = 'keep_all', # Future: 'random', 'max_annotations', 'max_detections'
                 default_sr: int = 32000,
                 n_jobs: int = 0,
                 destn_depth: int = 2,
                 ):
        
        self.source_dir = Path(source_dir)
        self.destn_dir = Path(destn_dir)
        self.audio_destn = self.destn_dir / 'audio'
        self.audio_destn.mkdir(exist_ok=True, parents=True)
        self.name_map = name_map
        self.save_audio = save_audio
        self.max_seconds = max_seconds
        self.max_hz = max_hz
        self.min_hz = min_hz
        self.end_padding = end_padding
        self.crop_method = crop_method
        self.audio = None
        self.results = {}
        self.default_sr = default_sr
        self.n_jobs = n_jobs
        self.parallel = False if n_jobs == 0 else True
        self.destn_depth = destn_depth

        # --- Validation check ---
        if self.save_audio and self.source_dir == self.destn_dir:
            raise ValueError(
                f"`save_audio` cannot be True when `source_dir` and `destn_dir` "
                f"are the same ({self.source_dir})"
            )

    def _validate_labels(self, 
                         df: pd.DataFrame,
                         wave: np.ndarray,
                         sr: int,
                         ) -> pd.DataFrame:
        """
        Validate the annotations, & fill in any missing values with the limits
        """

        df = df.copy()
        df = df.reset_index(drop=True)

        # Fill defaults for missing values
        df['Start Time (s)'] = df['Start Time (s)'].fillna(0.0)
        df['End Time (s)'] = df['End Time (s)'].fillna(round(len(wave) / sr, 1))
        df['Low Freq (Hz)'] = df['Low Freq (Hz)'].fillna(self.min_hz).astype(float)
        df['High Freq (Hz)'] = df['High Freq (Hz)'].fillna(self.max_hz).astype(float)

        # Swap bounds if out of order
        mask_time = df['End Time (s)'] < df['Start Time (s)']
        if mask_time.any():
            df.loc[mask_time, ['Start Time (s)', 'End Time (s)']] = \
                df.loc[mask_time, ['End Time (s)', 'Start Time (s)']].values

        mask_freq = df['High Freq (Hz)'] < df['Low Freq (Hz)']
        if mask_freq.any():
            df.loc[mask_freq, ['Low Freq (Hz)', 'High Freq (Hz)']] = \
                df.loc[mask_freq, ['High Freq (Hz)', 'Low Freq (Hz)']].values

        return df    

    def save_one_segment(self, segment: dict):
        file_stem = Path(segment['filename']).stem
        if self.save_audio:
            destn = self.audio_destn / f'{file_stem}.flac'
            wav = segment['wave']
            if segment['sr'] != self.default_sr:
                num_samples = int(len(wav) * self.default_sr / segment['sr'])
                wav = resample(wav, num_samples)
            sf.write(destn, wav, self.default_sr)

    def segment_audio(self,
                      wav: np.ndarray,
                      sr: int,
                      filename: str, 
                      df: pd.DataFrame,
                      meta_dict: dict,
                      min_remainder_length_sec: float = 1):
        """Break up longer audio files into fixed-length segments,
        optionally pad shorter ones, and adjust label times accordingly.
        The remainder from any non-whole segments will be discarded if less
        than min_remainder_length_sec"""

        def compute_num_segments(wav_length, segment_length, min_segment):
            num = int(np.ceil(wav_length / segment_length))
            remainder = wav_length - (num - 1) * segment_length
            if num > 1 and remainder < min_segment:
                num -= 1
                remainder = 0
            return max(1, num), remainder

        max_seconds = self.max_seconds
        end_padding = self.end_padding
        crop_method = self.crop_method
        segment_length = int(max_seconds * sr)
        wav_length = len(wav)
        original_len_secs = wav_length // sr

        #Need to identify if this is a second or subsequent conversion.     
        # Filepath will have a   from_xx.flac
        # Same for the filename column in the dataframe
        # Need to extract this integer (if any) and add it to the future filename integer
        # Also need to avoid doubling up the _from_xx  characters.

        file_stem = Path(filename).stem
        segmented_previously = bool(re.search(r"_from_\d+$", file_stem))
        if segmented_previously:
            #Default to the metadata offset value, but use the filepath extracted one as a fallback
            if meta_dict is not None:
                offset = int(re.search(r"_from_(\d+)$", file_stem).group(1)) if segmented_previously else 0
                offset = meta_dict.get('source_start_s') or offset
        else:
            offset = 0

        clean_stem = re.sub(r"_from_\d+$", "", file_stem) if segmented_previously else file_stem

        # Fill default boxes to the whole clip
        df['Start Time (s)'] = df['Start Time (s)'].fillna(0.0)
        df['End Time (s)'] = df['End Time (s)'].fillna(round(original_len_secs, 1)) #no point including the padding
        df['Low Freq (Hz)'] = df['Low Freq (Hz)'].fillna(0).astype(float)
        df['High Freq (Hz)'] = df['High Freq (Hz)'].fillna(16000).astype(float)

        # Need to identify any previous crops from the source
        # Then calculate the start times as per the various possible cropping schemes
        # Create a list of start and end times.
        # Have only one branch that iterates through that list.

        num_segments, remainder = compute_num_segments(wav_length,
                                                       segment_length,
                                                       sr*min_remainder_length_sec)

        # --- Optional padding step for final segment ---
        if remainder != 0 and end_padding is not None:
            pad_length = segment_length - remainder
            if end_padding == 'noise':
                pad = np.random.normal(0, np.std(wav[-min(wav_length, segment_length):]), pad_length)
            elif end_padding == 'pad':
                pad = np.resize(wav, pad_length)
            elif end_padding == 'zero':
                pad = np.zeros(pad_length)
            else:
                pad = np.array([])
            wav = np.concatenate([wav, pad])

        seg_idxs = []
        if wav_length <= segment_length:
            start_idx = 0
            end_idx = wav_length
            seg_idxs.append((start_idx, end_idx))
        elif crop_method == 'keep_all':
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = min((i+1) * segment_length, len(wav))
                seg_idxs.append((start_idx, end_idx))
        else:
            start_idx = np.random.choice(wav_length - segment_length)
            end_idx = start_idx + segment_length
            seg_idxs.append((start_idx, end_idx))

        segments = []

        base_meta = meta_dict.copy() if meta_dict is not None else {}

        for idxs in seg_idxs:
            start_idx = idxs[0]
            end_idx = idxs[1]
            start_time = start_idx / sr
            end_time = end_idx / sr
            #Take only rows where the label overlaps with that segment
            seg_df = df[
                    (df["Start Time (s)"] < end_time) &
                    (df["End Time (s)"] > start_time)
                ].copy()

            ref_start_time = start_time + offset
            ref_end_time = end_time + offset
            seg_fname = f"{clean_stem}_from_{int(ref_start_time)}.flac" #for save destn
            col_fname = tail_path(str(seg_fname),depth=self.destn_depth) #for csv
            seg_meta_dict = base_meta.copy()
            seg_meta_dict['filename'] = col_fname
            seg_meta_dict['source_start_s'] = f'{ref_start_time:.1f}'
            seg_meta_dict['source_end_s'] = f'{ref_end_time:.1f}'

            seg_df["Filename"] = col_fname
            seg_df["Start Time (s)"] -= start_time 
            seg_df["End Time (s)"] -= start_time
            seg_df["Start Time (s)"] = seg_df["Start Time (s)"].clip(lower=0)
            seg_df["End Time (s)"] = seg_df["End Time (s)"].clip(lower=0, upper=(end_time - start_time))

            segment_wav = wav[start_idx:end_idx]
            seg_df = self._validate_labels(seg_df, wav, sr)
            power_results = []

            for idx, row in seg_df.iterrows():
                power = avg_power_density(
                        segment_wav,
                        sr,
                        float(row["Start Time (s)"]),
                        float(row["End Time (s)"]),
                        float(row["Low Freq (Hz)"]),
                        float(row["High Freq (Hz)"]),
                    )
                power_results.append(power)

            seg_df["Avg Power Density (dB FS/Hz)"] = power_results
            seg_meta_dict.pop('filepath', None)
            seg_df['Delta Time (s)'] = seg_df['End Time (s)'] - seg_df['Start Time (s)']
            seg_df['Delta Freq (Hz)'] = seg_df['High Freq (Hz)'] - seg_df['Low Freq (Hz)']

            segments.append({
                    "filename": seg_fname,
                    "wave": segment_wav,
                    "sr": sr,
                    "labels_df": seg_df,
                    "meta_dict": seg_meta_dict,
                })

        return segments


    def convert_one_file(self, filename: str, df: pd.DataFrame, df_meta: pd.DataFrame) -> list[dict]:
        """Convert a single file into a raven-compatible slections table and standard length .flac file"""  
        
        cols_to_keep = ['Filename', 'Label', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)']
        df = df[cols_to_keep].copy()

        if df_meta is not None:
            if len(df_meta) != 1:
                raise ValueError(f"Unexpected duplicate metadata rows for {df_meta.iloc[0].get('filename')}, {len(df_meta)} rows found")
            meta_dict = df_meta.iloc[0].to_dict()

        wav, sr = load_audio(self.source_dir / filename)
        if wav is None:  # Should handle None return from load_audio
            return []  # Return empty list if audio failed to load

        if self.save_audio:
            segmented = self.segment_audio(wav, sr, filename, df, meta_dict)
        else:
            meta_dict.pop('filepath', None)
            segmented = [{'filename': filename, 'labels_df': df, 'wave' : wav, 'sr': sr, 'meta_dict': meta_dict.copy() if meta_dict is not None else {}}]
        #At this point the df in 'labels_df' should contain one row for any split up files.

        if self.save_audio:
            for segment in segmented:
                self.save_one_segment(segment)
        return segmented


    def convert_all(self, df_labels: pd.DataFrame, df_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert all grouped recordings from two DataFrames using different grouping columns."""

        grouped_labels = df_labels.groupby('Filename')
        grouped_meta = df_meta.groupby('filename')

        # Find filenames that exist in both
        #common_filenames = grouped_labels.groups.keys() & grouped_meta.groups.keys()
        common_filenames = [fname for fname in df_labels['Filename'].unique() if fname in grouped_meta.groups]

        # Build a list of (filename, group_from_df1, group_from_df2)
        groups = [(fname, grouped_labels.get_group(fname), grouped_meta.get_group(fname)) for fname in common_filenames]

        if self.parallel:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.convert_one_file)(fname, labels, metadata)
                for fname, labels, metadata in tqdm(groups, desc="Converting recordings")
            )
        else:
            results = [
                self.convert_one_file(fname, labels, metadata)
                for fname, labels, metadata in tqdm(groups, desc="Converting recordings")
            ]

        flattened_labels = [seg['labels_df'] for file_segments in results for seg in file_segments]
        flattened_metadata = [seg['meta_dict'] for file_segments in results for seg in file_segments]

        self.review_labels = pd.concat(flattened_labels, ignore_index=True)
        metadata_df = pd.DataFrame(flattened_metadata)

        #Need to add any missing columns with empty values
        #Ignore primary_label, type & rating.  Type and rating really ought to be in annotations
        self.review_labels = add_missing_label_columns(self.review_labels)

        final_label_cols = ['Filename',	'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 
                            'High Freq (Hz)',	'Label', 'Type', 'Rating', 'Delta Time (s)',
                            'Delta Freq (Hz)', 'Avg Power Density (dB FS/Hz)']
        
        final_metadata_cols = ['filename', 'collection', 'secondary_labels', 'url',
                               'latitude', 'longitude', 'author', 'license',
                               'recorded_on', 'reviewed_by', 'reviewed_on',
                               'source_filename', 'source_start_s', 'source_end_s', 'models_used']
        
        for df, cols in zip([self.review_labels, metadata_df],
                            [final_label_cols, final_metadata_cols]):
            missing_cols = set(cols) - set(df.columns)
            for col in missing_cols:
                df[col] = pd.NA

        return self.review_labels[final_label_cols], metadata_df[final_metadata_cols]


def predictions_to_annotations(df: pd.DataFrame,
                               name_map: Optional[dict] = None,
                               threshold: float = 0.5,
                               ):
    """_summary_  # ISSUE #12: Incomplete docstring - should describe what function does

    Args:
        df (pd.DataFrame): _description_  # ISSUE #12: Incomplete docstring
        name_map (Optional[dict], optional): _description_. Defaults to None.
        threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: Another dataframe with aggregated and thresholded predictions
        using what ever naming scheme was stored in the name_map dictionary
    """
    # Placeholder for future implementation
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
        destn_path (str): _description_  # ISSUE #12: Parameter name mismatch - docstring says destn_path but parameter is destn_dir
        df (pd.DataFrame): _description_
    """
    annots = predictions_to_annotations(df,
                                        name_map=name_map,
                                        threshold=threshold)
    # Placeholder for future implementation
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
        destn_dir (str): a directory path for the .selections.txt files
        df (pd.DataFrame): dataframe with 5-second presence/absence values
    """
    annots = predictions_to_annotations(df,
                                        name_map=name_map,
                                        threshold=threshold)
    
    #create any empty columns for things like frequency & power density
    #write to a .selections.text file with tabs as the seperator  
    return


###############################################################################
#####Interactive tools for audio labelling and label reviewing#################
###############################################################################


class VoiceDetector():
    def __init__(self, chunk_len: float, threshold: float = 0.1, no_voice: float = 0, voice: float = 20):
        MODEL_PATH = 'snakers4/silero-vad'
        model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(repo_or_dir=MODEL_PATH,
                                                                 model='silero_vad', verbose=False)
        self.model=model
        self.chunk_len=chunk_len
        self.threshold=threshold
        self.get_stamps = get_speech_timestamps
        self.voice_threshold 

    def detect(self, np_wav: np.ndarray) -> np.ndarray:
        speech_timestamps = self.get_stamps(torch.Tensor(np_wav), self.model, threshold=self.threshold)
        voice_detect = np.zeros_like(np_wav)
        for st in speech_timestamps:
            voice_detect[st['start']: st['end']] = 20
    
        #downsample to match the power plot axis
        voice_detect = np.pad(voice_detect, 
                            (0, int(np.ceil(len(voice_detect) / self.chunk_len) * self.chunk_len - len(voice_detect))))
        voice_detect = voice_detect.reshape((-1, self.chunk_len)).max(axis=1)  # Use max to preserve speech detection
        return voice_detect
    
def calc_signal_pwr(wav, chunk_len, sr=32000):
    power = wav ** 2 
    power = np.pad(power, (0, int(np.ceil(len(power) / chunk_len) * chunk_len - len(power))))
    power = power.reshape((-1, chunk_len)).sum(axis=1)
    return power

class MelSpecMaker():
    def __init__(self, sr=32000, n_mels=128, n_fft=2048, f_min = 20, f_max=14000):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.mel_transform = T.MelSpectrogram(sample_rate=self.sr,
                                              n_mels=self.n_mels,
                                              f_min=self.f_min,
                                              f_max=self.f_max,
                                              n_fft=self.n_fft)

    def create_melspec(self, waveform):
        waveform = torch.tensor(waveform).unsqueeze(0)  #We need a channel-first Torch tensor
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = torchaudio.functional.amplitude_to_DB(
            mel_spec, 
            multiplier=10.0, 
            amin=1e-10,  
            db_multiplier=0.0  
        ).squeeze(0).numpy()

        num_frames = mel_spec_db.shape[1]
        duration = waveform.shape[1] / self.sr 
        time_axis = np.linspace(0, duration, num=num_frames)
        mel_frequencies = librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)

        return mel_spec_db, time_axis, mel_frequencies

def interactive_plot(mel_spec_db,
                     mel_frequencies,
                     power,
                     segmentation,
                     chunk_duration,
                     common_nm,
                     zoo_cls):
    """Interactive plot with click-based marking, auto-spacing, and drag-to-mark functionality."""
    
    END_BUFFER = 6  
    START_BUFFER = 6
    MAX_SPACING = 12

    duration = len(power) * chunk_duration
    t_power = np.arange(len(power)) * chunk_duration
    t_seg = np.arange(len(segmentation)) * chunk_duration
    t_melspec = np.linspace(0, duration, num=mel_spec_db.shape[1])

    marked_times = []
    marked_lines = []
    dragging = False
    start_time = None
    
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_title(f'Recording of a {common_nm} ({zoo_cls})')

    img = ax.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma',
                    extent=[t_melspec[0], t_melspec[-1], mel_frequencies[0], mel_frequencies[-1]],
                    zorder=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)", color='m')
    ax.tick_params(axis='y', labelcolor='m')

    ax2 = ax.twinx()
    ax2.plot(t_power, 10 * np.log10(power), 'b', label='Power', zorder=2)  
    ax2.plot(t_seg, segmentation, 'k', label='Voice', zorder=2)
    ax2.set_ylabel("Power (dB) / Voice Detection", color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.legend(loc="upper right")

    def add_marker(time):
        if time not in marked_times:
            marked_times.append(time)
            line1 = ax.axvline(time - 6, color='gray', linestyle='--', zorder=3)  
            line2 = ax.axvline(time + 6, color='gray', linestyle='--', zorder=3)
            area = ax.axvspan(time - 6, time + 6, color='gray', alpha=0.5)
            line3 = ax.axvline(time, color='g', linestyle='--', zorder=4)
            marked_lines.append((line1, line2, line3, area))
            fig.canvas.draw_idle()

    def onclick(event):
        nonlocal dragging, start_time
        if event.inaxes is not None and event.button == 1:
            dragging = True
            start_time = round(event.xdata, 1)
            add_marker(start_time)

    def onmotion(event):
        if dragging and event.inaxes is not None:
            current_time = round(event.xdata, 1)
            if current_time > start_time:
                time_offset = current_time - start_time
                next_marker = start_time + 12 * (time_offset // 12)
                if next_marker <= duration and next_marker not in marked_times:
                    add_marker(next_marker)

    def onclick(event):
        nonlocal dragging, start_time
        if event.inaxes is not None:
            if event.button == 1:  # Left click to add markers
                dragging = True
                start_time = round(event.xdata, 1)
                add_marker(start_time)
            elif event.button == 3:  # Right click to remove the most recent mark
                if marked_times:
                    marked_times.pop()
                    line_set = marked_lines.pop()
                    for line in line_set:
                        line.remove()
                    fig.canvas.draw_idle()

    def onrelease(event):
        nonlocal dragging
        if event.button == 1:
            dragging = False
    
    def onkeypress(event):
        if event.key == 'a':
            for line_set in marked_lines:
                for line in line_set:
                    line.remove()
            marked_lines.clear()
            marked_times.clear()
            

            time_to_cover = max(0, duration - START_BUFFER - END_BUFFER)
            num_spaces = time_to_cover // MAX_SPACING + 1
            spacing = time_to_cover / num_spaces
            #spacing = time_to_cover/num_marks
            time = START_BUFFER
            while time <= (duration - END_BUFFER):
                add_marker(round(time,1))
                time += spacing
        elif event.key == ' ':  # Spacebar to clear all
            for line_set in marked_lines:
                for line in line_set:
                    line.remove()
            marked_lines.clear()
            marked_times.clear()
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('motion_notify_event', onmotion)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('key_press_event', onkeypress)

    plt.show()
    return marked_times



def mark_one_sample(filename: Path, common_nm: str, zoo_cls: str):
    try:
        wav, sr = librosa.load(filename, sr=None)  #returns a mono-channel NumPy array
    except Exception as e:
        logger.error(f"The file '{filename.name}' failed to load: {e}")
        return None

    CHUNK_DURATION = 0.1
    chunk_len  = int(CHUNK_DURATION * sr)
    specmaker = MelSpecMaker(sr=sr)
    voice_detector = VoiceDetector(chunk_len)
    mel_spec_db, time_axis, mel_frequencies = specmaker.create_melspec(wav)
    power = calc_signal_pwr(wav, chunk_len)
    voice_detections = voice_detector.detect(wav)
    # Set the audio player width to match the plot width
    display(HTML("<style>audio { width: 800px; margin-left: 35px; }</style>"))
    display(ipd.Audio(filename)) 
    marked_times = interactive_plot(mel_spec_db,
                                    mel_frequencies,
                                    power,
                                    voice_detections,
                                    CHUNK_DURATION,
                                    common_nm=common_nm,
                                    zoo_cls=zoo_cls,)
    return marked_times


def plot_duration_mix(df, threshold):
    df['duration_category'] = df['duration'].apply(lambda x: f'< {threshold} s' if x < threshold else f'> {threshold} s')
    df_counts = df.groupby(['primary_label', 'duration_category']).size().reset_index(name='count')
    df_counts = df_counts.sort_values(by='count', ascending=False)
    custom_colors = {f'< {threshold} s': 'blue', f'> {threshold} s': 'red'}

    fig = px.bar(
        df_counts, 
        x="primary_label", 
        y="count", 
        color="duration_category",
        title="Stacked Occurrences by Duration",
        labels={"duration_category": "Duration", "count": "Occurrences", "primary_label": "Label"},
        barmode="stack",
        color_discrete_map=custom_colors,
        opacity=1.0  # Ensure full opacity
    )
    fig.show()