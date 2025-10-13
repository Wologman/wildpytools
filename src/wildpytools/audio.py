import pandas as pd
from pathlib import Path
import json
from typing import Optional, Union
import xml.etree.ElementTree as ET
from tqdm import tqdm
import IPython.display as ipd
from IPython.display import display, HTML
import torch
import torchaudio
import numpy as np
import librosa
import plotly as px
import matplotlib.pyplot as plt
import torchaudio.transforms as T


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

   Prior to splitting & training, crops are made and a new ML-friendly data format is 
   created for the merged training dataset.  
   For this use the Kaggle Primary/secondary approach plus allowance for marked
   primary-call centre times

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


###############################################################################
#####Interactive tools for audio labelling and label reviewing#################
###############################################################################


class VoiceDetector():
    def __init__(self, chunk_len, threshold=0.1, no_voice=0, voice=20):
        model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                                 model='silero_vad', verbose=False)
        self.model=model
        self.chunk_len=chunk_len
        self.threshold=threshold
        self.get_stamps = get_speech_timestamps

    def detect(self, np_wav):
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
        mel_frequencies = librosa.mel_frequencies(n_mels=self.n_mels, fmin=20, fmax=14000)

        return mel_spec_db, time_axis, mel_frequencies

def interactive_plot(mel_spec_db,
                     mel_frequencies,
                     power,
                     segmentation,
                     chunk_duration,
                     common_nm,
                     zoo_cls):
    """Interactive plot with click-based marking, auto-spacing, and drag-to-mark functionality."""
    
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
            
            end_buffer = 6
            start_buffer= 6
            max_spacing = 12
            time_to_cover = max(0, duration - start_buffer - end_buffer)
            num_spaces = time_to_cover // max_spacing + 1
            spacing = time_to_cover / num_spaces
            #spacing = time_to_cover/num_marks
            time = start_buffer
            while time <= (duration - end_buffer):
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





def mark_one_sample(filename, common_nm, zoo_cls):
    wav, sr = librosa.load(filename, sr=None)  #returns a mono-channel NumPy array
    chunk_duration = 0.1
    chunk_len  = int(chunk_duration * sr)
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
                                    chunk_duration,
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