# importing element tree
# under the alias of ET
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import Counter


# Function to detect outliers in rolling windows
def detect_and_interpolate_outliers_rolling(df, window_size, threshold):
    df_cleaned = df.copy()
    
    for col in df.columns:
        for start in range(0, len(df), window_size):
            end = start + window_size
            
            # Extract rolling window
            window_data = df[col][start:end]
            
            # Compute mean and standard deviation for this window
            mean = window_data.mean()
            std = window_data.std()
            
            # Compute Z-scores
            z_scores = (window_data - mean) / std
            
            # Identify outliers and mark as NaN
            df_cleaned.loc[start:end-1, col] = df[col][start:end].mask(np.abs(z_scores) > threshold, np.nan)
    
        # Interpolate missing values
        df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both')
    
    return df_cleaned


def motion_load(file):
    # Define the joints and special joints with position values
    joints_with_position = ["pelvis", "thorax", "head"]
    joints = [
        "pelvis", "lfemur", "ltibia", "lfoot", "ltoes",
        "rfemur", "rtibia", "rfoot", "rtoes", "thorax",
        "head", "lclavicle", "lhumerus", "lradius", "lhand",
        "rclavicle", "rhumerus", "rradius", "rhand"
    ]
    
    # Generate rotation columns first
    rotation_columns = [f"{joint}_{axis}" for joint in joints for axis in ["x", "y", "z"]]
    
    # Generate position columns at the end
    position_columns = [f"{joint}_p{axis}" for joint in joints_with_position for axis in ["x", "y", "z"]]
    
    # Combine both lists
    columns = rotation_columns + position_columns
    
    # Identify columns where all values are 0
    motion = pd.read_csv(file, delimiter=' ', names=columns)
    # zero_columns = motion.columns[(motion == 0).all()].tolist()
    zero_columns = ['ltibia_x', 'ltibia_z', 'ltoes_x', 'ltoes_z', 'rtibia_x', 'rtibia_z', 'rtoes_x', 'rtoes_z', 'lclavicle_y', 'lradius_x', 'lhand_z', 'rclavicle_y', 'rradius_x', 'rhand_z']
    # print("Columns containing only zeros:", zero_columns)
    # Remove those columns
    motion = motion.drop(columns=zero_columns)
    
    # # Define rolling window size
    # window_size = 1000
    # threshold = 3  # Z-score threshold for detecting outliers
    
    # motion_interpolated = detect_and_interpolate_outliers_rolling(motion, window_size, threshold)
    
    
    identity = [motion['pelvis_z'][0], motion['pelvis_px'][0], motion['pelvis_py'][0], motion['pelvis_pz'][0],
                motion['thorax_px'].mean(),motion['thorax_py'].mean(),motion['thorax_pz'].mean(),
                motion['head_px'].mean(),motion['head_py'].mean(),motion['head_pz'].mean()]
    
    
    # Calculate relative pelvis rotation and position
    motion['pelvis_z'] = motion['pelvis_z']- motion['pelvis_z'][0]
    motion['pelvis_px'] = motion['pelvis_px']- motion['pelvis_px'][0]
    motion['pelvis_py'] = motion['pelvis_py']- motion['pelvis_py'][0]
    motion['pelvis_pz'] = motion['pelvis_pz']- motion['pelvis_pz'][0]
    
    
    # Remove spine and head positions
    spine_and_head = ["thorax_px", "thorax_py", "thorax_pz", "head_px", "head_py", "head_pz"]
    motion = motion.drop(columns=spine_and_head)

    return motion, identity


def align_data(motion_1, motion_2, motion_3,
               gaze_1, gaze_2, gaze_3,
               word,
               waveform_1, waveform_2, waveform_3, sample_rate):

    # 🔹 Compute samples per frame (assuming 60 FPS motion & gaze data)
    samples_per_frame = sample_rate / 60  # Samples per 60 FPS frame

    # 🔹 Find the minimum frame-based length
    min_length_frames = min(len(motion_1), len(motion_2), len(motion_3), 
                            len(gaze_1), len(gaze_2), len(gaze_3), len(word))
    
    # 🔹 Compute expected waveform length for the min frame length
    expected_waveform_length = int(samples_per_frame * min_length_frames)

    # 🔹 Find actual minimum waveform length (if shorter than expected)
    min_length_samples = min(waveform_1.shape[1], waveform_2.shape[1], waveform_3.shape[1], expected_waveform_length)

    # 🔹 Adjust min_length_frames based on waveform constraints
    min_length_frames = min(int(min_length_samples / samples_per_frame), min_length_frames)

    # 🔹 Trim motion, gaze, and word dataframes
    motion_1, motion_2, motion_3 = motion_1.iloc[:min_length_frames], motion_2.iloc[:min_length_frames], motion_3.iloc[:min_length_frames]
    gaze_1, gaze_2, gaze_3 = gaze_1.iloc[:min_length_frames], gaze_2.iloc[:min_length_frames], gaze_3.iloc[:min_length_frames]
    word = word.iloc[:min_length_frames]

    # 🔹 Trim waveforms
    waveform_1 = waveform_1[:, :min_length_samples]
    waveform_2 = waveform_2[:, :min_length_samples]
    waveform_3 = waveform_3[:, :min_length_samples]

    return motion_1, motion_2, motion_3, gaze_1, gaze_2, gaze_3, word, waveform_1, waveform_2, waveform_3


# def align_data(motion_1, motion_2, motion_3,
#                gaze_1, gaze_2, gaze_3,
#                word,
#                waveform_1, waveform_2, waveform_3, sample_rate):

#     # Convert to NumPy arrays early if they aren't already
#     motion_1, motion_2, motion_3 = map(np.asarray, (motion_1, motion_2, motion_3))
#     gaze_1, gaze_2, gaze_3 = map(np.asarray, (gaze_1, gaze_2, gaze_3))
#     word = np.asarray(word)  # Assuming words are stored as strings

#     # Find the minimum length across all data types
#     min_length = min(len(motion_1), len(motion_2), len(motion_3), 
#                      len(gaze_1), len(gaze_2), len(gaze_3), len(word))

#     # Compute expected waveform sample count for the min_length
#     samples_per_frame = sample_rate / 60  # Samples per 60 FPS frame
#     expected_waveform_length = int(samples_per_frame * min_length)

#     # Trim motion, gaze, and word data (use NumPy slicing)
#     motion_1, motion_2, motion_3 = motion_1[:min_length], motion_2[:min_length], motion_3[:min_length]
#     gaze_1, gaze_2, gaze_3 = gaze_1[:min_length], gaze_2[:min_length], gaze_3[:min_length]
#     word = word[:min_length]

#     # Trim waveforms (assuming they are PyTorch tensors)
#     waveform_1 = waveform_1[..., :expected_waveform_length]
#     waveform_2 = waveform_2[..., :expected_waveform_length]
#     waveform_3 = waveform_3[..., :expected_waveform_length]

#     return motion_1, motion_2, motion_3, gaze_1, gaze_2, gaze_3, word, waveform_1, waveform_2, waveform_3

# Function to chunk data with given window and stride
def chunk_dataframe(df, window_size, stride):
    chunks = [df.iloc[i:i+window_size] for i in range(0, len(df) - window_size + 1, stride)]
    return chunks


def chunk_array(arr, window_size, stride):
    """
    Chunks a NumPy array into overlapping windows.

    Parameters:
        arr (np.ndarray): Input NumPy array of shape (T, ...)
        window_size (int): The size of each window.
        stride (int): The step size between windows.

    Returns:
        list of np.ndarray: A list of chunks of shape (window_size, ...)
    """
    return [arr[i:i + window_size] for i in range(0, len(arr) - window_size + 1, stride)]


def chunk_waveform(waveform, sample_rate, window_size_sec, stride_sec):
    window_size_samples = int(sample_rate * window_size_sec)
    stride_samples = int(sample_rate * stride_sec)
    chunks = [waveform[:, i:i+window_size_samples] for i in range(0, waveform.shape[1] - window_size_samples + 1, stride_samples)]
    return chunks


def interp_nans_1d(values):
    """
    values: 1D NumPy array (shape (T,)) with possible NaNs.
    returns: 1D array with NaNs replaced by linear interpolation.
    
    If everything is NaN, we do nothing (remain NaN).
    Leading/trailing NaNs get extrapolated based on the first/last valid points.
    """
    x = np.arange(len(values))
    valid_mask = ~np.isnan(values)

    # If all are NaN, can't interpolate
    if not valid_mask.any():
        return values

    # Interpolate (or extrapolate) for missing points
    values[~valid_mask] = np.interp(x[~valid_mask], x[valid_mask], values[valid_mask])
    return values

def interpolate_gaze_nans(gaze_data):
    """
    gaze_data: shape (B, T, 2) with possible NaNs
    We'll do an in-place interpolation along T for each batch b and feature f.
    """
    B, T, F = gaze_data.shape  # e.g. 16, 600, 2
    for b in range(B):
        for f in range(F):
            # slice is shape (T,) (one dimension in time)
            arr_1d = gaze_data[b, :, f]
            arr_1d = interp_nans_1d(arr_1d)  # fill NaNs
            gaze_data[b, :, f] = arr_1d
    return gaze_data


def compute_minmax(data: np.ndarray, axes):
    """
    data: np.ndarray
    axes: tuple of ints over which to reduce (min/max)
    returns:
      data_min: np.ndarray of shape (feature_dim,)
      data_max: np.ndarray of shape (feature_dim,)
    """
    data_min = data.min(axis=axes)   # or np.amin(data, axis=axes)
    data_max = data.max(axis=axes)   # or np.amax(data, axis=axes)
    return data_min, data_max

def minmax_scale(data: np.ndarray, data_min: np.ndarray, data_max: np.ndarray, eps=1e-8):
    """
    Per-feature min-max scaling for NumPy arrays.
    data: shape (..., feature_dim)
    data_min, data_max: shape (feature_dim,)
    eps: small constant to avoid division by zero if data_max==data_min
    returns scaled_data in [0,1], same shape as 'data'
    """
    return (data - data_min) / (data_max - data_min + eps)

def minmax_invert(scaled_data: np.ndarray, data_min: np.ndarray, data_max: np.ndarray):
    """
    Invert the min-max scaling to get back the original range.
    scaled_data: shape (..., feature_dim)
    data_min, data_max: shape (feature_dim,)
    returns unscaled data, same shape as scaled_data
    """
    return scaled_data * (data_max - data_min) + data_min

def load_feature_from_npy(filename):
    """Loads a NumPy array from an .npy file."""
    feature_array = np.load("Training/"+filename)
    print(f"✅ Loaded: {filename} with shape {feature_array.shape}")
    return feature_array

def majority_vote(arr):
    """Find the majority element in an array using Counter."""
    counter = Counter(arr)
    return counter.most_common(1)[0][0]

# def downsample_3d(data, group_size=9):
#     """
#     Downsample a 3D array of shape (size, 450, 2) to (size, 90, 2)
#     by finding the majority speaker and word within each group of 5 rows.
#     """
#     size, num_rows, num_cols = data.shape
#     num_groups = num_rows // group_size  # Should be 50

#     downsampled = np.empty((size, num_groups, num_cols), dtype=object)  # Initialize output array

#     for i in range(size):
#         for j in range(num_groups):
#             group = data[i, j * group_size: (j + 1) * group_size]
#             downsampled[i, j, 0] = majority_vote(group[:, 0])  # Majority speaker
#             downsampled[i, j, 1] = majority_vote(group[:, 1])  # Majority word

#     return downsampled

def downsample_3d(data, group_size=9):
    """
    Downsample a 3D array of shape (size, 450, 2) to (size, 90, 2)
    by finding the majority gaze direction within each group of 9 rows.
    """
    size, num_rows, num_cols = data.shape
    num_groups = num_rows // group_size  # Should be 50

    downsampled = np.empty((size, num_groups, num_cols), dtype=object)  # Initialize output array

    for i in range(size):
        for j in range(num_groups):
            group = data[i, j * group_size: (j + 1) * group_size]
            for k in range(num_cols):               
                downsampled[i, j, k] = majority_vote(group[:, k])  


    return downsampled
    
def parse_xml_to_dataframe_word(xml_file):
    """Parse XML file and return a pandas DataFrame."""

    # Passing the path of the xml document to enable the parsing process
    # and getting the parent tag of the xml document
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for child in root:
        if child.tag == 'w':
            row = {}
            row['starttime'] = child.attrib['starttime']
            row['endtime'] = child.attrib['endtime']
            row['word'] = child.text
            data.append(row)

    df = pd.DataFrame(data)
    return df

def parse_xml_to_dataframe_segment(xml_file):
    """Parse XML file and return a pandas DataFrame."""

    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for child in root:
        row = {}
        row['starttime'] = child.attrib['transcriber_start']
        row['endtime'] = child.attrib['transcriber_end']
        data.append(row)

    df = pd.DataFrame(data)
    return df


def time_to_frame(df_a, df_b, df_c, df_d):
    # Frame rate and duration of each frame in seconds
    frame_rate = 30  # 30 frames per second
    frame_duration = 1 / frame_rate  # 33.33 milliseconds per frame
    
    # Determine the total number of frames
    total_time = max(float(max(df_a["endtime"])),float(max(df_b["endtime"])),float(max(df_c["endtime"])),float(max(df_d["endtime"])))  # Total duration in seconds
    total_frames = int(total_time / frame_duration) + 1
    
    # Create a DataFrame with all frames
    speaking_status = np.zeros([4,total_frames], dtype=int)  # Initialize all frames as "not speaking"

    list_of_df = [df_a,df_b,df_c,df_d]
    
    # Update speaking status based on intervals
    for index, df in enumerate(list_of_df):
        for _, row in df.iterrows():
            start_frame = int(float(row["starttime"]) / frame_duration)
            end_frame = int(float(row["endtime"]) / frame_duration)
            speaking_status[index][start_frame:end_frame + 1] = 1  # Mark frames as "speaking"
    
    # Create the new DataFrame
    frame_df = pd.DataFrame({
        "A": speaking_status[0],
        "B": speaking_status[1],
        "C": speaking_status[2],
        "D": speaking_status[3]
    })
    return frame_df


def generate_speaking_state(df):
    df.index.name = 'frame'
    
    # Initialize a column for the speaker label
    df['Speak'] = None
    
    # Track start times and whether each speaker is currently speaking
    speakers = ['A', 'B', 'C', 'D']
    start_time = {spk: -1 for spk in speakers}       # last frame they started speaking
    is_speaking = {spk: False for spk in speakers}  # whether speaker is speaking right now
    
    for i, row in df.iterrows():
        for spk in speakers:
            # If in this frame `spk` has a 1 but wasn't speaking before, mark start
            if row[spk] == 1 and not is_speaking[spk]:
                start_time[spk] = i
                is_speaking[spk] = True
            # If in this frame `spk` has a 0 but was speaking before, mark stop
            elif row[spk] == 0 and is_speaking[spk]:
                is_speaking[spk] = False
    
        # Now figure out who "has the floor" at frame i:
        #   (a) If exactly one speaker is on, it’s that one.
        #   (b) If multiple are on, pick who has the latest (largest) start_time.
        currently_on = [spk for spk in speakers if is_speaking[spk]]
    
        if len(currently_on) == 0:
            df.at[i, 'Speak'] = None
        elif len(currently_on) == 1:
            df.at[i, 'Speak'] = currently_on[0]
        else:
            # among all speakers speaking now, pick the one who started most recently
            latest_starter = max(currently_on, key=lambda spk: start_time[spk])
            df.at[i, 'Speak'] = latest_starter
    
    print(df)

def generate_speaking_state_three(df):
    df.index.name = 'frame'
    
    # Initialize a column for the speaker label
    df['Speak'] = None
    
    # Track start times and whether each speaker is currently speaking
    speakers = ['A', 'B', 'C']
    start_time = {spk: -1 for spk in speakers}       # last frame they started speaking
    is_speaking = {spk: False for spk in speakers}  # whether speaker is speaking right now
    
    for i, row in df.iterrows():
        for spk in speakers:
            # If in this frame `spk` has a 1 but wasn't speaking before, mark start
            if row[spk] == 1 and not is_speaking[spk]:
                start_time[spk] = i
                is_speaking[spk] = True
            # If in this frame `spk` has a 0 but was speaking before, mark stop
            elif row[spk] == 0 and is_speaking[spk]:
                is_speaking[spk] = False
    
        # Now figure out who "has the floor" at frame i:
        #   (a) If exactly one speaker is on, it’s that one.
        #   (b) If multiple are on, pick who has the latest (largest) start_time.
        currently_on = [spk for spk in speakers if is_speaking[spk]]
    
        if len(currently_on) == 0:
            df.at[i, 'Speak'] = None
        elif len(currently_on) == 1:
            df.at[i, 'Speak'] = currently_on[0]
        else:
            # among all speakers speaking now, pick the one who started most recently
            latest_starter = max(currently_on, key=lambda spk: start_time[spk])
            df.at[i, 'Speak'] = latest_starter
    
    print(df)

def get_current_word(df_speaker, t):
    """
    Given a speaker's DataFrame with columns:
      - 'starttime'
      - 'endtime'
      - 'word'
    and a time t (in seconds),
    return the single word active at time t, or None if none.
    """
    df_speaker['starttime'] = df_speaker['starttime'].astype(float)
    df_speaker['endtime']   = df_speaker['endtime'].astype(float)
    # Filter rows where starttime <= t < endtime
    active_rows = df_speaker[(df_speaker['starttime'] <= t) & (t < df_speaker['endtime'])]
    if len(active_rows) == 1:
        return active_rows.iloc[0]['word']
    else:
        # Either no word or overlapping words
        return None


def get_speak_word(row):
    spk = row['Speak']
    if spk == 'A':
        return row['word_A']
    elif spk == 'B':
        return row['word_B']
    elif spk == 'C':
        return row['word_C']
    elif spk == 'D':
        return row['word_D']
    else:
        return None



def segment_data(data, window_size, step_size):
    """
    Segments data into sliding windows.

    Parameters:
    - data: The input data as a NumPy array (e.g., shape [num_frames, features]).
    - window_size: The size of each window in frames.
    - step_size: The step size for sliding windows in frames.

    Returns:
    - A list of segmented windows as NumPy arrays.
    """
    windows = []
    for start_idx in range(0, len(data) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window = data[start_idx:end_idx]
        windows.append(window)
    return np.array(windows)

### To clean our single person word data
def ours_word_clean(file):
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    # Convert column to datetime
    df['start'] = pd.to_datetime(df['start'], format='%H:%M:%S.%f')
    df['end'] = pd.to_datetime(df['end'], format='%H:%M:%S.%f')
    # Convert to seconds
    df['starttime'] = df['start'].dt.minute * 60 + df['start'].dt.second + df['start'].dt.microsecond / 1_000_000
    df['endtime'] = df['end'].dt.minute * 60 + df['end'].dt.second + df['end'].dt.microsecond / 1_000_000
    df = df.drop(columns = ['id', 'SID', 'start', 'end'])

    return df