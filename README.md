# Multi-TPC
A Multimodal Dataset for Three-Party Conversations with Speech, Motion, and Gaze

# Installing
* python version >=3.8.0
```
pip install -r requirements.txt
```
## Tools

- **ViconIQ** — Motion capture and motion data processing  
- **[D-Lab](https://ergoneers.com/faq/latest-d-lab-version/)** — Gaze tracking and audio capture  
- **[Audacity](https://www.audacityteam.org/)** — Audio trimming and channel-level processing  
- **[Praat](https://www.fon.hum.uva.nl/praat/)** — Prosodic feature extraction (pitch and intensity)

---

## Data Capture
![Layout](Figures/layout.png)
### Synchronization
All modalities are synchronized using a physical clapboard instrumented with motion-capture markers.  
The clap event provides a shared temporal reference across motion, gaze, and audio streams.

---

## Data Pre-processing

### ViconIQ
Motion data are processed using **ViconIQ**, including:
- Gap interpolation  
- Temporal smoothing  

A step-by-step demonstration is available in this [video tutorial](https://youtu.be/e_mJbUDvP28).

### D-Lab
Gaze and audio data are exported from **D-Lab**.  
See this [video tutorial]()(Release soon) for the export workflow.

### Audacity
Audio files are processed using **Audacity** to:
- Trim recordings  
- Mute other participants’ voices in each individual audio track  

A demonstration is available in this [video tutorial](https://youtu.be/nyvr48YhuvU).

---

## Data processing

### Gaze alignement, anomaly detection and gap interpolation
For gaze alignement between three participants, please follow this [video tutorial](https://www.youtube.com/watch?v=LpspvewNe6o)
We detect anomalies using a z-score–based method and fill temporal gaps via cubic spline interpolation using the provided [MATLAB app](processing/GazeProcessing.mlapp).

### Gaze Angle Conversion
Pixel-based gaze targets are converted into **pitch** and **yaw** angle representations using **[Angle Convert](processing/GazeAngleConvert.ipynb)** following the formulation illustrated in the reference image:

![convert](Figures/conversion.png)

### Gaze Target Calculation
We compute discrete gaze targets to determine whether each participant is looking at:
- the left listener,
- the right listener, or
- neither listener.

This is performed using **[Visualization repo](https://github.com/MCMartinLee/Conversation_Demo/blob/master/Data/main.cpp)**, which integrates gaze and motion-capture data.

**Input**
- Capture date (e.g., `12-15-2021`)
- Session index (e.g., `1`)

**Output**
- `date/Session_X_gazeBehavior.txt`

**Output Format**

**Encoding**
- `0` — looking at neither participant  
- `1` — looking at the left participant  
- `2` — looking at the right participant  

**Example**

0 1 0

This indicates that Participant 2 is looking at the left participant, while Participants 1 and 3 are not looking at either interlocutor.

---

### Speech-to-Text
At the time of data collection, speech transcripts were generated using YouTube’s built-in automatic captioning system.

Workflow:
1. Upload participant audio to YouTube  
2. Download the generated `.srt` and `.vtt` caption files  
3. Parse timestamps at the word or sentence level using the provided notebook:  
   **[Captions Processing Notebook](processing/Captions.ipynb)**

---

### Prosodic Features
Prosodic features are extracted using **Praat**, including:
- Fundamental frequency (pitch)  
- Intensity (energy)  

We provide a Praat script for reproducibility:  
**[PitchAndIntensity.praat](processing/PitchAndIntensity.praat)**


# Analysis
1. Download the dataset and put inside the Data folder
2. Run Jupyter Notebook of example
* [Analysis](Analysis.ipynb)

# Visualization
```
git clone https://github.com/MCMartinLee/Conversation_Demo
```
