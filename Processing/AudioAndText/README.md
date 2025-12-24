# Audio and Text Processing

This folder contains tools to generate speech transcripts and extract prosodic features from WAV exports ([Audacity](https://www.audacityteam.org/)).



## Speech-to-Text
At the time of data collection, speech transcripts were generated using YouTube’s built-in automatic captioning system.

Workflow:
1. Upload participant audio to YouTube  
2. Download the generated `.srt` and `.vtt` caption files  
3. Parse timestamps at the word or sentence level using the provided notebook:  
   **[Captions Processing Notebook](processing/Captions.ipynb)**

---

## Prosodic Features
Prosodic features are extracted using **Praat**, including:
- Fundamental frequency (pitch)  
- Intensity (energy)  

We provide a Praat script for reproducibility:  
**[PitchAndIntensity.praat](processing/PitchAndIntensity.praat)**

