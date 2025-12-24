# Annotated features
This folder contains scripts that combine modalities (audio, gaze, and motion) into unified CSV annotations. Each output row encodes speaker activity, gaze target, and gestural backchannel (head nods and shakes).


## Gaze Target Calculation
We compute discrete gaze targets by combing motion files and gaze files to determine whether each participant is looking at:
- the left listener,
- the right listener, or
- neither listener.

This is performed using **[Visualization repo](https://github.com/MCMartinLee/Conversation_Demo/blob/master/Data/main.cpp)**, which integrates gaze and motion-capture data.

**Input**
- Capture date (e.g., `12-15-2021`)
- Session index (e.g., `1`)

**Output**
- `date/Session_X_gazeBehavior.txt`


## Final annotation
Use the provided [`Annotation.ipynb`](Annotation.ipynb) to compute speaker states and merge them with the gaze targets produced above. The intermediate output includes:

P1 audio | P2 audio | P3 audio | P1 Gaze | P2 Gaze | P3 Gaze

Then run [`GesturalBackchannel.ipynb`](GesturalBackchannel.ipynb) to detect head nods and shakes (binary). The final annotation contains 12 columns:

P1 audio | P2 audio | P3 audio | P1 Gaze | P2 Gaze | P3 Gaze | P1_headshaking | P1_nodding | P2_headshaking | P2_nodding | P3_headshaking | P3_nodding

## Encoding

Audio (columns 1–3)
- `0` — silent
- `1` — speaking
- `2` — backchanneling (e.g., "yeah", "oh", "um", "okay", "cool", "right", "uh")

Gaze (columns 4–6)
- `0` — not looking at either participant
- `1` — looking at the left participant
- `2` — looking at the right participant

Gestural backchannel (columns 7–12)
- `0` — not performing the gesture
- `1` — performing the gesture (head shake or nod)

