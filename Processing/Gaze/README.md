# Gaze processing
This folder contains tools for processing gaze data exported from [D-Lab](https://ergoneers.com/faq/latest-d-lab-version/). 



## Gaze alignement, anomaly detection and gap interpolation
For gaze alignement between three participants, please follow this [video tutorial](https://www.youtube.com/watch?v=LpspvewNe6o)
We detect anomalies using a z-score–based method and fill temporal gaps via cubic spline interpolation using the provided [MATLAB app](GazeProcessing.mlapp).

## Gaze Angle Conversion
Pixel-based gaze targets are converted into **pitch** and **yaw** angle representations using **[Angle Convert](GazeAngleConvert.ipynb)** following the formulation illustrated in the reference image:

![convert](../../Figures/conversion.png)


## Final format
- Gaze: The final eye gaze files is txt format with two columns: pitch and yaw angles.