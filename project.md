# Smartphone-Based Fall Detection Using IMU Sensor Data

## 1. Introduction

Falls are a major safety risk in many areas, especially in elderly care, workplace safety, and situations where people work alone or in physically demanding environments. A delayed response after a fall can lead to serious health consequences, particularly if the affected person is unable to call for help.

The goal of this project is to develop a smartphone-based fall detection prototype using inertial measurement unit (IMU) sensor data. Modern smartphones contain sensors such as accelerometers and gyroscopes, which can record body movement and impact patterns during daily activities and falls. By analysing these signals, the system should be able to distinguish between normal activities of daily living and fall events.

For data collection, the project uses the Sensor Logger app to record smartphone IMU data. The planned fall classes are forward falls, backward falls, sideways falls, and trip-stumble events. In addition, normal activities such as walking, sitting, stair movement, and jumping are recorded as non-fall examples.

The technical goal is to build a machine learning pipeline that processes Sensor Logger ZIP files, extracts relevant features from the sensor signals, and predicts whether a fall occurred. If a fall is detected, the system should also estimate the fall type and the impact intensity. The prototype will be demonstrated through a Streamlit web application that allows users to upload a Sensor Logger ZIP file and view the prediction results in an understandable way.

From a business and application perspective, the project is relevant for workplace safety, health monitoring, and assisted living scenarios. For an industrial audience such as Schaeffler, smartphone-based fall detection can be understood as a lightweight and low-cost example of sensor-based safety analytics. The project also demonstrates how machine learning can be applied to real-world time-series data collected with commonly available devices.

## 2. Related Work

Fall detection using smartphones and wearable sensors has been widely researched. Most systems use inertial measurement unit (IMU) data, especially accelerometer and gyroscope signals, to distinguish between falls and normal activities of daily living. Smartphones are especially useful for this task because they already include the required motion sensors and do not require additional hardware.

Casilari et al. (2015) provide a broad overview of Android-based fall detection systems. Their work shows that smartphone sensors can be used for fall detection, but also highlights important challenges such as sensor placement, device differences, and realistic evaluation conditions. These challenges are relevant for this project because the collected data may come from different smartphones and placements.

Public fall datasets provide useful guidance for the design of this project. The SisFall dataset by Sucerquia et al. (2017) contains multiple fall types and activities of daily living and includes a structured data collection protocol with safety considerations. The MobiFall dataset by Vavoulas et al. (2014) is particularly relevant because it focuses on fall detection and classification using a smartphone. Its fall categories, such as forward, backward, and sideways falls, are similar to the fall taxonomy used in this project.

Machine learning approaches are commonly used for sensor-based fall detection. Zurbuchen et al. (2021) show that classical machine learning models such as Random Forest and Gradient Boosting can achieve strong results on wearable sensor data. For this project, these methods are more realistic than complex deep learning models because the dataset is collected by a small student team and is therefore limited in size.

Feature extraction is an important part of fall detection. Typical features include statistical measures such as mean, standard deviation, minimum, maximum, signal magnitude, acceleration peaks, and jerk. The tsfresh library described by Christ et al. (2018) is useful for extracting a large number of time-series features and can support a classical machine learning pipeline.

Based on the reviewed literature, this project follows a practical approach: smartphone IMU data is collected with the Sensor Logger app, preprocessed in Python, transformed into features, and then used for fall detection and fall-type classification. The results are presented in a Streamlit web application that allows users to upload Sensor Logger ZIP files and view the prediction output.

### References

Casilari, E., Luque, R., & Morón, M. J. (2015). Analysis of Android Device-Based Solutions for Fall Detection. Sensors, 15(8), 17827–17894. https://doi.org/10.3390/s150817827

Sucerquia, A., López, J. D., & Vargas-Bonilla, J. F. (2017). SisFall: A Fall and Movement Dataset. Sensors, 17(1), 198. https://doi.org/10.3390/s17010198

Vavoulas, G., Pediaditis, M., Chatzaki, C., Spanakis, E. G., & Tsiknakis, M. (2014). The MobiFall Dataset: Fall Detection and Classification with a Smartphone. International Journal of Monitoring and Surveillance Technologies Research, 2(1), 44–56. https://doi.org/10.4018/ijmstr.2014010103

Zurbuchen, N., Wilde, A., & Bruegger, P. (2021). A Machine Learning Multi-Class Approach for Fall Detection Systems Based on Wearable Sensors with a Study on Sampling Rates Selection. Sensors, 21(3), 938. https://doi.org/10.3390/s21030938

Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series Feature Extraction on Basis of Scalable Hypothesis Tests. Neurocomputing, 307, 72–77. https://doi.org/10.1016/j.neucom.2018.03.067

# 3. Methodology

## 3.3 Modeling and Evaluation

The classification system uses a two-stage architecture. Stage 1 is a binary classifier that decides whether a given window of sensor data represents a fall event or an activity of daily living (ADL). Stage 2 is a five-class classifier that, given a window which Stage 1 identified as a fall, predicts the fall direction (backwards, forward, left, right, or stumble). This separation reflects the operational use case: the primary question for a deployed fall-detection system is whether a fall occurred, while the fall direction is auxiliary information that informs severity estimation and post-event response.

Both stages use a Random Forest classifier with 300 trees and balanced class weighting, implemented with scikit-learn 1.5. We evaluate both stages with Leave-One-Subject-Out (LOSO) cross-validation, in which the model is trained on data from two of the three subjects and tested on the held-out third subject. LOSO is the appropriate validation protocol for this application because it directly measures how well the model generalises to a person it has never seen during training, which is the relevant condition for a deployed app. A simple random shuffle would substantially overstate performance, because windows from the same subject share strong individual movement patterns that the model can exploit at training and testing time alike.

Features are extracted from each windowed segment of sensor data. For every window we compute seventy hand-crafted features grouped into four families. The first family consists of per-axis time-domain statistics on the accelerometer and gyroscope channels, including mean, standard deviation, minimum, maximum, range, root-mean-square, skewness, and kurtosis. The second family contains signal-vector-magnitude statistics computed on the three-axis accelerometer and gyroscope signals. The third family consists of jerk features (peak, root-mean-square, and integrated absolute value of the time derivative of the acceleration magnitude), which capture the rapid change in acceleration that characterises an impact event. The fourth family contains frequency-domain energy of the accelerometer magnitude in three bands (0–3 Hz, 3–6 Hz, 6–10 Hz), computed via the fast Fourier transform. We chose hand-crafted features over automated extraction libraries such as tsfresh because the smaller, interpretable feature set is more appropriate to our sample size and easier to audit for the conference presentation.

For Stage 1, the input is a mixture of windows. Each fall recording contributes a single peak-aligned three-second window centred on the impact moment, identified by an acceleration magnitude threshold of 1.8 g. Each ADL recording contributes its full set of overlapping two-second sliding windows. We initially trained Stage 1 on sliding windows of fall recordings as well, but this introduced substantial label noise because a typical fall recording contains approximately twenty-five seconds of standing still before the fall, the fall itself lasting under one second, and several seconds of stillness afterwards. With every two-second window labelled as a fall, the model was effectively learning that "lying still" is a fall event, which produced a macro-F1 near random chance. The peak-aligned labelling resolved this issue.

Stage 1 achieves a mean LOSO macro-F1 of 0.969 across the three subjects, with per-fold scores of 1.000 (Felix held out), 0.939 (Lea), and 0.967 (Leopold). The random baseline for a binary task is 0.500. The aggregated confusion matrix shows that the classifier produces no false positives on the ADL class and approximately ten percent false negatives on the fall class, meaning that the model is conservative in its fall predictions but does not raise spurious alarms during normal activities. We consider this an appropriate operating point for a safety-critical detection system. 

![Stage 1 Confusion Matrix](docs/figures/confusion_stage1.png)

Stage 2 achieves a mean LOSO macro-F1 of 0.213 with the best of five evaluated classifier families (Random Forest, shallow Random Forest, Gradient Boosting, SVM with RBF kernel, and Logistic Regression with feature selection). The random baseline for a five-class task is 0.200. The aggregated confusion matrix shows that the model is unable to recover the `left` and `stumble` classes at all (both columns are essentially empty) and tends to predict everything as one of the other three classes. We additionally evaluated a reduced three-class setup (frontal, sideways, stumble) in which the best classifier reached a macro-F1 of 0.295 against a baseline of 0.333, again below the chance level.

![Stage 2 Confusion Matrix](docs/figures/confusion_stage2.png)

We attribute the limited generalisation of Stage 2 to three compounding factors. First, our data collection produced a strong subject-placement confounding: Felix recorded predominantly in the trouser pocket, Lea in the hand, and Leopold in the chest pocket. LOSO splits therefore cannot disentangle subject-specific from placement-specific variance, and a model trained on two placements has no information about the third when evaluated on the held-out subject. Second, the per-class sample size after splitting is small. With approximately twenty-five fall samples per class spread across three subjects, the per-fold training set contains only seventeen examples per class, which is below the threshold at which subject-invariant geometric features of fall direction become reliably learnable. Third, we observed substantial inter-subject variability in fall execution. Twenty-seven percent of Leopold's fall recordings (twelve out of forty-five) produced acceleration peaks below the 1.8 g detection threshold and could not be included in the Stage 2 dataset, whereas Felix and Lea contributed all forty of their fall recordings each. Even with a robust classifier, the underlying biomechanics differ enough between volunteers to limit cross-subject transfer at our sample size.

These findings are consistent with prior work in smartphone-based fall classification, in which inter-subject generalisation is identified as a primary failure mode (Casilari et al. 2015). The Stage 2 classifier is included in the deployed Streamlit application for completeness, but its predictions should be interpreted as auxiliary information rather than a reliable diagnostic. Reaching a deployable fall-type classification would require either substantially more subjects to average out individual biomechanics, or a deliberate multi-placement data collection protocol in which every subject contributes recordings from every phone placement, thereby decoupling the two confounded factors.

The complete inference pipeline is implemented in `src/inference.py` and is wired into the Streamlit application through a single `predict(folder_path)` function. The function returns a dictionary containing the binary fall decision, the predicted fall direction, the Stage 1 confidence score, the peak acceleration in g, and a coarse severity bucket derived from the peak acceleration. The pipeline is reproducible end-to-end: rebuilding the dataset and retraining both models on an updated set of recordings requires only two commands (`python -m src.build_dataset` followed by `python -m src.train`).

## 4. Results

_To be added by the team._

## 5. Discussion

_To be added by the team._
