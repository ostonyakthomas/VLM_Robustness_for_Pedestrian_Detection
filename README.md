# Seen But Not Found: Demographic Robustness of Vision Language Models for Pedestrian Detection
This repository contains the code for experiments evaluating the demographic robustness of Vision Language Models (VLMs) for pedestrian detection across age (adult vs. child) and gender (male vs. female). Two detection scenarios are assessed: recognition, which evaluates whether a VLM can confirm the presence of a pedestrian in a cropped image, and localization, which evaluates whether a VLM can predict bounding boxes for all pedestrians in a full street scene image.
The experiments use the CityPersons dataset augmented with demographic labels provided by Li et al. (2025). The dataset and label files are available in the shared Google Drive folder linked below.

All VLMs are queried through the OpenRouter API. To reproduce the experiments, place the data, label files, and scripts in the same directory and configure your OpenRouter API key before running.
Scripts
**Cropper.py** — Crops individual pedestrians from whole images with a 20-pixel padding buffer. Each crop is saved with a filename encoding the pedestrian's index, age label, and gender label for downstream label recovery.
**Query.py** — Runs the recognition experiment. Submits cropped pedestrian images to each VLM and evaluates how reliably each model confirms pedestrian presence across demographic subgroups.
**BB_Eval.py** — Runs the localization experiment. Submits whole images to each VLM and evaluates how reliably each model predicts bounding box coordinates for all visible pedestrians across demographic subgroups.
