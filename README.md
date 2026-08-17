# SkinScanApp

SkinScan is a full-stack machine learning application that classifies skin lesion images into **seven diagnostic categories** using a Convolutional Neural Network (CNN).

## Highlights

* Trained on **10,000+ skin lesion images**
* Achieved **91% test accuracy** and **0.86 F1 score**
* Built a **Flask API** for image upload and model inference
* Integrated **MongoDB** for application data
* Used a **DCGAN** to generate synthetic images for data augmentation
* Returned predictions in **under 200 ms**

## Tech Stack

`Python` `TensorFlow` `Flask` `MongoDB` `CNN` `DCGAN`

## How It Works

1. User uploads a skin lesion image.
2. The Flask backend preprocesses the image.
3. The CNN predicts one of seven lesion classes.
4. The application returns the predicted diagnosis.

