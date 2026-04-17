# Emotion and Action Prediction using Deep Learning

## Introduction

This project aims to classify emotions and predict actions based on facial expressions and enviromental analysis using deep convolutional neural networks (CNNs) and machine learning models. The system is trained on the **FER-2013** dataset for emotion detection and the **EMOTIC** and **MPII Human Pose** datasets for action prediction.

The model classifies a person's emotion into one of **seven categories** (angry, disgusted, fearful, happy, neutral, sad, and surprised) and predicts possible actions based on facial expressions and environmental context.

## Features

- **Emotion Detection**: Classifies facial expressions into seven emotions.
- **Action Prediction**: Uses mapped datasets to predict potential actions based on detected emotions.
- **Live Video Processing**: Real-time emotion and action recognition using webcam feed.
- **Deep Learning Model Optimization**: Improved accuracy through dataset enhancements and hyperparameter tuning.

## Dependencies

* Python 3
* [OpenCV](https://opencv.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [NumPy](https://numpy.org/)
* [Matplolib](https://matplotlib.org)
* [Scipy](https://scipy.org)
* [Pandas](https://pandas.pydata.org)

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## Run Locally on Windows

Use a virtual environment so the project dependencies stay isolated:

```bash
cd emotify
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To launch the webcam demo with the included `src/model.h5` file:

```bash
cd src
python emotions.py --mode display
```

Press `q` to close the camera window.

## Directory Structure
```bash
Emotify/
│── .gitignore
│── README.md
│── requirements.txt
│── imgs/
│── src/
│   │── data/  # Contains datasets (ignored in .gitignore)
        │── test/
        │── train/
│   │── images/ # Contains image samples (ignored in .gitignore)
│   │── emotions.py  # Emotion detection script
│   │── action.py  # Action prediction script
│   │── action_mapping.py
│   │── actions.txt
│   │── dataset_prepare.py  # Data preprocessing
│   │── haarcascade_frontalface_default.xml  # Haar Cascade
│   │── load_mpii.py 
│   │── model.h5  # Pre-trained model weights
│   │── mpii_annotations.csv
│   │── mpii_human_pose_v1_u12_1.mat

```
## Basic Usage

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository and enter the folder

```bash
git clone https://github.com/miracneroid/Emotify.git
cd Emotion-detection
```

* Download the FER-2013 dataset inside the `src` folder.

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

* If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1Ohtj9Zamv71mSNrjO9o_iMQuoT_nFPlQ/view?usp=share_link) and then run:  

```bash
cd src
python emotions.py --mode display
```

The webcam demo only needs `src/model.h5` and `src/haarcascade_frontalface_default.xml`. The FER-2013 dataset is only required for training mode.

* This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

![Accuracy plot](imgs/accuracy.png)

## Data Preparation (optional)

* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing.

* In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

## References

* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.

* FER2013 Dataset - Kaggle 
* MPII Human Pose Dataset

If you find any issues or need help, feel free to raise an issue or download the structured project from [Google Drive](https://drive.google.com/drive/folders/1W9JlTjq5G0kKuZV-Zmj0NSlZyzewpSCY?usp=share_link)

```bash
Let me know if you need any further modifications! 🚀
```
## Publish to Your Own GitHub

This folder already contains the original repository's `.git` history and remote.

To publish it to your own GitHub repository while keeping the existing history:

```bash
git remote remove origin
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

To start fresh with your own commit history instead:

```bash
git remote remove origin
Remove-Item -Recurse -Force .git
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
