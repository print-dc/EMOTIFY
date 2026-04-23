# 🎭 Emotify — Real-Time Emotion & Action Recognition System

A real-time AI system that detects human emotions and predicts actions using facial expressions and pose estimation.

> Built using deep learning (CNN) and computer vision pipelines for live emotion-aware interaction.

---

## 📸 Demo

<p align="center">
  <img src="images/demo.png" alt="Emotify Demo" width="700"/>
</p>

---

## 🚀 Key Features

- Real-time facial emotion detection (7 classes)
- Pose-based action recognition (e.g., thinking, celebrating, hands raised)
- Live webcam processing using Streamlit + OpenCV
- Modular pipeline: detection → classification → action mapping
- Supports both webcam and image input

---

## 🏗️ System Architecture

Webcam/Input  
→ Face Detection (Haar Cascade)  
→ Preprocessing (48x48 grayscale)  
→ CNN Model  
→ Emotion Prediction  
→ Pose Detection  
→ Action Mapping  
→ Output  

---

## 📊 Dataset

- **FER-2013** → emotion classification  
- **MPII Human Pose** → action understanding  
- **EMOTIC dataset** → contextual emotion signals  

---

## ⚙️ Model

- CNN-based architecture trained on FER-2013  
- Input: 48×48 grayscale facial images  
- Output: Probability distribution over 7 emotions  

---

## 🧪 Evaluation

- Train/Test Split: 80/20  
- Accuracy: ~63% (baseline CNN)  
- Validated on real-time webcam inputs  

---

## 💡 Real-World Use Cases

- Emotion-aware UI systems  
- Surveillance and behavior monitoring  
- Adaptive gaming and applications  
- Human-computer interaction  

---

## ▶️ Run Locally

```bash
git clone https://github.com/print-dc/emotify
cd emotify
pip install -r requirements.txt
streamlit run app.py

### Desktop Script

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

## Deploy a Shareable Link

The fastest way to turn this into a link you can share is Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app from this repository.
4. Set the entrypoint file to `app.py`.
5. Deploy.

After the first deploy, future pushes to GitHub will automatically update the hosted app.

Important notes:

- browser camera access requires `https`
- users still have to click allow when the browser asks for camera permission
- the action layer in this app is pose-based heuristic recognition, not a separately trained action-classification model
