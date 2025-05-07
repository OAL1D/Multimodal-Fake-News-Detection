# Multimodal-Fake-News-Detection

This project explores fake news detection using **multimodal data**—both **text** and **images**. It combines text features from BERT and image features from ResNet50 to classify news as real or fake.

---

## 📌 Summary

- **Modality**: Text (headlines/titles) + Images (linked via IDs)
- **Models**:
  - BERT (`bert-base-uncased`) for text embeddings
  - ResNet50 for image embeddings
  - Random Forest for final classification
- **Current Labels**: Randomly generated (0 = fake, 1 = real) for testing purposes only

---

## 🧾 Dataset Structure

- `coda_text.csv`: Contains text titles and image IDs (no labels)
- `private_image_set/`: Folder with images (named as `<id>.jpg`)
- Dummy labels are created at runtime for experimentation

dataset link- https://drive.google.com/drive/folders/1bIsm7NvJg66m20IZIIMlPfM-AkQhCdjh?usp=sharing
---

## 🔧 How It Works

1. **Text features** extracted using the [CLS] token from BERT
2. **Image features** extracted using ResNet50 (pretrained on ImageNet)
3. **Features are concatenated** and fed into a Random Forest classifier
4. **Train/test split** and evaluation performed

---

## 📁 Project Files

Multimodal-Fake-News-Detection/
├── data/
│ ├── coda_text.csv
│ └── private_image_set/
├── multimodal_fakenews.py
├── README.md
├── requirements.txt



---

## 🚀 How to Run

1. Install required libraries:
```bash
pip install -r requirements.txt
2. Run the main script
python multimodal_fakenews.py



 🛑 Disclaimer
This version uses random labels since the original dataset has no ground-truth annotations. The accuracy is not meaningful and is only used to test the multimodal pipeline.


🛠 Future Work
Use a properly labeled dataset

Try deep multimodal fusion (e.g., transformers for vision + text)

Improve performance with hyperparameter tuning or finetuning models
