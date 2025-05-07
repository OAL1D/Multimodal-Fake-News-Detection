import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# ===========================
# STEP 1: Load & Prepare Dataset
# ===========================
df = pd.read_csv('E:/Multimodal Fake News Detection/data/coda_text.csv')
df.columns = df.columns.str.strip()
print("Columns in the dataset:", df.columns.tolist())

# Create dummy label column if 'label' does not exist
if 'label' not in df.columns:
    print("Label column not found. Creating a dummy label column.")
    df['label'] = np.random.choice([0, 1], size=len(df))

# Add image path column
df['image_path'] = df['id'].apply(lambda x: f'E:/Multimodal Fake News Detection/data/private_image_set/{x}.jpg')

# Filter missing images
df = df[df['image_path'].apply(os.path.exists)]
print(f"Dataset size after filtering missing images: {len(df)}")

# ✅ Sample 5000 rows for faster testing
df = df.sample(n=5000, random_state=42).reset_index(drop=True)
print("Sampled subset size:", len(df))

# ===========================
# STEP 2: Load Pretrained Models
# ===========================
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===========================
# STEP 3: Feature Extraction Functions
# ===========================
def extract_bert_features(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def extract_image_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = img_transform(image).unsqueeze(0)
        with torch.no_grad():
            features = resnet(image)
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Image error: {image_path} | {e}")
        return np.zeros(1000)

# ===========================
# STEP 4: Combine Features
# ===========================
features = []
labels = []

for idx, row in df.iterrows():
    try:
        print(f"Processing row {idx}: Text: {row['clean_title']} | Label: {row['label']}")
        text_feat = extract_bert_features(row['clean_title'])
        image_feat = extract_image_features(row['image_path'])
        combined = np.concatenate((text_feat, image_feat))
        features.append(combined)
        labels.append(row['label'])
    except Exception as e:
        print(f"Skipping row {idx} due to error: {e}")

X = np.array(features)
y = np.array(labels)

# ===========================
# STEP 5: Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# STEP 6: Train Classifier
# ===========================
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ===========================
# STEP 7: Evaluation
# ===========================
y_pred = clf.predict(X_test)
print("\n=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
