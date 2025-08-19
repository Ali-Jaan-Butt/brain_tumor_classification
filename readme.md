# Brain Tumor Detection using Deep Learning 🧠

This project focuses on detecting **brain tumors from MRI images** using **Convolutional Neural Networks (CNNs)** with **transfer learning (ResNet50)**. The notebook includes preprocessing, model training, and evaluation steps to classify MRI scans as **tumor** or **non-tumor**.

---

## 📂 Dataset
The dataset used is [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) available on Kaggle.  
- Images are categorized into **two classes**:  
  - `Yes` → MRI showing brain tumor  
  - `No` → MRI without tumor  

---

## ⚙️ Workflow
The notebook follows these main steps:

1. **Import Libraries**  
   Uses `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv`, `tensorflow/keras`.

2. **Data Preprocessing**  
   - Load images and labels  
   - Split into train and test sets (80/20)  
   - Image normalization and augmentation

3. **Model Architecture**  
   - Based on **ResNet50** with additional Dense and Dropout layers  
   - Activation: ReLU, Softmax  
   - Optimizer: Adam  

4. **Training**  
   - Early stopping used to prevent overfitting  
   - Data augmentation applied via `ImageDataGenerator`  

5. **Evaluation**  
   - Accuracy, Confusion Matrix, and Classification Report  

---

## 📊 Results
- Achieved **high classification accuracy** on test data  
- Clear distinction between tumor and non-tumor cases  
- Visualized performance using plots and confusion matrix  

---

## 🚀 Requirements
Make sure you have the following installed:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow pillow scikit-learn