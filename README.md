# Machine Learning & Neural Networks Projects

This repository contains implementations of multiple projects featuring **Neural Networks and Machine Learning**, developed as part of the course *Neural-Networks-and-Deep-Learning* at the **Department of Electrical and Computer Engineering, Aristotle University of Thessaloniki**.  

The projects cover a variety of architectures and algorithms including **MLPs, CNNs, k-NN, Nearest Centroid, SVMs, RBF Networks, Hebbian Learning, and Autoencoders**.  

---

## 🧠 Project 1: Feedforward Neural Network (MLP / CNN)

- **Description:**  
  Implementation of a **feedforward neural network** (MLP, CNN, or hybrid) trained with the **backpropagation algorithm** to solve a **multi-class classification problem** (excluding MNIST).  
- **Datasets (options):**  
  - CIFAR-10 / CIFAR-100  
  - SVHN  
  - Imagenet100 / Tiny-Imagenet  
  - Any other Kaggle multi-class dataset  
- **Features:**  
  - Training with **supervised or self-supervised learning**  
  - Optional **feature extraction** (raw input, handcrafted features, or PCA)  
  - Comparison of classification accuracy across:  
    - Different hidden layer sizes  
    - Different learning parameters  
    - Different architectures (MLP vs CNN)  
  - Evaluation on both **training** and **testing** sets  
- **Comparison:**  
  Results compared with **Nearest Neighbor (k-NN)** and **Nearest Class Centroid** classifiers (from the interim assignment).  
- **Deliverables:**  
  - Source code (Python / TensorFlow / PyTorch / Keras or other language)  
  - PDF report with:  
    - Algorithm description  
    - Accuracy plots  
    - Training vs testing performance  
    - Examples of correct & incorrect classifications  
    - Runtime analysis  

---

## 📊 Interim Assignment: k-NN vs Nearest Centroid

- **Description:**  
  Implementation of a program that compares the performance of:  
  - **k-Nearest Neighbor (k=1 and k=3)**  
  - **Nearest Class Centroid**  
- **Dataset:** Same as selected for Project 1.  
- **Output:**  
  - Accuracy evaluation on training and test sets  
  - Comparative analysis  
- **Purpose:**  
  Forms the baseline for evaluating the neural network in Project 1.  

---

## 📈 Project 2: Support Vector Machines (SVM)

- **Description:**  
  Implementation of a **Support Vector Machine** classifier for multi-class classification.  
- **Datasets (options):**  
  - CIFAR-10 / SVHN  
  - Any dataset from UCI ML Repository, CMU datasets, or Kaggle  
- **Features:**  
  - **PCA-based dimensionality reduction** (retain >90% variance)  
  - Evaluation with **linear and non-linear kernels**  
  - Testing with different hyperparameters  
- **Comparison:**  
  Results compared with:  
  - k-NN (k=1,3)  
  - Nearest Class Centroid  
  - MLP with hinge loss (SVM-like optimization)  
- **Deliverables:**  
  - Source code  
  - Report with:  
    - Accuracy, confusion matrices, runtime  
    - Correct vs incorrect examples  
    - Discussion of results  

---

## 🔄 Project 3: RBF / Hebbian Learning / Autoencoders

- **Description:**  
  Implementation of one of the following:  
  - **Radial Basis Function (RBF) Network**  
  - **Hebbian Learning Network**  
  - **Autoencoder / Transformer**  
- **Possible Tasks:**  
  - Classification on CIFAR-10, SVHN, or similar  
  - **Reconstruction** (e.g., digit reconstruction on MNIST)  
  - **Function approximation**  
  - **Data generation tasks**, e.g.:  
    - Autoencoder that reconstructs the next digit (3 → 4, etc.)  
    - Neural adder: takes 2 digits as input and generates the sum as an image (e.g., 9 + 7 → 1, 6)  
- **Features:**  
  - PCA-based feature reduction (optional)  
  - Experiments with training methods:  
    - k-means initialization  
    - Random centers  
    - Different numbers of hidden neurons  
  - Evaluation of **training vs testing performance**  
- **Comparison:**  
  - RBF → compared with k-NN & Nearest Centroid  
  - Hebbian / Autoencoder → compared with PCA-based reconstruction  
  - For MNIST → reconstructed samples tested with a digit classifier  
- **Deliverables:**  
  - Source code  
  - Report with methodology, visual examples, and quantitative results  

---

## 🛠️ Tools & Environment

- **Programming Languages:** Python (recommended) or any other  
- **Libraries (optional):**  
  - [PyTorch](https://pytorch.org/)  
  - [TensorFlow](https://www.tensorflow.org/)  
  - [Keras](https://keras.io/)  
  - [scikit-learn](https://scikit-learn.org/)  
  - [NumPy](https://numpy.org/) & [Matplotlib](https://matplotlib.org/)  
