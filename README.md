# Metaphor-Detection-using-Hybrid-Machine-Learning-Models

**Group Project – Indiana State University**  
**Team 8: Shri Padmavathi Manoharan, Anusha Gadgil, Harshithavalli**  

---

## Project Overview

This project focuses on **detecting metaphors in text** using both traditional Machine Learning and Deep Learning models. We explore how metaphor detection enhances Natural Language Processing applications such as chatbots, sentiment analysis, and machine translation.

We evaluated:
- Traditional ML models: Logistic Regression, Naive Bayes, SVM, Random Forest  
- Deep Learning models: CNN and BERT  
- Explainability: LIME for model interpretation

---

## Objective

To identify metaphorical language in text by training classifiers that can distinguish between **literal** and **figurative** usage using a labeled dataset.

---

## Models Implemented

###  Traditional Machine Learning
- Logistic Regression (TF-IDF)
- Naive Bayes (TF-IDF)
- SVM (TF-IDF)
- Random Forest (Count Vectorizer)

### Deep Learning
- **CNN**: Captures local patterns in text
- **BERT**: Uses contextual embeddings for classification

###  Explainability
- **LIME**: Highlights words contributing to metaphor prediction

---

##  Results Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.77     | 0.77      | 0.99   | 0.86     |
| Naive Bayes         | 0.79     | 0.79      | 0.97   | 0.87     |
| SVM                 | 0.79     | 0.82      | 0.95   | 0.87     |
| Random Forest       | 0.81     | 0.81      | 0.97   | 0.88     |
| **CNN**             | **0.89** | **0.91**  | 0.89   | **0.90** |
| BERT                | 0.84     | 0.85      | 0.84   | 0.84     |

---

##  Tech Stack

- Python  
- Scikit-learn  
- PyTorch  
- HuggingFace Transformers  
- NLTK  
- LIME  
- Google Colab

---

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/metaphor-detection-ml.git
   cd metaphor-detection-ml
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run experiments:
   - For ML models: `python traditional_models.py`
   - For CNN/BERT: `python deep_learning_models.py`
   - For LIME: `python lime_explainer.py`
