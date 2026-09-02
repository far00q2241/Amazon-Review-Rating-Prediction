# ⭐ Amazon Review Rating Prediction

## 📌 Project Overview

This project predicts Amazon product review ratings from **1 to 5 stars** using **Natural Language Processing (NLP)** and Machine Learning.

The model analyzes the text of an Amazon review and predicts the most likely rating.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* LinearSVC
* Joblib
* Streamlit

## 🔄 Machine Learning Pipeline

```text
Amazon Review Text
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
LinearSVC
        ↓
Predicted Rating
        ↓
1 ⭐ – 5 ⭐
```

## 🤖 Model

The project uses:

* **TF-IDF Vectorizer** for converting review text into numerical features
* **LinearSVC (Linear Support Vector Classifier)** for multi-class classification
* **Review Text** as the primary prediction feature

The model was trained using a stratified train-test split to preserve the distribution of rating classes.

## 📊 Model Performance

| Metric            |     Score |
| ----------------- | --------: |
| Test Accuracy     | **68.9%** |
| Macro F1 Score    | **47.2%** |
| Weighted F1 Score | **68.5%** |

### Classification Performance

The model performs considerably better on **5-star reviews** because the dataset contains a large number of 5-star ratings.

Performance on minority classes, particularly ratings **2, 3, and 4**, is lower.

## ⚖️ Class Imbalance

The dataset is highly imbalanced.

Five-star reviews represent the majority of observations, while ratings 2 and 3 have considerably fewer examples.

Because of this imbalance, **accuracy alone does not fully represent model performance**.

Therefore, **Macro F1 Score** is also reported to evaluate performance across all rating classes.

## 🔍 Model Limitations

The model performs significantly better on 5-star reviews than on minority rating classes.

This indicates that additional improvements may be possible through:

* Collecting more reviews for minority rating classes
* Improved text preprocessing
* Feature engineering
* Advanced NLP techniques
* Transformer-based models
* Ordinal classification approaches

## 🚀 Deployment

The trained model is deployed as an interactive **Streamlit** web application.

Users can enter an Amazon review and receive a predicted rating from **1 to 5 stars**.

## 📁 Project Structure

```text
amazon-review-rating-prediction/
│
├── app.py
├── requirements.txt
├── tfidf_vectorizer.pkl
├── rating_model.pkl
└── README.md
```

## ▶️ Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🎯 Future Improvements

* Improve minority-class prediction
* Experiment with word and character n-grams
* Try advanced NLP models
* Explore transformer-based approaches
* Investigate ordinal classification
* Improve the Streamlit user interface

## 👨‍💻 Author

**Mohammed Farooq Khan**

GitHub: `far00q2241`
