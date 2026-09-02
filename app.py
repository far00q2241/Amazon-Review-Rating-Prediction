import streamlit as st
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Amazon Review Rating Predictor",
    page_icon="⭐",
    layout="centered"
)


# --------------------------------------------------
# Load Model and TF-IDF Vectorizer
# --------------------------------------------------

@st.cache_resource
def load_model():

    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("rating_model.pkl")

    return vectorizer, model


tfidf, model = load_model()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("⭐ Amazon Review Rating Prediction")

st.write(
    "Enter an Amazon product review below and the machine learning "
    "model will predict the review rating from 1 to 5 stars."
)


st.divider()


# --------------------------------------------------
# Review Input
# --------------------------------------------------

st.subheader("📝 Enter Your Review")

review = st.text_area(
    "Review Text",
    placeholder="Example: The product is excellent. "
                "Good quality and works perfectly.",
    height=180
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Rating", use_container_width=True):

    if not review.strip():

        st.warning("Please enter a review before making a prediction.")

    else:

        # Convert review into TF-IDF features
        review_tfidf = tfidf.transform([review])

        # Make prediction
        prediction = model.predict(review_tfidf)[0]

        st.divider()

        st.subheader("🎯 Prediction")

        # Display stars
        stars = "⭐" * int(prediction)

        st.success(
            f"Predicted Rating: {prediction} / 5"
        )

        st.write(
            f"### {stars}"
        )

        # Interpretation
        if prediction == 1:
            st.error("The model predicts a very negative review.")

        elif prediction == 2:
            st.warning("The model predicts a negative review.")

        elif prediction == 3:
            st.info("The model predicts a neutral/mixed review.")

        elif prediction == 4:
            st.success("The model predicts a positive review.")

        elif prediction == 5:
            st.success("The model predicts a very positive review.")


# --------------------------------------------------
# About the Model
# --------------------------------------------------

st.divider()

st.subheader("🤖 About the Model")

st.write(
    """
    **Machine Learning Model:** Linear Support Vector Classifier (LinearSVC)

    **Text Feature Extraction:** TF-IDF

    **Input Feature:** Review Text

    **Target:** Amazon Review Rating (1–5 stars)

    **Test Accuracy:** Approximately 68.9%

    **Macro F1 Score:** Approximately 47.2%
    """
)


# --------------------------------------------------
# Important Note
# --------------------------------------------------

st.info(
    "The dataset is highly imbalanced, with 5-star reviews representing "
    "the majority of observations. Therefore, performance varies across "
    "different rating classes."
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Amazon Review Rating Prediction | Machine Learning Portfolio Project"
)
