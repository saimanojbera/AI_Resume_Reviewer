import streamlit as st
import openai
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load model, vectorizer, and label encoder
try:
    model = joblib.load("resume_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    st.error("Model, vectorizer, or label encoder file not found. Ensure all required files are in the current directory.")
    st.stop()

# Set OpenAI API key (use environment variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Set it as an environment variable 'OPENAI_API_KEY'.")
    st.stop()

# Function to generate feedback
def generate_feedback(resume_text, predicted_category_label, specified_role=None):
    role_context = f"The role specified is '{specified_role}'." if specified_role else ""
    prompt = f"""
    You are an AI expert reviewing resumes. The candidate's resume is categorized as '{predicted_category_label}'.

    {role_context}

    Analyze the following resume and provide actionable feedback on:
    1. Missing skills or keywords for a {predicted_category_label} professional.
    2. Areas for improvement in content and structure.
    3. Suggestions to make the resume more appealing.

    Resume Content:
    {resume_text}

    Provide feedback clearly and concisely.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI resume reviewer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,  # Increased token limit
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"].strip()

# Streamlit UI
st.title("AI-Powered Resume Reviewer")

# Allow users to specify job role or industry
specified_role = st.text_input("Optional: Specify the job role or industry to tailor feedback")

uploaded_files = st.file_uploader("Upload your resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        try:
            if uploaded_file.name.endswith(".txt"):
                resume_text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".pdf"):
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                resume_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            elif uploaded_file.name.endswith(".docx"):
                from docx import Document
                doc = Document(uploaded_file)
                resume_text = "\n".join([p.text for p in doc.paragraphs])
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue
        except Exception as e:
            st.error(f"Error processing the file {uploaded_file.name}: {e}")
            continue

        st.text_area(f"Uploaded Resume: {uploaded_file.name}", resume_text, height=300)

        # Predict the category and map to label
        try:
            resume_vectorized = vectorizer.transform([resume_text])
            predicted_category_index = model.predict(resume_vectorized)[0]
            predicted_category_label = label_encoder.inverse_transform([predicted_category_index])[0]  # Map index to label

            # Get confidence score
            prediction_probabilities = model.predict_proba(resume_vectorized)
            confidence_score = max(prediction_probabilities[0]) * 100

            # Display top 3 predicted categories
            top_3_indices = prediction_probabilities[0].argsort()[-3:][::-1]
            top_3_categories = label_encoder.inverse_transform(top_3_indices)

            st.subheader("Top Predicted Categories:")
            for i, category in enumerate(top_3_categories):
                st.write(f"{i+1}. {category} ({prediction_probabilities[0][top_3_indices[i]]*100:.2f}%)")

        except Exception as e:
            st.error(f"Error predicting category for {uploaded_file.name}: {e}")
            continue

        # Generate GPT Feedback
        try:
            feedback = generate_feedback(resume_text, predicted_category_label, specified_role)
        except Exception as e:
            st.error(f"Error generating feedback for {uploaded_file.name}: {e}")
            continue

        # Debug raw feedback
        st.text_area(f"Raw GPT Feedback (for debugging): {uploaded_file.name}", feedback, height=300)

        # Display results
        st.subheader(f"Predicted Category for {uploaded_file.name}:")
        st.write(predicted_category_label)
        st.subheader(f"Prediction Confidence for {uploaded_file.name}:")
        st.write(f"{confidence_score:.2f}%")

        # Safeguard against missing sections
        if "Areas for Improvement in Content and Structure:" in feedback and "Suggestions to Make the Resume More Appealing:" in feedback:
            st.subheader(f"Resume Feedback for {uploaded_file.name}:")
            st.markdown(f"""
            ### Missing Skills or Keywords:
            {feedback.split("Areas for Improvement in Content and Structure:")[0].strip()}

            ### Areas for Improvement in Content and Structure:
            {feedback.split("Areas for Improvement in Content and Structure:")[1].split("Suggestions to Make the Resume More Appealing:")[0].strip()}

            ### Suggestions to Make the Resume More Appealing:
            {feedback.split("Suggestions to Make the Resume More Appealing:")[1].strip()}
            """)
        else:
            st.subheader(f"Resume Feedback for {uploaded_file.name}:")
            st.write(feedback)

        # Download feedback
        st.download_button(
            label=f"Download Feedback for {uploaded_file.name}",
            data=feedback,
            file_name=f"feedback_{uploaded_file.name.split('.')[0]}.txt",
            mime="text/plain"
        )
