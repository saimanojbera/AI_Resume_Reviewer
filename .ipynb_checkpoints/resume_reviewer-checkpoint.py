import streamlit as st
import openai
import pandas as pd
import joblib
import os

# Load model and vectorizer
try:
    model = joblib.load("resume_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    st.error("Model or vectorizer file not found. Please ensure both 'resume_model.pkl' and 'tfidf_vectorizer.pkl' are in the current directory.")
    st.stop()

# Set OpenAI API key (use environment variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Set it as an environment variable 'OPENAI_API_KEY'.")
    st.stop()

# Function to generate feedback
def generate_feedback(resume_text, predicted_category_label):
    prompt = f"""
    You are an AI expert reviewing resumes. The candidate's resume is categorized as '{predicted_category_label}'.

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
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"].strip()

# Streamlit UI
st.title("AI-Powered Resume Reviewer")
uploaded_file = st.file_uploader("Upload your resume", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Process the uploaded file
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
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        st.stop()

    st.text_area("Uploaded Resume:", resume_text, height=300)

    # Predict the category and map to label
    try:
        resume_vectorized = vectorizer.transform([resume_text])
        predicted_category_index = model.predict(resume_vectorized)[0]
        predicted_category_label = model.classes_[predicted_category_index]  # Map index to label
    except Exception as e:
        st.error(f"Error predicting category: {e}")
        st.stop()

    # Generate GPT Feedback
    try:
        feedback = generate_feedback(resume_text, predicted_category_label)
    except Exception as e:
        st.error(f"Error generating feedback: {e}")
        st.stop()

    # Display results
    st.subheader("Predicted Category:")
    st.write(predicted_category_label)

    st.subheader("Resume Feedback:")
    st.markdown(feedback)
