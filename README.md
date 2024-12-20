# AI-Powered Resume Reviewer

This project is an **AI-powered resume reviewer** that leverages natural language processing (NLP) techniques and OpenAI's GPT models to provide actionable feedback on resumes. The application is designed to analyze uploaded resumes, predict their relevant job categories, and generate personalized feedback to improve their quality and effectiveness.

---
## Demo
You can try the deployed application here: [AI-Powered Resume Reviewer](https://airesumereviewer.streamlit.app/)

---

## Features

### 1. Resume Analysis
- Extracts text from resumes in `.txt`, `.pdf`, and `.docx` formats.
- Uses NLP to process resume content and identify key skills, experiences, and other details.

### 2. Job Category Prediction
- Predicts the most relevant job category using a trained machine learning model.
- Displays top three predicted categories with confidence scores.

### 3. Feedback Generation
- Provides actionable feedback on:
  - Missing skills or keywords for the predicted job category.
  - Areas for improvement in content and structure.
  - Suggestions to make the resume more appealing.

### 4. Download Feedback
- Users can download the generated feedback as a text file.

### 5. Multi-Resume Processing
- Allows uploading and analyzing multiple resumes simultaneously.

### 6. Customization
- Users can specify a job role or industry to tailor feedback more effectively.

---

## Technologies Used

### Frontend
- **Streamlit**: For building an interactive and user-friendly web interface.

### Backend
- **OpenAI GPT-3.5**: To generate feedback on resumes.
- **Python Libraries**:
  - `scikit-learn`: For machine learning.
  - `joblib`: For saving and loading the trained models.
  - `PyPDF2`: For processing `.pdf` files.
  - `python-docx`: For processing `.docx` files.
  - `pandas` and `numpy`: For data handling and preprocessing.

### Deployment
- **Streamlit Cloud**: For hosting the application online.

---

## Installation

### Prerequisites
- Python 3.8+
- An OpenAI API key

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/saimanojbera/AI_Resume_Reviewer.git
   cd AI_Resume_Reviewer
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key"  # On Windows: set OPENAI_API_KEY="your-api-key"
   ```

5. Run the application:
   ```bash
   streamlit run resume_reviewer.py
   ```

---

## Usage
1. **Upload Resumes**: Upload `.txt`, `.pdf`, or `.docx` files to the application.
2. **Specify Role/Industry (Optional)**: Tailor feedback by specifying the target job role or industry.
3. **Analyze and Review**: View the predicted job category and detailed feedback.
4. **Download Feedback**: Download the generated feedback as a text file.

---

## Project Structure
```
AI_Resume_Reviewer/
├── resume_reviewer.py        # Main application script
├── requirements.txt          # Python dependencies
├── resume_model.pkl          # Trained ML model
├── tfidf_vectorizer.pkl      # Vectorizer for text processing
├── label_encoder.pkl         # Encoder for job categories
├── README.md                 # Project documentation
```

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
---

## Acknowledgments
- [OpenAI](https://openai.com/) for GPT models
- [Streamlit](https://streamlit.io/) for simplifying web app development
