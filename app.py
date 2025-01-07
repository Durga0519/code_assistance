import numpy as np
import streamlit as st
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK data only once
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Cache expensive operations
@st.cache_resource
def load_models():
    # Load pre-trained models
    sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    summarization_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    return sentence_transformer_model, sentiment_pipeline, summarization_pipeline, sentiment_analyzer

# Load models once
model, sentiment_analyzer_pipeline, summarizer, sia = load_models()

# UI Setup
st.title("AI Soft Skills Coach")
st.write("Simulate mock interviews and get actionable feedback to improve your communication skills!")

# User Profile Input
st.sidebar.header("Interview Preferences")
user_name = st.sidebar.text_input("Your Name", "John Doe")

# Dropdown menu for Role/Domain
roles = [
    "Software Engineer",
    "Product Manager",
    "Data Scientist",
    "Marketing Specialist",
    "Human Resources Manager",
    "Sales Executive",
    "UI/UX Designer",
    "Business Analyst"
]
role = st.sidebar.selectbox("Select Your Role/Domain", roles)

# Experience Level Input
experience = st.sidebar.slider("Years of Experience", 0, 20, 2)

# Determine experience level
if experience <= 2:
    level = "Junior"
elif 3 <= experience <= 7:
    level = "Mid-Level"
else:
    level = "Senior"

# Dynamic Question Generation
def generate_question(role, level):
    questions = {
        "Software Engineer": {
            "Junior": [
                "What projects have you worked on during your studies or internships?",
                "How do you approach learning a new programming language?",
                "Can you explain the basics of object-oriented programming?"
            ],
            "Mid-Level": [
                "Describe a challenging project you worked on and how you overcame the challenges.",
                "How do you debug a complex issue in a codebase?",
                "How do you ensure your code is maintainable and scalable?"
            ],
            "Senior": [
                "How do you mentor junior developers on your team?",
                "What is your approach to designing system architecture?",
                "How do you balance technical debt with delivering features on time?"
            ]
        }
    }
    return questions.get(role, {}).get(level, ["Tell me about yourself.", "Why should we hire you?"])

questions = generate_question(role, level)

# Display Questions
st.subheader(f"Mock Interview ({level} - {role})")
for i, q in enumerate(questions):
    st.write(f"**Question {i+1}:** {q}")

# Answer Input
st.subheader("Your Answer")
answer_mode = st.radio("Input Mode", ["Type", "Speak"])
answer = ""  # Initialize the answer variable

if answer_mode == "Type":
    answer = st.text_area("Type your answer here...")
    if st.button("Submit Answer"):
        st.write("Answer submitted.")
elif answer_mode == "Speak":
    if st.button("Record Answer"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            try:
                audio = recognizer.listen(source, timeout=10)
                answer = recognizer.recognize_google(audio)
                st.write("You said:", answer)
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
                answer = ""
            except sr.RequestError as e:
                st.error(f"Speech Recognition service error: {e}")
                answer = ""

# Grammar and Syntax Check using LanguageTool API
def get_language_tool_feedback(text):
    # LanguageTool API URL
    api_url = "https://api.languagetool.org/v2/check"
    
    # Data to be sent to the LanguageTool API
    payload = {
        'text': text,
        'language': 'en-US'
    }

    # Send POST request to the LanguageTool API
    response = requests.post(api_url, data=payload)
    
    if response.status_code == 200:
        # Parse response and collect issues
        result = response.json()
        issues = result.get('matches', [])
        
        feedback = []
        for issue in issues:
            feedback.append(f"Issue: {issue['message']}, Context: {issue['context']['text']}")
        
        if not feedback:
            feedback.append("No grammar issues detected.")
        return feedback
    else:
        return ["Error: Unable to connect to LanguageTool API."]

# Feedback using Hugging Face models and LanguageTool
def get_feedback(answer):
    # 1. Sentiment analysis feedback (tone)
    sentiment_result = sentiment_analyzer_pipeline(answer)
    sentiment = sentiment_result[0]['label']
    
    if sentiment == 'POSITIVE':
        tone_feedback = "Your tone is positive and engaging. Keep it up!"
    elif sentiment == 'NEGATIVE':
        tone_feedback = "Your tone comes across as too negative. Try to rephrase certain parts to sound more optimistic."
    else:
        tone_feedback = "Your tone is neutral. Adding a bit more enthusiasm can make your response more compelling."

    # 2. Clarity feedback (wordiness check)
    clarity_feedback = "Your answer is clear and well-structured."

    word_count = len(answer.split())
    if word_count < 50:
        clarity_feedback = "Your answer is a bit short. Consider elaborating more on your points."
    elif word_count > 150:
        clarity_feedback = "Your answer is quite long. Try to be more concise while covering the main points."

    # 3. Structure feedback (using summarization)
    summarized = summarizer(answer, max_length=50, min_length=25, do_sample=False)
    structure_feedback = f"Your answer summary: {summarized[0]['summary_text']}."

    # 4. Grammar and Syntax Check (using LanguageTool API)
    grammar_feedback = get_language_tool_feedback(answer)

    # 5. Semantic Relevance (using Sentence Transformers)
    reference_answer = "Provide an answer that directly addresses the question with specific examples."
    sentence_embeddings = model.encode([answer, reference_answer])
    cosine_similarity = np.dot(sentence_embeddings[0], sentence_embeddings[1]) / (np.linalg.norm(sentence_embeddings[0]) * np.linalg.norm(sentence_embeddings[1]))
    
    relevance_feedback = f"Your answer is semantically {cosine_similarity * 100:.2f}% relevant to the ideal answer."

    # Return all feedback
    return {
        "tone_feedback": tone_feedback,     
        "clarity_feedback": clarity_feedback,
        "structure_feedback": structure_feedback,
        "grammar_feedback": grammar_feedback,
        "relevance_feedback": relevance_feedback
    }

# Show feedback if the answer is provided
if answer.strip():
    feedback = get_feedback(answer)
    st.subheader("Feedback")
    st.write(f"**Tone:** {feedback['tone_feedback']}")
    st.write(f"**Clarity:** {feedback['clarity_feedback']}")
    st.write(f"**Structure:** {feedback['structure_feedback']}")
    st.write(f"**Grammar:** {feedback['grammar_feedback']}")
    st.write(f"**Relevance:** {feedback['relevance_feedback']}")
else:
    st.write("Please provide an answer to receive feedback.")

# Conclusion
st.sidebar.subheader("About")
st.sidebar.write("This app helps you practice mock interviews and improve communication skills.")
