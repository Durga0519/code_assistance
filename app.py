import streamlit as st
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
nltk.download('vader_lexicon')

# Load pre-trained models
model = SentenceTransformer('all-MiniLM-L6-v2')
sia = SentimentIntensityAnalyzer()

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
        },
        "Product Manager": {
            "Junior": [
                "How do you gather and prioritize customer requirements?",
                "Can you describe a time you worked with cross-functional teams?",
                "What tools or frameworks do you use for product management?"
            ],
            "Mid-Level": [
                "Describe how you prioritize features for a product roadmap.",
                "How do you handle conflicts between stakeholders?",
                "What strategies do you use to ensure a product launch is successful?"
            ],
            "Senior": [
                "How do you define and communicate a product vision?",
                "What’s your approach to managing and growing a product team?",
                "How do you measure the long-term success of a product?"
            ]
        },
        "Data Scientist": {
            "Junior": [
                "What tools or libraries are you most comfortable with in data analysis?",
                "Describe a machine learning project you’ve worked on during your studies.",
                "How do you handle missing or incomplete data?"
            ],
            "Mid-Level": [
                "What’s your approach to selecting the right machine learning algorithm for a project?",
                "Describe a time you translated data insights into actionable business decisions.",
                "How do you communicate complex data findings to non-technical stakeholders?"
            ],
            "Senior": [
                "How do you lead a data science team and ensure collaboration?",
                "What’s your approach to building scalable data pipelines?",
                "How do you align data science goals with business objectives?"
            ]
        }
        # Add more roles and experience levels here
    }
    # Fallback questions if role or level is not matched
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

# Analysis and Feedback
if answer.strip():  # Ensure answer is not empty or whitespace
    # Answer Embedding Analysis
    embeddings = model.encode(answer, convert_to_tensor=True)

    # Clarity and Relevance Feedback
    clarity_score = len(answer.split()) / 50  # Basic word count ratio
    if clarity_score < 1:
        clarity_feedback = "Your answer is too short and lacks detail. Consider expanding on your points with examples or explanations."
    elif clarity_score > 3:
        clarity_feedback = "Your answer is too long and may lack focus. Try to be more concise while covering the main points."
    else:
        clarity_feedback = "Well explained! Your answer is detailed and focused."

    # Sentiment Analysis
    sentiment = sia.polarity_scores(answer)
    if sentiment['pos'] > 0.5:
        tone_feedback = "Your tone is positive and engaging. Keep it up!"
    elif sentiment['neg'] > 0.5:
        tone_feedback = "Your tone comes across as too negative. Try to rephrase certain parts to sound more optimistic."
    else:
        tone_feedback = "Your tone is neutral. Adding a bit more enthusiasm can make your response more compelling."

    # Actionable Feedback on Structure and Specific Suggestions
    structure_feedback = "Your answer could benefit from a clearer structure. Consider organizing it into three parts: an introduction, the main points, and a conclusion."
    specific_feedback = "Consider revising the following parts of your answer: \n- Opening: Does it grab attention? \n- Examples: Are they specific and relevant? \n- Closing: Does it leave a strong impression?"

    # Display Feedback
    st.subheader("Feedback")
    st.write(f"**Clarity:** {clarity_feedback}")
    st.write(f"**Tone:** {tone_feedback}")
    st.write(f"**Structure:** {structure_feedback}")
    st.write(f"**Specific Suggestions:** {specific_feedback}")
else:
    st.write("Please provide an answer to receive feedback.")

# Conclusion
st.sidebar.subheader("About")
st.sidebar.write("This app helps you practice mock interviews and improve communication skills.")
