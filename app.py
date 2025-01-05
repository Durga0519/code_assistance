import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from streamlit_ace import st_ace
import time
import matplotlib.pyplot as plt

# Load environment variables from the .env file
load_dotenv()

# Configure the Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the Google Gemini API with the API key
genai.configure(api_key=GEMINI_API_KEY)

def debug_code(code: str, language: str):
    """ Debug code and provide detailed feedback on errors and fixes """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Please review the following {language} code and provide detailed debugging feedback. "
            f"Explain the errors, suggest fixes, and provide detailed explanations:\n\n{code}\n\n"
        )
        response = model.generate_content(prompt)
        feedback = response.text.strip()
        fixed_code = None
        if "Fixed Code:" in feedback:
            fixed_code = feedback.split("Fixed Code:", 1)[1].strip()
        return feedback, fixed_code
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}", None

def convert_code(code: str, source_language: str, target_language: str):
    """ Convert code from one language to another """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Convert the following {source_language} code into {target_language} code:\n\n"
            f"{code}\n\n"
            f"Provide equivalent {target_language} code with comments explaining any notable changes."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"

def analyze_complexity(code: str, language: str):
    """ Analyze space and time complexity of the code """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Analyze the space and time complexity of the following {language} code. "
            f"Suggest ways to optimize its performance:\n\n{code}\n\n"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"

def generate_practice_questions(skill_gap_data: str, language: str):
    """ Generate coding questions based on skill-gap data """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Based on the following skill-gap data:\n\n{skill_gap_data}\n\n"
            f"Generate 3 coding problems for {language} that help improve the weak areas. "
            f"Include problem statements, sample inputs/outputs, and solutions."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"

def interactive_mentoring(code: str, language: str):
    """ Provide inline comments and real-time hints for the code """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Review the following {language} code and provide inline comments explaining each block. "
            f"Focus on making complex sections easy to understand:\n\n{code}\n\n"
            f"Additionally, suggest hints for solving this code step-by-step."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"

def main():
    """ Main function for the Streamlit app """
    st.set_page_config(page_title="Code Assistance with Google Gemini API", layout="wide")
    st.title("Code Assistance with Google Gemini API")

    if not GEMINI_API_KEY:
        st.error("API key not configured. Please set the GEMINI_API_KEY environment variable.")
        return

    # Sidebar for selecting the feature
    with st.sidebar:
        st.title("Features")
        feature = st.radio(
            "Choose a feature:",
            [
                "Code Debugging",
                "Code Conversion",
                "Space & Time Complexity Analysis",
                "Skill-Gap-Based Question Generation",
                "Interactive Mentoring",
            ],
        )

        st.title("User Feedback")
        rating = st.slider("Rate the app (1-5):", 1, 5, step=1)
        comments = st.text_area("Leave your comments:")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

    if feature == "Code Debugging":
        st.subheader("Code Debugging")
        code = st_ace(language="python", theme="monokai", height=400, font_size=14)
        language = st.selectbox("Select the programming language:", ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "Other"])
        if st.button("Submit Code"):
            if not code:
                st.error("Please provide code to analyze.")
            else:
                with st.spinner("Analyzing your code..."):
                    feedback, fixed_code = debug_code(code, language)
                    st.subheader("Feedback:")
                    st.write(feedback)
                    if fixed_code:
                        st.subheader("Suggested Fixed Code:")
                        st.code(fixed_code, language=language.lower())

    elif feature == "Code Conversion":
        st.subheader("Code Conversion")
        code = st_ace(language="python", theme="monokai", height=400, font_size=14)
        source_language = st.selectbox("Select the source language:", ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "Other"])
        target_language = st.selectbox("Select the target language:", ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "Other"])
        if st.button("Convert Code"):
            if not code:
                st.error("Please provide code to convert.")
            else:
                with st.spinner("Converting your code..."):
                    converted_code = convert_code(code, source_language, target_language)
                    st.subheader("Converted Code:")
                    st.code(converted_code, language=target_language.lower())

    elif feature == "Space & Time Complexity Analysis":
        st.subheader("Space & Time Complexity Analysis")
        code = st_ace(language="python", theme="monokai", height=400, font_size=14)
        language = st.selectbox("Select the programming language:", ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "Other"])
        if st.button("Analyze Complexity"):
            if not code:
                st.error("Please provide code to analyze.")
            else:
                with st.spinner("Analyzing complexity..."):
                    analysis = analyze_complexity(code, language)
                    st.subheader("Analysis:")
                    st.write(analysis)

                    # Add dynamic visualization
                    st.subheader("Complexity Visualization")
                    complexities = {
                        "Best Case": 0.2,
                        "Average Case": 0.5,
                        "Worst Case": 1.0,
                    }
                    fig, ax = plt.subplots()
                    ax.bar(
                        complexities.keys(),
                        complexities.values(),
                        color=["green", "orange", "red"],
                    )
                    st.pyplot(fig)

    elif feature == "Skill-Gap-Based Question Generation":
        st.subheader("Skill-Gap-Based Question Generation")
        skill_gap_data = st.text_area("Enter skill-gap data:", height=150)
        language = st.selectbox("Select the programming language:", ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "Other"])
        if st.button("Generate Questions"):
            if not skill_gap_data:
                st.error("Please provide skill-gap data.")
            else:
                with st.spinner("Generating questions..."):
                    questions = generate_practice_questions(skill_gap_data, language)
                    st.subheader("Generated Questions:")
                    st.write(questions)

    elif feature == "Interactive Mentoring":
        st.subheader("Interactive Mentoring")
        code = st_ace(language="python", theme="monokai", height=400, font_size=14)
        language = st.selectbox("Select the programming language:", ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "Other"])
        if st.button("Get Mentoring"):
            if not code:
                st.error("Please provide code for mentoring.")
            else:
                with st.spinner("Providing mentoring feedback..."):
                    mentoring_feedback = interactive_mentoring(code, language)
                    st.subheader("Mentoring Feedback:")
                    st.write(mentoring_feedback)

if __name__ == "__main__":
    main()
