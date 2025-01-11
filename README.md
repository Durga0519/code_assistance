# AI Soft Skills Coach

AI Soft Skills Coach is an interactive web application designed to help users simulate mock interviews and receive actionable feedback to improve their communication and soft skills. The app leverages natural language processing (NLP) models and sentiment analysis to evaluate user responses and provide feedback on tone, clarity, structure, grammar, and relevance.

---

## Features

- **Dynamic Role-Based Questions:**
  - Customize your interview experience based on your role (e.g., Software Engineer, Product Manager) and experience level (Junior, Mid-Level, Senior).

- **Interactive Input Modes:**
  - Type your answers or record them using speech recognition.

- **Comprehensive Feedback:**
  - Receive actionable feedback in five key areas:
    1. **Tone:** Sentiment analysis to assess positivity or neutrality.
    2. **Clarity:** Evaluates the conciseness and completeness of your response.
    3. **Structure:** Provides a summary of your answer and assesses logical flow.
    4. **Grammar:** Detects grammatical and syntactical errors.
    5. **Relevance:** Measures semantic alignment with the ideal answer.

- **Speech Recognition:**
  - Converts spoken answers into text for analysis.

- **User-Friendly Interface:**
  - Simple and intuitive design with clear navigation.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repository/ai-soft-skills-coach.git
   cd ai-soft-skills-coach
   ```

2. **Install Dependencies:**
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Access the App:**
   Open your browser and go to:
   ```
   http://localhost:8501
   ```

---

## How to Use

1. **Set Preferences:**
   - Enter your name.
   - Select your role/domain (e.g., Software Engineer, Data Scientist).
   - Choose your experience level (0–20 years).

2. **Answer Mock Questions:**
   - Respond to the dynamically generated questions.
   - Use the typing area or speech input to provide your answers.

3. **View Feedback:**
   - Receive detailed feedback based on your responses.
   - Suggestions include areas to improve tone, grammar, and answer structure.

---

## Technologies Used

- **Streamlit:** Interactive UI for web-based applications.
- **SpeechRecognition:** Converts spoken answers into text.
- **Sentence-Transformers:** Semantic similarity and embedding generation.
- **Hugging Face Transformers:** Sentiment analysis and summarization.
- **NLTK:** Sentiment intensity analysis.
- **LanguageTool:** Grammar and syntax correction.
- **NumPy:** Mathematical operations for embeddings.

---

## File Structure

```plaintext
.
├── app.py               # Main application code
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
├── .gitignore           # Git ignore file
```

---

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- Hugging Face for their NLP models.
- Streamlit for enabling rapid application development.
- LanguageTool for robust grammar analysis.

