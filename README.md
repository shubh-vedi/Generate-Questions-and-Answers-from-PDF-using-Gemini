# Generate Questions and Answers from PDF using Gemini

Generate Questions and Answers from PDF using Gemini is a Streamlit application that allows you to generate questions and answers from PDF files using the ChatGoogleGenerativeAI model from LangChain. It extracts text from the uploaded PDF files, splits it into chunks, creates a vector store, and then generates three types of questions and answers based on the given text: True or False questions, Multiple Choice Questions (MCQs), and One-word answer questions.

## Features

- Upload multiple PDF files
- Extract text content from PDF files
- Split text into chunks for efficient processing
- Create a vector store using FAISS
- Generate questions and answers using the ChatGoogleGenerativeAI model
- Display generated questions and answers in the Streamlit app

## Prerequisites

- Python 3.x
- Google API Key (for using the ChatGoogleGenerativeAI model)
- Required Python packages (listed in the `requirements.txt` file)

## Installation

1. Clone the repository:
    ```bash
   git Clone https://github.com/shubh-vedi/Generate-Questions-and-Answers-from-PDF-using-Gemini.git
    ```

2. Navigate to the project directory:
    ```bash
    cd pdf-question-answer-generator
    ```

3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

5. Create a `.env` file in the project directory and add your Google API Key:
    ```plaintext
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. The application will open in your default web browser.
3. Upload the PDF files you want to generate questions and answers from by clicking the "Upload your PDF Files and Click on the Submit & Process Button" button in the sidebar.
4. Once the PDF files are uploaded, click the "Submit & Process" button to process the files.
5. After processing is complete, you can click the "Generate Q&A" button to generate questions and answers based on the uploaded PDF files.
6. The generated questions and answers will be displayed on the page.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

