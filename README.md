# Sebayhi - Your Arabic Educator

Sebayhi is a chatbot designed to help users learn Arabic grammar. It leverages various libraries and tools to provide an interactive learning experience.

## Features

- Interactive chatbot for learning Arabic grammar
- Utilizes Haystack for document retrieval and processing
- Uses OCR to process Arabic Grammer Books
- Integrates with Streamlit for a web-based interface


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/sebayhi.git
    cd sebayhi
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run main.py
    ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Project Structure

- [main.py](http://_vscodecontentref_/0): The main script that runs the Streamlit application.
- [config.py](http://_vscodecontentref_/1): Configuration settings for the application.
- [haystack_pipeline.py](http://_vscodecontentref_/2): Contains the pipeline for processing user inputs and generating responses.

## Dependencies

- [os](http://_vscodecontentref_/3): For environment variable manipulation
- [config](http://_vscodecontentref_/4): Custom configuration module
- [haystack_pipeline](http://_vscodecontentref_/5): Custom pipeline for processing inputs
- [fitz](http://_vscodecontentref_/6): PyMuPDF library for PDF processing
- [pytesseract](http://_vscodecontentref_/7): OCR tool for extracting text from images
- [streamlit](http://_vscodecontentref_/8): Framework for creating web applications
- [PIL](http://_vscodecontentref_/9): Python Imaging Library for image processing
- [cv2](http://_vscodecontentref_/10): OpenCV library for computer vision tasks
- [numpy](http://_vscodecontentref_/11): Library for numerical computations

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
