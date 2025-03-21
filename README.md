# Drug Q&A System with RAG Approach

Welcome to the Drug Q&A System! This project demonstrates a question-answering system in the pharmaceutical domain using the Retrieval-Augmented Generation (RAG) method. We combine robust information retrieval techniques (TF-IDF and BM25) with a powerful language model (Llama-3.1-8B-4bit) to deliver accurate answers based on a provided PDF document.

## Project Overview

- **Data Extraction & Preprocessing:**  
  - Extract text from a PDF file using [PyPDF2](https://pypi.org/project/PyPDF2/).  
  - Split the extracted text into smaller chunks for efficient retrieval.

- **Information Retrieval:**  
  - Utilize BM25 (with optimized parameters: k1=1.6, b=0.7, epsilon=0.25) and TF-IDF to score and retrieve the most relevant text chunks.
  - Combine scores from both methods (50% BM25 + 50% TF-IDF) to rank the chunks.

- **Language Generation:**  
  - Leverage the Llama-3.1-8B-4bit model for generating precise answers.  
  - Use a carefully designed prompt that guides the model to provide concise and context-aware responses.

- **Web Interface:**  
  - A simple Gradio web interface lets users input their questions in English (or Persian) and receive instant answers.

- **Evaluation Metrics:**  
  - The system is evaluated using Exact Match (EM) and F1-Score metrics on a test dataset.
  - Example evaluation results:  
    ```
    ==================================================
    Average Exact Match (EM): 0.20
    Average F1-Score: 0.50
    ==================================================
    ```

## Installation & Setup

This project is implemented in Google Colab with versions aligned to Colab's coda 12.4 environment. To get started, install the following packages:

```bash
!pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install bitsandbytes==0.45.3
!pip install transformers==4.49.0
!pip install PyPDF2
!pip install rank_bm25
!pip install accelerate
!pip install safetensors
!pip install sentencepiece
!pip install google-colab
!pip install gradio
!pip install --upgrade gradio
```

### Step 1: Upload the PDF

Upload your `Drugs.pdf` file using the Colab file uploader. Ensure that the file is placed in the `/content/Drugs.pdf` path.

### Step 2: Run the Code

- **Extract & Preprocess Text:**  
  The script extracts text from the PDF and splits it into manageable chunks.

- **Retrieve Relevant Chunks:**  
  BM25 and TF-IDF models are applied to identify the most relevant text segments for a given query.

- **Load the Language Model:**  
  The Llama-3.1-8B-4bit model is loaded with 4-bit quantization (using BitsAndBytes) for optimal GPU performance.

- **Generate Answers & Launch Web Interface:**  
  Use the Gradio interface to interact with the system by entering your question and viewing the generated response.

- **Model Evaluation:**  
  Evaluate the system using defined test questions and compute EM and F1-Score metrics to measure performance.

## Code Structure

- **Data Extraction & Preprocessing:**  
  Functions to extract text from the PDF and clean it by removing punctuation, digits, and stopwords.

- **Retrieval Module:**  
  Functions using BM25 and TF-IDF to score text chunks, and a combined scoring mechanism to select the best candidates.

- **Answer Generation:**  
  A function that crafts a prompt for the language model and generates an answer based on the retrieved context.

- **User Interface:**  
  A Gradio-based web interface that allows easy interaction with the Q&A system.

- **Evaluation Metrics:**  
  Custom functions to calculate Exact Match (EM) and F1-Score for model evaluation.

## Launching the Web Interface

To try out the Q&A system, simply launch the Gradio interface by executing:

```python
iface.launch()
```

This opens up an interactive web page where you can enter your questions and see the model's answers in real time.

## Final Notes

- **Environment:**  
  The project is designed to run in Google Colab and uses package versions compatible with coda 12.4 Colab.

- **Quantization:**  
  The 4-bit quantization is applied using BitsAndBytes to optimize memory usage and inference speed on the GPU.

- **Feedback & Contributions:**  
  We welcome your feedback, suggestions, and contributions. Feel free to open issues or submit pull requests on GitHub.
