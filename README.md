Drug Q&A System with RAG

This project implements a question-answering system in the pharmaceutical domain using Retrieval-Augmented Generation (RAG). It leverages BM25 and TF-IDF for retrieving relevant text chunks from a PDF document and uses the Llama-3.1-8B-4bit model for generating precise answers.
Key Features

    Data Extraction & Preprocessing:
    Extracts text from a PDF (Drugs.pdf) and splits it into manageable chunks after cleaning.

    Information Retrieval:
    Uses BM25 (k1=1.6, b=0.7, epsilon=0.25) and TF-IDF to identify the most relevant chunks based on the query, combining their scores equally.

    Answer Generation:
    The Llama-3.1-8B-4bit model generates concise and context-aware answers from the retrieved text.

    Web Interface:
    A simple Gradio interface allows users to input questions and view real-time answers.

    Evaluation:
    Model performance is evaluated using Exact Match (EM) and F1-Score metrics.

Setup Instructions

    Install Dependencies:

!pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install bitsandbytes==0.45.3 transformers==4.49.0 PyPDF2 rank_bm25 accelerate safetensors sentencepiece google-colab gradio
!pip install --upgrade gradio

Upload the PDF:
Use Colab’s file uploader to place your Drugs.pdf in /content/Drugs.pdf.

Run the Code:
Execute the notebook to extract text, retrieve relevant chunks, generate answers, and launch the Gradio web interface.
