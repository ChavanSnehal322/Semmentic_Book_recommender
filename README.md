# Semmentic_Book_recommender

A machine learning-powered book recommendation system that uses semantic similarity, zero-shot classification, and emotion analysis to suggest books matching user queries and emotional tones.


****************************************************************************************************************************** 
Project Overview

This project:
- Loads and cleans a books dataset.
- Fills missing categories using a zero-shot classification pipeline (facebook/bart-large-mnli).
- Extracts emotional signals from book descriptions using the (j-hartmann/emotion-english-distilroberta-base) model.
- Creates embeddings with Hugging Face or Ollama models.
- Stores vector embeddings in a Chroma vector database.
- Uses semantic search to find books similar to user queries.
- Provides an interactive dashboard via Gradio.

****************************************************************************************************************************** 

 Main Features

**Zero-shot text classification**  
**Emotion detection for book descriptions**  
**Vector similarity search with sentence embeddings**  
**Interactive Gradio interface for recommendations**  
**Supports category & emotion-based filtering**

****************************************************************************************************************************** 

Project Structure

Sementic_Book_Recommender
├── gradio_dashboard.py
├── books_cleaned.csv
├── books_categories.csv
├── books_emotions.csv
├── tagged_description.txt
├── cover-not-found.jpg
├── .env
├── README.md
├── requirements.txt


****************************************************************************************************************************** 

Setup Instructions

1] Clone the repository

  > git clone https://github.com/yourusername/Sementic_Book_Recommender.git
  > cd Sementic_Book_Recommender

2] Create a virtual environment
  
  > python3 -m venv venv
  > source venv/bin/activate  # macOS/Linux
    # OR
  > venv\Scripts\activate  # Windows

3] Install dependencies
  > pip install -r requirements.txt

****************************************************************************************************************************** 

How to Run
- Generate vector database & embeddings
Edit gradio_dashboard.py to switch between OpenAI or Ollama embeddings as needed.

- Run the Gradio dashboard

  > python gradio_dashboard.py

Visit http://127.0.0.1:7860 to try the book recommender.

****************************************************************************************************************************** 

- Core Libraries:
  
Pandas: Data manipulation

Transformers (Hugging Face): Zero-shot classification & emotion detection

LangChain: Document loading, embeddings, vector stores

Chroma: Local vector database

Gradio: Interactive web app

****************************************************************************************************************************** 

Usage of libraries in project:

Load data: Cleaned CSV with book metadata.

Predict missing categories: Zero-shot classifier guesses Fiction or NonFiction.

Detect emotions: Extract dominant emotions per book description.

Store embeddings: Use Hugging Face / Ollama to create sentence embeddings.

Semantic search: Query vector store for similar books.

Serve recommendations: Gradio dashboard shows recommendations with images, titles, authors, and short summaries.
