

import pandas as pd  # pandas for handling CSV and DataFrame operations
import numpy as np  # numpy for numerical operations (here, to handle missing values)
import gradio as gr  # Gradio for building the interactive web dashboard

from dotenv import load_dotenv  # To load API keys or env vars from a `.env` file

# LangChain document loader, embeddings, text splitter, vector database
from langchain_community.document_loaders import TextLoader  # Loads raw text files as LangChain docs
from langchain_openai import OpenAIEmbeddings  # OpenAI embeddings for vector storage (unused, commented out)
from langchain_text_splitters import CharacterTextSplitter  # For splitting large text into chunks
from langchain_chroma import Chroma  # Vector database to store embeddings

#  Load environment variables 
load_dotenv()  # Loads any .env variables (e.g., API keys)

# Loading dataset
books = pd.read_csv("books_emotions.csv")  # Read books dataset with metadata & emotion scores

# Fill missing authors with an empty string to avoid errors later
books["authors"] = books["authors"].fillna("")

# Create a larger thumbnail URL (append size param for better quality images)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# If the thumbnail is missing (NaN), use a local placeholder image
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    #"cover-not-found.jpg",
    "title",
    books["large_thumbnail"],
)

#  Load and split the raw description 

# Load tagged descriptions from a text file
raw_doc = TextLoader("tagged_description.txt").load()

# Using a character-based splitter — splits on newlines, chunks of ~2000 chars
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=0)

# Actually split the document into chunks
doc = text_splitter.split_documents(raw_doc)

#  Store the chunks in a vector database

# db_books = Chroma.from_documents( doc, OpenAIEmbeddings()). .. if using openAI

from langchain.embeddings import HuggingFaceEmbeddings  # (Deprecated import — you got the warning)

# Instantiate the embeddings model to turn text chunks into vectors
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store the chunks as vectors inside a Chroma DB
db_books = Chroma.from_documents(doc, embeddings)

#  Semantic recommendation logic 

# Function: get relevant books given a query, category filter, and emotional tone
def retrieve_sementic_recomm(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    Final_top_k: int = 16,
) -> pd.DataFrame:
    # Search the vector DB for similar chunks to the query
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)

    # Extract matching ISBNs from the vector DB results
    books_list = [int(doc.page_content.strip('"').split()[0]) for doc, score in recs]

    # Filter books DataFrame for these ISBNs
    books_rec = books[books["isbn13"].isin(books_list)].head(Final_top_k)

    # Further filter by category if one is selected
    if category != "All":
        books_rec = books_rec[books_rec["categories"] == category].head(Final_top_k)
    else:
        books_rec = books_rec.head(Final_top_k)

    # Sort by emotional tone if selected
    if tone == "Happy":
        books_rec.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_rec.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_rec.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Sad":
        books_rec.sort_values(by="sadness", ascending=False, inplace=True)

    return books_rec

#  Function to display results on Gradio 

def recommend_books(query: str, category: str, tone: str):
    # Get DataFrame of recommended books
    recommendatn = retrieve_sementic_recomm(query, category, tone)

    res = []  # List to store image+caption tuples for the Gradio Gallery

    for _, row in recommendatn.iterrows():
        descriptn = row["description"]

        # Shorten long descriptions to first 30 words
        truncated_desc_split = descriptn.split()
        truncated_descriptn = " ".join(truncated_desc_split[:30]) + "..."

        # Format authors: handle cases with multiple authors separated by ":"
        authors_split = row["authors"].split(":")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create a caption: title, authors, snippet of description
        caption = f"{row['title']} by {authors_str} : {truncated_descriptn}"

        # Add tuple (image URL, caption) to result list
        res.append((row["large_thumbnail"], caption))

    return res

# Building Gradio UI 

# Prepare list of unique categories
categories = ["All"] + sorted(books["categories"].unique())

# List of possible tones
tones = ["All"] + ["Happy", "Surpasing", "Angry", "Suspenseful", "Sad"]

# Create the Gradio Blocks dashboard
with gr.Blocks(theme=gr.themes.Glass) as dashboard:
    gr.Markdown("# Sementic book recommender")

    with gr.Row():
        # User input: book description query
        user_query = gr.Textbox(
            label="Please enter a description of a book: ",
            placeholder="e.g., A story about forgiveness"
        )

        # Dropdown for category filter
        category_dropdown = gr.Dropdown(
            choices=categories, label="Select a category:", value="All"
        )

        # Dropdown for tone filter
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Select an emotion tone:", value="All"
        )

        # Button to trigger search
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")

    # Display results as a Gallery (images with captions)
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    # Link button click → function → output
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )



if __name__ == "__main__":

    dashboard.launch()