
import pandas as pd
import tiktoken
from openai import OpenAI
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create OpenAI client with API key loaded from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants for the embedding model
embedding_model = "text-embedding-3-large"
embedding_encoding = "cl100k_base"
max_tokens = 8000

# Load encoding
encoding = tiktoken.get_encoding(embedding_encoding)

def split_text(text: str) -> List[str]:
    """
    Split the text into chunks based on token limit using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8050,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    try:
        return text_splitter.split_text(text)
    except Exception as e:
        print(f"An error occurred while splitting text: {str(e)}")
        return [text]

def reduce_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures OpenAI input doesn't exceed token limits by splitting overlong rows into multiple.
    """
    try:
        df["n_tokens"] = [len(encoding.encode(x)) for x in df["Text"]]

        new_rows = []
        for index, row in df.iterrows():
            if row["n_tokens"] > max_tokens:
                texts = split_text(row["Text"])
                for text in texts:
                    new_row = row.copy()
                    new_row["Text"] = text
                    new_rows.append(new_row)
            else:
                new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        new_df.drop(columns=["n_tokens"], inplace=True)

        return new_df
    except Exception as e:
        print(f"An error occurred while reducing DataFrame: {str(e)}")
        return df

def get_embedding(text: str, model: str = embedding_model) -> List[float]:
    """
    Generates an embedding for a given string using OpenAI's embedding model.
    """
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"An error occurred while creating embedding: {str(e)}")
        return None
