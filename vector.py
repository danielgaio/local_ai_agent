from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("motorcycle_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=str(row.to_dict()),
            metadata={"source": f"motorcycle_reviews.csv - row {row.name}"},
            id=str(i)
        )

        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="motorcycle_reviews",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})