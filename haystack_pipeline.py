from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.utils import Secret
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Document
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator, GoogleAIGeminiChatGenerator
from pathlib import Path
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack import component
from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder, FastembedTextEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
#import config
import streamlit as st
#import time

qdrant_doc_store = QdrantDocumentStore(
    url="https://3d6b132b-9c57-4c08-9ced-28d76516d7f4.us-west-2-0.aws.cloud.qdrant.io",
    index="Document",
    embedding_dim=768,
    recreate_index=False,
    api_key = Secret.from_token(st.secrets["Qdrant_key"])
)


gemini_chat = GoogleAIGeminiGenerator(model="gemini-1.5-flash-8b", api_key=Secret.from_token(st.secrets["GOOGLE_API_KEY"]))

prompt_template = """
Given the following information, answer the question.
You are a helpful and fluent Arabic grammar school teacher. You have the ability to explain Arabic grammar concepts to students in a simple and easy-to-understand manner. 
You provide clear and straightforward examples to illustrate concepts.  
If the question is outside the domain of Arabic grammar, you end the conversation by stating, "This is not my domain, I am only restericted to answer in Arabic Language"
You must always answer the user in arabic.
"



Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}

"""
txt_embedder = SentenceTransformersTextEmbedder()

pipeline = Pipeline()


pipeline.add_component("text_embedder", txt_embedder)
pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=qdrant_doc_store))
pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipeline.add_component("gemini", gemini_chat)
#time.sleep(1)


pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "gemini")
