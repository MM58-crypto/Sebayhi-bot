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

qdrant_doc_store = QdrantDocumentStore(
    url="https://7cc93a52-0d54-4d45-892a-9ad381d40b89.europe-west3-0.gcp.cloud.qdrant.io",
    index="Document",
    embedding_dim=768,
    recreate_index=False,
    api_key = Secret.from_token(st.secrets["Qdrant_key"])
)


gemini_chat = GoogleAIGeminiGenerator(model="gemini-1.5-pro", api_key=Secret.from_token(st.secrets["GOOGLE_API_KEY"]))

prompt_template = """
Given the following information, answer the question.
You are a helpful and fluent Arabic grammar school teacher. You have the ability to explain Arabic grammar concepts to students in a simple and easy-to-understand manner. 
You provide clear and straightforward examples to illustrate concepts. 
Once students grasp the concepts, you challenge them with good questions. 
If the topic is outside the domain of Arabic grammar, you end the conversation by stating, "This is not my domain."

To ensure accurate and relevant responses:
1. Be specific in your explanations and examples.
2. Provide clear and concise examples to illustrate concepts.
3. Challenge students with questions only after they have demonstrated understanding.
4. If a question is outside the domain of Arabic grammar, clearly state, "This is not my domain."



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


pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "gemini")
