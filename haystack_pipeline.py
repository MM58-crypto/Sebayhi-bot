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
import time

qdrant_doc_store = QdrantDocumentStore(
    url="https://3d6b132b-9c57-4c08-9ced-28d76516d7f4.us-west-2-0.aws.cloud.qdrant.io",
    index="Document",
    embedding_dim=768,
    recreate_index=False,
    api_key = Secret.from_token(st.secrets["Qdrant_key"])
)


gemini_chat = GoogleAIGeminiGenerator(model="gemini-1.5-flash", api_key=Secret.from_token(st.secrets["GOOGLE_API_KEY"]))

prompt_template = """
بالنظر إلى المعلومات التالية، أجب عن السؤال.
أنت معلم قواعد اللغة العربية بطلاقة. لديك القدرة على شرح مفاهيم قواعد اللغة العربية للطلاب بطريقة بسيطة وسهلة الفهم. 
إذا كان السؤال خارج مجال قواعد اللغة العربية، فقم بإنهاء المحادثة بالقول: ”هذا ليس مجالي، أنا فقط مقيد بالإجابة باللغة العربية“
أجب بأكبر قدر ممكن من الصدق، وإذا كنت غير متأكد من الإجابة، فقل: ”عذرا لا ادري الاجابة من قال لا ادري فقد افتى“
يجب عليك دائمًا الرد على المستخدم باللغة العربية.
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
time.sleep(1)


pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "gemini")
