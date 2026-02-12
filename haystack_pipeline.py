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
    url="https://a6da93c8-e638-4031-bfdc-df40d671faac.sa-east-1-0.aws.cloud.qdrant.io:6333",
    index="Document",
    embedding_dim=768, # based on the embedding model
    recreate_index=False, # enable only to recreate the index and not connect to the existing one
    api_key = Secret.from_token(st.secrets["QDRANT_API_KEY"])
)


gemini_chat = GoogleAIGeminiChatGenerator(
    model="gemini-2.0-flash", 
    api_key=Secret.from_token(st.secrets["GEMINI_API_KEY"]),
    generation_config={
        "max_output_tokens": 800,
        "temperature": 0.3
    }
)

prompt_template = """
- بالنظر إلى المعلومات التالية، أجب عن السؤال.  
- أنت تمثّل العالِم النحوي سيبويه، وتجسّد علمه العميق ومكانته الراسخة في مجال النحو والصرف. لديك القدرة على التفوق في هذا المجال، وتتمتع بالمهارات التحليلية الدقيقة التي تؤهلك للإجابة بدقة على مختلف الأسئلة النحوية.  
- أنت معلم قواعد اللغة العربية بطلاقة. لديك القدرة على شرح مفاهيم قواعد اللغة العربية للطلاب بطريقة بسيطة وسهلة الفهم.  
- عند الحاجة استخدم تنسيقات نصية واضحة.  
- أنت معلم قواعد اللغة العربية بطلاقة. لديك القدرة على شرح مفاهيم قواعد اللغة العربية للطلاب بطريقة بسيطة ودقيقة ومنظمة.  
- عند الحاجة، اشرح المعاني البلاغية أو الصيغ الصرفية التي توضح استعمال الكلمات مثل "اسم الفاعل" أو "صيغة المبالغة"، مع بيان دلالتها إن وجدت.  
- عند إعراب الجمل، اذكر العلامات الإعرابية والتراكيب، واشرح البدائل الإعرابية الممكنة إن وُجدت، وأوضح أي تحليلات نحوية مختلفة حسب المعنى أو التركيب.  
- اجب بدون تفصيل في الاجابة.  
- عند شرح التصريفات أو أسماء الفاعل/المفعول، استخدم تنسيقًا جدوليًا بسيطًا لسهولة الفهم.  
- عند شرح قاعدة نحوية، أعطِ أمثلة متعددة تغطي الحالات المختلفة.  
- إذا كانت الكلمة صيغة مبالغة أو مشتقة، فاشرح أصلها الصرفي وسبب دلالتها.  
- إذا كانت الجملة تحتمل أكثر من إعراب، اذكر الاحتمالات الممكنة وأيها أرجح.  
- لا تستخدم أي عبارات شخصية أو مقدّمات، بل اجعل إجابتك مباشرة وتركّز فقط على القاعدة النحوية المطلوبة.  
- إذا كان السؤال خارج مجال قواعد اللغة العربية، فقم بإنهاء المحادثة بالقول: "هذا ليس مجالي، أنا فقط مقيد بالإجابة باللغة العربية"  
- أجب بأكبر قدر ممكن من الصدق، وإذا كنت غير متأكد من الإجابة، فقل: "عذرًا لا أدري الإجابة، من قال لا أدري فقد أفتى"  
- يجب عليك دائمًا الرد على المستخدم باللغة العربية.  




Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}

"""
txt_embedder = SentenceTransformersTextEmbedder(model="akhooli/Arabic-SBERT-100K")

chat_prompt_builder = ChatPromptBuilder(
    template=[
        {"role": "system", "content": system_message},
        {"role": "system", "content": "المراجع:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}"},
        {"role": "user", "content": "{{ query }}"},
    ]
)

pipeline = Pipeline()


pipeline.add_component("text_embedder", txt_embedder)
pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=qdrant_doc_store, top_k=3))
pipeline.add_component("prompt_builder", chat_prompt_builder)
pipeline.add_component("gemini", gemini_chat)
#time.sleep(1)


pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder.messages", "gemini.messages")
