# importing libs
import os # Import the os module for environment variable manipulation
#import config 
import haystack_pipeline
import fitz
import pytesseract
import streamlit as st
from PIL import Image
from haystack.utils import Secret
import cv2
import numpy as np


st.title("Sebayhi - Your Arabic Educator")
st.divider()
st.caption("A chatbot for learning arabic grammar")

# Intiialize Chat/Convo history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat msgs  
for message in st.session_state.messages:
    with st.chat_message(
            message["role"]):
        st.markdown(message["content"])

# user input
if question:= st.chat_input("Greetings!"):
    # convo history to pass to the pipeline
    conversation_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages
    )
    response = haystack_pipeline.pipeline.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"template": question + conversation_history}
    })
    with st.chat_message("user"):
        st.markdown(question)  
    # Append user to chat history 
    st.session_state.messages.append({"role": "user", "content": question})
    # Display assistant msgs in chat msg container

    with st.chat_message("assistant"):
        messages=[

        ]
        clean_response = "\n\n".join(response["gemini"]["replies"])
        st.markdown(clean_response) 
    
    st.session_state.messages.append({"role": "assisant", "content": clean_response})




#chat = st.button("Chat!")
#
#if chat:
#    response = pipeline.run({
#    "text_embedder": {"text": question},
#    #"retriever": {"query_embedding": question},
#    "prompt_builder": {"template": question }
#   })
#    st.info("\n\n".join(response["gemini"]["replies"])) 



