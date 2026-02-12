# importing libs
import os # Import the os module for environment variable manipulation
#import config 
import haystack_pipeline
#import fitz
#import pytesseract
import streamlit as st
#from PIL import Image
from haystack.utils import Secret
#import cv2
from haystack.dataclasses import ChatMessage
import numpy as np
import time

st.title("Sebayhi - Your Arabic Educator")
st.divider()
st.caption("A chatbot for learning arabic grammar")
#st.sidebar()

# Intiialize Chat/Convo history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat msgs  
for message in st.session_state.messages:
    with st.chat_message(
            message["role"]):
        st.markdown(message["content"])

# To stream response (i.e. typewriter effect)
def response_generator(response):

    for word in response.split():
            yield word + " "
            time.sleep(0.02)

# user input
if question:= st.chat_input("تفضل"):
    # convo history to pass to the pipeline
    conversation_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages
    )

    with st.chat_message("user"):
        st.markdown(question)  
    # Append user to chat history 
    st.session_state.messages.append({"role": "user", "content": question})
    # Display assistant msgs in chat msg container

    #full_prompt = f"{conversation_history}\nUser: {question}"
    with st.chat_message("assistant"):
            # Run pipeline
        messages = [ChatMessage.from_user("{{ question }}")]
        res = haystack_pipeline.pipeline.run(
            data={
                "text_embedder": {"text": question},
                "prompt_builder": {
                    "template": messages,
                    "template_variables": {"question": question}
                    }
            }
        )

    # Get assistant response
        assistant_response = "\n\n".join(res["gemini"]["replies"])

    # Display streamed response
        with st.chat_message("assistant"):
            st.write_stream(response_generator(assistant_response))

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})




#chat = st.button("Chat!")
#
#if chat:
#    response = pipeline.run({
#    "text_embedder": {"text": question},
#    #"retriever": {"query_embedding": question},
#    "prompt_builder": {"template": question }
#   })
#    st.info("\n\n".join(response["gemini"]["replies"])) 



