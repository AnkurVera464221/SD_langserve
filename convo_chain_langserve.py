from fastapi import FastAPI, Request
from langserve import add_routes
import os
import uvicorn

# import streamlit as st
from dotenv import load_dotenv
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
#from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


import torch
# from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline

from pinecone import Pinecone, PodSpec, ServerlessSpec
from pinecone import Config

import os
from langchain.vectorstores import Pinecone as Pineconestore


embed = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

# SYSTEM_PROMPT = "You are a friendly and helpful assistant providing information about AMP Impact. If the user greets you, respond with a greeting and offer assistance. Otherwise, focus on answering the question directly using the provided context.  Be factual, avoid opinions, and explain if a question is unclear. If you cannot answer, say 'I'm still learning about that. Can you rephrase the question or provide more context?'"

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you do not know the answer to a question, please do not share false information.".strip()
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Use the following pieces of context to answer the question at the end. Always answer as helpfully as possible, while being safe. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you do not know the answer to a question, please do not share false information."
DEFAULT_SYSTEM_PROMPT_2 = "You are a helpful, respectful and honest assistant. Always answer what is asked and be to the point. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you do not know the answer to a question, please do not share false information."
# SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
NEW_SYSTEM_PROMPT = "You are a helpful, respectful and honest chat assistant. Only answer what is asked. Don't add anything extra to the answer."
JIRA_SYSTEM_PROMPT = """
You are a helpful, respectful, and honest assistant, specially designed to utilize a comprehensive retrieval database that includes a detailed knowledge base and data from previous Jira tickets. Use these pieces of context to answer questions and solve issues effectively. Always answer as helpfully as possible while ensuring safety and accuracy. If a question does not make any sense, or is not factually coherent, explain why instead of providing incorrect information. If you do not know the answer to a question, or if it falls outside the scope of our database, please do not share false information. Ensure all responses adhere to ethical standards, avoiding harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Strive for socially unbiased and positive interactions.
""".strip()
AMP_JIRA_SYSTEM_PROMPT = """
You are a helpful, respectful, and honest assistant to help answer questions and solve issues. Use these pieces of context to answer questions at the end and solve issues effectively. Always answer as helpfully as possible while ensuring safety and accuracy. If a question does not make any sense, or is not factually coherent, explain why instead of providing incorrect information. If you do not know the answer to a question, or if it falls outside the scope of our database, please do not share false information.
""".strip()
ASSISTANT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. You your knowledge and always reply as helpfully as possible, while being safe. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you do not know the answer to a question, please do not share false information.".strip()
FOLLOWUP_QUESTION_PROMPT = "You are a helpful, respectful and honest assistant. Always stick to the latest user input and dont add anything extra to your response. Instead of assuming anything ask followup questions to confirm everything with the user. If anything is outside of your knowledge or you do not know the answer to a question, please do not share false information."

# Initialize Pinecone
PINECONE_API_KEY ='8bbb5041-527f-49a3-a009-8f8a88e59a4e'
pc = Pinecone(api_key=PINECONE_API_KEY)

# import os
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
# embed = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

text_field = "text"
index_name = 'service-delivery-kb'
index = pc.Index(index_name)

vectorstore = Pineconestore(
    index, embed, text_field, namespace="AMP_KB_plus_Jira_with_source_links"
)


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        {prompt} [/INST]
        """.strip()

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.1, "max_length":1024})
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.1, "max_new_tokens":512, "return_full_text" : False}, huggingfacehub_api_token="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")
    # llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.1})


    template = generate_prompt(
        """
    YOUR KNOWLEDGE:
    {context}
    CHAT HISTORY:
    {chat_history}

    LATEST USER INPUT: {question}

    YOUR REPLY:
    """,
        system_prompt=FOLLOWUP_QUESTION_PROMPT,
    )

    # template = generate_prompt(
    #     """
    # CONTEXT:
    # {context}

    # QUESTION:
    # {question}

    # CHAT HISTORY:
    # {chat_history}

    # ANSWER:
    # """,
    #     system_prompt=DEFAULT_SYSTEM_PROMPT
    # )
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, llm=llm, input_key='question', output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        # max_tokens_limit=1024,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
        rephrase_question=False,
        return_source_documents=True,
        return_generated_question=True,
        output_key='answer',
    )
    return conversation_chain

convo_chain = get_conversation_chain(vectorstore)

def handle_userinput(user_question, chain=convo_chain):
    #get responce by using conversation chain
    # with st.chat_message("user"):
    #     st.markdown(user_question)
    # # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": user_question})

    response = chain({'question': user_question})
    # response = st.session_state.conversation({'question': user_question})
    print(response)


    # with st.chat_message("assistant"):
    #     st.markdown(response["answer"])
    # # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    # # st.write(st.session_state.messages)

    source_metadatas = [source.metadata for source in response['source_documents']]
    # source_docs_links = list(set(source_docs_links))
    source_docs_links = set()

    # generated_question = response["generated_question"]
    # with st.chat_message("assistant"):
        # Initial assistant response
    assistant_response = response["answer"]
    # Check if there are source documents to display
    # assistant_response += "\n\n**Generated question:** \n"
    # assistant_response += f"{generated_question}\n"
    if source_metadatas:
        # Append a header for sources
        assistant_response += "\n\n**Sources:**\n"
        # Iterate through each source document link
        for idx, metadata in enumerate(source_metadatas, start=1):
            # Assume 'title' and 'url' are keys in the source document's metadata
            # Format: 1. [Title](URL)
            source_link = metadata['source_link']
            if source_link not in source_docs_links:
                source_docs_links.add(source_link)
                source = metadata['source']
                if source[0] == "p":
                    source = source[10:-18]
                else:
                    last_slash_index = source_link.rfind('/')
                    source = source_link[last_slash_index + 1:]
                assistant_response += f"{idx}. [{source}]({source_link})\n"
        # Display the formatted message with source links
    return assistant_response
    # st.session_state.messages.append({"role": "assistant", "content": assistant_response})


# convo_chain is the major conversation chain
#Remove the following to remove the Langserve part

#LangServe Code

app = FastAPI(
    title="SD Chatbot",
    description="Vera SD Chatbot Trained on AMP KB and Jira",
    version="0.1.0"
)

# add_routes(app, convo_chain, enable_feedback_endpoint=True)
add_routes(app, handle_userinput, enable_feedback_endpoint=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

