import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

#load the pdf files from the path
loader = DirectoryLoader('data/',glob="*.csv",loader_cls=CSVLoader)
documents = loader.load()

#split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)
openai_api_key = "<ENTER_OPEN_API_KEY>"
#create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)
vector_store.save_local("faiss_index_react")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create OpenAI

llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)

st.title("Hex-GPT ChatBot 🧑🏽")
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Welcome to Hex-Tax-GPT. How can I help you?", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()