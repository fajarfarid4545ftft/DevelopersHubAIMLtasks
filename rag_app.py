import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chains import create_retrieval_chain
from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

with open("knowledge.txt", "w") as f:
    f.write("The official colors of MUST University are blue and gold.\n")
    f.write("The AI department was established to train students in advanced machine learning and robotics.\n")

@st.cache_resource
def setup_rag_system():
    loader = TextLoader("knowledge.txt")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using only the provided context below:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, context_prompt)
    conversational_rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return conversational_rag_chain

rag_chain = setup_rag_system()

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

with_history_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

st.title("Context-Aware Help Bot")
st.write("Ask questions based on your custom document store!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask something about your data:"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    response = with_history_chain.invoke(
        {"input": user_query},
        config={"configurable": {"session_id": "user_session_1"}}
    )
    
    bot_answer = response["answer"]
    
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
    with st.chat_message("assistant"):
        st.markdown(bot_answer)
