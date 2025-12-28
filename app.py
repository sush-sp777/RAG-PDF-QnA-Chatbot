import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACE_TOKEN']=os.getenv("HUGGINGFACE_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#setup streamlit
st.title("Conversational RAG with PDF uploads with chat history")
st.write("Upload PDF's and chat with their content")

#input the groq api key
api_key=st.text_input("Enter your Groq API key:",type="password")

#check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model="llama-3.1-8b-instant")
    
    session_id=st.text_input("Session ID",value="Default Session")

    # Initialize chat history store
    if 'vectorstores' not in st.session_state:
        st.session_state.vectorstores = {}
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files=st.file_uploader("Choose PDF file",type="pdf",accept_multiple_files=True)
    
    #process my uploaded files
    if uploaded_files:
        file_names=sorted([f.name for f in uploaded_files])
        document_set_id="_".join(file_names)
        if document_set_id not in st.session_state.vectorstores:
            documents=[]
            for uploaded_file in uploaded_files:
                temppdf=f"./temp_{uploaded_file.name}.pdf"
                with open(temppdf,'wb') as file:
                    file.write(uploaded_file.getvalue()) 
                    

                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                documents.extend(docs)

            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=400)
            splits=text_splitter.split_documents(documents)
            if not splits:
                st.error("❌ No readable text found in the uploaded PDF(s).")
                st.stop()
            st.session_state.vectorstores[document_set_id] = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
        vectorstore = st.session_state.vectorstores[document_set_id]
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        contextualize_q_system_prompt=(
            "Given a chat history and latest user question "
            "which might reference content in the chat history,"
            "formulate a standalone question which can be understood "
            "without the chat history.Do not answer the question; "
            "just reformulate it if needed otherwise return it as is."
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ]
        )

        history_aware_retriever=(
            {
                "chat_history": itemgetter("chat_history"),
                "input":itemgetter("input")
            }
            |contextualize_q_prompt
            |llm
            |(lambda x:x.content)
            |retriever
        )

        #answer question prompt
        system_prompt=(
            "You are an assistant for question answering task."
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer say that you dont know."
            "Use five sentences maximum and keep context concise."
            "\n\n"
            "{context}"
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        question_answer_chain=(
            qa_prompt
            |llm
            |(lambda x:{"answer": x.content})
        )

        rag_chain=(
            {
                "context":history_aware_retriever,
                "chat_history": itemgetter("chat_history"),
                "input":itemgetter("input")
            }
            |question_answer_chain
        )

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session]=ChatMessageHistory()
            return st.session_state.store[session]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input=st.text_input("Your Question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            # st.write(st.session_state.store)
            st.markdown(f"Assistant: {response['answer']}")
            st.write("Chat History:",session_history.messages)
            

else:
    st.warning("Please Enter your Groq API Key")



 

