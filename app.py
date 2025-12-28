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
    
    # session_id=st.text_input("Session ID",value="Default Session")

    # Initialize chat history store
    
    if 'vectorstores' not in st.session_state:
        st.session_state.vectorstores = {}
    
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = {}
    
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if "active_pdf_id" not in st.session_state:
        st.session_state.active_pdf_id = None
    
    if 'pdf_settings' not in st.session_state:
        st.session_state.pdf_settings = {}

    uploaded_files=st.file_uploader("Choose PDF file",type="pdf",accept_multiple_files=True)
    
    #process my uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_id = uploaded_file.name  # stable identity

            # Skip if already processed
            if pdf_id in st.session_state.vectorstores:
                continue

            # Save temp file
            temp_path = f"./temp_{pdf_id}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            num_pages=len(documents)

            if num_pages <= 20:
                chunk_size, chunk_overlap, k = 1000, 400, 3
            elif num_pages <= 50:
                chunk_size, chunk_overlap, k = 1500, 300, 5
            else:
                chunk_size, chunk_overlap, k = 2000, 200, 7

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            splits=text_splitter.split_documents(documents)
            if not splits:
                st.error("❌ No readable text found in the uploaded PDF(s).")
                st.stop()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=pdf_id
            )
            st.session_state.vectorstores[pdf_id] = vectorstore
            st.session_state.uploaded_pdfs[pdf_id] = pdf_id
            st.session_state.chat_histories[pdf_id] = ChatMessageHistory()
            
            st.session_state.pdf_settings[pdf_id] = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "k": k}

            if st.session_state.active_pdf_id is None:
                st.session_state.active_pdf_id = pdf_id

            st.success(f"Indexed: {pdf_id}")

    if st.session_state.uploaded_pdfs:
        st.subheader("Select Active PDF")

        selected_pdf = st.selectbox(
            "Active document",
            options=list(st.session_state.uploaded_pdfs.keys()),
            index=list(st.session_state.uploaded_pdfs.keys()).index(
                st.session_state.active_pdf_id
            )
        )

        st.session_state.active_pdf_id = selected_pdf
        st.info(f"Active PDF: {selected_pdf}")
        
        if st.button("🔄 Reset Chat for Active PDF"):
            st.session_state.chat_histories[selected_pdf] = ChatMessageHistory()
            st.success(f"Chat history cleared for {selected_pdf}")

        vectorstore = st.session_state.vectorstores[selected_pdf]
        k = st.session_state.pdf_settings[selected_pdf]["k"]
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})


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
        def get_session_history(_: str) -> BaseChatMessageHistory:
            return st.session_state.chat_histories[selected_pdf]

        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input=st.text_input("Your Question:")
        if user_input:
            
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":selected_pdf}
                }
            )
            
            st.markdown(f"Assistant: {response['answer']}")
            with st.expander("Chat History"):
                for msg in st.session_state.chat_histories[selected_pdf].messages:
                    st.write(msg)

else:
    st.warning("Please enter your Groq API key")



 

