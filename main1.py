import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()



# Set the Google API key
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY"))

# Function for extracting the text from PDFs in a folder or a single PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Embed text chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Check for embedding models and select one.
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Use FAISS as the vector store
    vector_store.save_local("faiss_index")

# Create a conversational chain for the PDF
def get_conv_chain():
    prompt_template = """
    You are an AI assistant trained to analyze and provide insights from a specific financial annual report provided by a bank. 
    Your task is to answer user queries strictly based on the information present in the provided PDF document. Do not use external knowledge, assumptions, or data not included in the report.
    If the user's query cannot be answered from the PDF, politely inform them that the requested information is not available in the document. 
    Ensure your responses are concise, accurate, and directly reference relevant sections or data from the PDF whenever applicable.\n\n
    Context: \n {context}?\n
    Question: \n{question}\n
    Answer : 
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, memory=memory)  # StuffDocumentsChain combines the context to a single window
    return chain

# Process user input and provide an answer
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    # Load vectors locally for similarity search
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform a similarity search to get the most relevant chunks for the user's question
    docs = new_db.similarity_search(user_question, k=10)

    chain = get_conv_chain()

    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Main Streamlit App
def main():
    st.title("Chat with PDF using Gemini 💁")

    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False

    # File uploader for PDF
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        try:
            # Process PDF files
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state.vector_store_ready = True
            st.success("PDF processing completed successfully.")
        except Exception as e:
            st.error(f"Error processing PDFs: {e}")
    
    # Display chat history dynamically
    for entry in st.session_state.chat_history:
        st.markdown(f"**User:** {entry['User']}")
        st.markdown(f"**AI:** {entry['AI']}")

    # Ask a question if vector store is ready
    if st.session_state.vector_store_ready:
        user_question = st.chat_input("Ask a question:")

        if user_question:
            try:
                response = user_input(user_question)
                
                # Append to chat history
                st.session_state.chat_history.append({"User": user_question, "AI": response})

                # Display the latest interaction
                st.markdown(f"**User:** {user_question}")
                st.markdown(f"**AI:** {response}")

            except Exception as e:
                st.error(f"Error in answering the question: {e}")
    else:
        st.warning("Please upload PDF files and process them first.")

if __name__ == "__main__":
    main()
