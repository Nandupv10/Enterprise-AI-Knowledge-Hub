import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Enterprise AI Knowledge Hub")

st.title("📚 Enterprise AI Knowledge Hub")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question about the PDF")

if uploaded_file is not None:

    st.success("PDF Uploaded Successfully!")
    st.write("File Name:", uploaded_file.name)

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0
    )

    # Store chunks in ChromaDB
    db = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    st.success("Chunks stored in ChromaDB!")

    if question:

        # Retrieve top 3 relevant chunks
        results = db.similarity_search(question, k=3)

        # Combine retrieved chunks
        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        # Prompt
        prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context, reply exactly:

"I couldn't find that information in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

        try:

            response = llm.invoke(prompt)

            st.subheader("🤖 AI Answer")
            st.write(response.content)

        except Exception as e:

            st.error("Gemini API is currently unavailable. Please try again in a few minutes.")

            st.expander("Error Details").write(str(e))