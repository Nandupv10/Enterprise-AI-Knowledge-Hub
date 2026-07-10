import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Enterprise AI Knowledge Hub",
    page_icon="📚"
)

st.title("📚 Enterprise AI Knowledge Hub")

# User inputs
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question about the PDF")


if uploaded_file is not None:

    st.success("PDF Uploaded Successfully!")
    st.write("File Name:", uploaded_file.name)

    # -------------------------
    # STEP 1: Read PDF
    # -------------------------

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Check whether text was successfully extracted
    if not text.strip():
        st.error("No readable text was found in the PDF.")
        st.stop()


    # -------------------------
    # STEP 2: Split into chunks
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)


    # -------------------------
    # STEP 3: Create embeddings
    # -------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # -------------------------
    # STEP 4: Store in ChromaDB
    # -------------------------

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    st.success("Chunks stored in ChromaDB!")


    # -------------------------
    # STEP 5: Connect to Llama
    # through Groq API
    # -------------------------

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


    # -------------------------
    # STEP 6: Answer question
    # -------------------------

    if question:

        try:

            # Find the top 3 most relevant chunks
            results = db.similarity_search(
                question,
                k=3
            )

            # Combine the retrieved chunks
            context = "\n\n".join(
                doc.page_content for doc in results
            )

            # Create the prompt for Llama
            prompt = f"""
You are an AI assistant for an Enterprise AI Knowledge Hub.

Answer the user's question using ONLY the information provided in the context below.

Do not use outside knowledge.

If the answer cannot be found in the context, reply exactly:

"I couldn't find that information in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

            # Send the prompt to Llama through Groq
            response = llm.invoke(prompt)

            # Display the final AI answer
            st.subheader("🤖 AI Answer")
            st.write(response.content)

        except Exception as e:

            st.error(
                "The AI service is currently unavailable. "
                "Please try again."
            )

            with st.expander("Error Details"):
                st.write(str(e))