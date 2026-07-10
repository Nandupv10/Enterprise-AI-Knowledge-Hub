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


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# --------------------------------------------------
# User inputs
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    key="pdf_uploader"
)

question = st.text_input(
    "Ask a question about the PDF"
)


if uploaded_file is not None:

    # Add document name to document list
    if uploaded_file.name not in st.session_state.uploaded_documents:
        st.session_state.uploaded_documents.append(uploaded_file.name)

    st.success("PDF Uploaded Successfully!")
    st.write("File Name:", uploaded_file.name)


    # --------------------------------------------------
    # STEP 1: Read PDF
    # --------------------------------------------------

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


    # --------------------------------------------------
    # STEP 2: Split text into chunks
    # --------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)


    # --------------------------------------------------
    # STEP 3: Create embeddings
    # --------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # --------------------------------------------------
    # STEP 4: Add source metadata
    # --------------------------------------------------

    metadatas = [
        {
            "source": uploaded_file.name,
            "chunk_number": index + 1
        }
        for index in range(len(chunks))
    ]


    # --------------------------------------------------
    # STEP 5: Store chunks in ChromaDB
    # --------------------------------------------------

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    st.success("Chunks stored in ChromaDB!")


    # --------------------------------------------------
    # STEP 6: Connect to Llama through Groq API
    # --------------------------------------------------

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


    # --------------------------------------------------
    # STEP 7: Answer the user's question
    # --------------------------------------------------

    if question:

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

        # --------------------------------------------------
        # Send prompt to Llama through Groq
        # --------------------------------------------------

        try:

            response = llm.invoke(prompt)

            answer = response.content

        except Exception as e:

            st.error(
                "The AI service is currently unavailable. "
                "Please try again."
            )

            with st.expander("Error Details"):
                st.write(str(e))

            st.stop()


        # --------------------------------------------------
        # Store question and answer in conversation history
        # --------------------------------------------------

        current_entry = {
            "question": question,
            "answer": answer,
            "source": uploaded_file.name
        }

        # Avoid adding the same question repeatedly
        if (
            not st.session_state.chat_history
            or st.session_state.chat_history[-1]["question"] != question
        ):
            st.session_state.chat_history.append(current_entry)


        # --------------------------------------------------
        # Display final AI answer
        # --------------------------------------------------

        st.subheader("🤖 AI Answer")
        st.write(answer)


        # --------------------------------------------------
        # Display source citation
        # --------------------------------------------------

        st.subheader("📄 Source Citation")

        st.write(
            f"**Source Document:** {uploaded_file.name}"
        )

        with st.expander("View Retrieved Source Chunks"):

            for index, doc in enumerate(results, start=1):

                chunk_number = doc.metadata.get(
                    "chunk_number",
                    index
                )

                st.markdown(
                    f"**Retrieved Chunk {index} "
                    f"(Original Chunk #{chunk_number})**"
                )

                st.write(doc.page_content)

                if index < len(results):
                    st.divider()


# --------------------------------------------------
# Document management
# --------------------------------------------------

if st.session_state.uploaded_documents:

    st.divider()

    st.subheader("📁 Document Management")

    st.write("Uploaded Documents:")

    for document_name in st.session_state.uploaded_documents:

        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(f"📄 {document_name}")

        with col2:

            if st.button(
                "🗑️ Delete",
                key=f"delete_{document_name}"
            ):

                # Remove document from the document list
                st.session_state.uploaded_documents.remove(
                    document_name
                )

                # Remove chat history belonging to this document
                st.session_state.chat_history = [
                    chat
                    for chat in st.session_state.chat_history
                    if chat["source"] != document_name
                ]

                st.success(
                    f"{document_name} deleted successfully!"
                )

                st.rerun()


# --------------------------------------------------
# Display conversation history
# --------------------------------------------------

if st.session_state.chat_history:

    st.divider()

    st.subheader("💬 Conversation History")

    for index, chat in enumerate(
        st.session_state.chat_history,
        start=1
    ):

        with st.expander(
            f"Question {index}: {chat['question']}"
        ):

            st.markdown("**Question:**")
            st.write(chat["question"])

            st.markdown("**AI Answer:**")
            st.write(chat["answer"])

            st.markdown("**Source:**")
            st.write(chat["source"])