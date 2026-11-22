
import os
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from data_chunker import (
    load_pdf,
    chunk_documents,
    get_doc_id
)

from Model_load import load_llm

from Vector_db import (
    store_chunks,
    search_chunks
)

from prompt import (
    SYSTEM_PROMPT,
    TOPIC_PROMPT,
    SUMMARY_PROMPT,
    CHAPTER_SUMMARY_PROMPT,
    QUIZ_PROMPT
)


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

PDF_FOLDER = Path(
    os.getenv("PDF_FOLDER", "uploaded_pdfs")
)

PDF_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "documind.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("DocuMind")


# ============================================================
# 3. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "indexed_docs": {},
    "topic_results": {},
    "summary_results": {},
    "quiz_results": {},
    "active_documents": [],
    "last_ingestion": None
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value.copy() if isinstance(
            value, (dict, list)
        ) else value


# ============================================================
# 4. CUSTOM CSS - CLASSY DOCUMENT LIBRARY THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F1E9;
}

[data-testid="stSidebar"] {
    background-color: #E9E1D4;
    border-right: 1px solid #D8CCBA;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1350px;
}

h1, h2, h3 {
    color: #37342F;
    font-family: Georgia, serif;
}

p, li, label {
    color: #655E54;
}

[data-testid="stMetric"] {
    background-color: #EAE2D6;
    border: 1px solid #DDD1C0;
    padding: 18px;
    border-radius: 14px;
}

.stButton button {
    background-color: #E7DDCE;
    color: #423A30;
    border: 1px solid #CDBDA7;
    border-radius: 10px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton button:hover {
    background-color: #D8C7B0;
    color: #302920;
    border-color: #B7A083;
}

.stTextInput input {
    border-radius: 10px;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
}

[data-testid="stTabs"] button {
    font-weight: 600;
}

hr {
    border-color: #DCD2C4;
}

.documind-subtitle {
    color: #8C7D69;
    font-size: 16px;
}

.documind-card {
    background-color: #EAE2D6;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #D8CCBA;
    min-height: 140px;
}

.documind-card h3 {
    font-size: 20px;
    margin-bottom: 8px;
}

.documind-card p {
    font-size: 14px;
}

.small-note {
    color: #8B806F;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 5. GENERAL HELPER FUNCTIONS
# ============================================================

def get_pdf_files():

    return sorted([
        file
        for file in PDF_FOLDER.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".pdf"
    ])


def get_selected_paths(names):

    return [
        PDF_FOLDER / name
        for name in names
        if (PDF_FOLDER / name).is_file()
    ]


def get_selected_doc_ids(paths):

    return [
        get_doc_id(str(path))
        for path in paths
    ]


def save_uploaded_pdfs(uploaded_files):

    saved = 0
    unchanged = 0

    for uploaded_file in uploaded_files:

        filename = Path(uploaded_file.name).name

        if not filename.lower().endswith(".pdf"):
            continue

        destination = PDF_FOLDER / filename

        new_content = uploaded_file.getvalue()

        if destination.exists():

            old_content = destination.read_bytes()

            if old_content == new_content:
                unchanged += 1
                continue

        destination.write_bytes(new_content)

        saved += 1

        logger.info(
            "PDF saved successfully: %s",
            filename
        )

    return saved, unchanged


def extract_document_text(paths):

    extracted = []

    for path in paths:

        documents = load_pdf(str(path))

        for document in documents:

            page = document.metadata.get("page", "?")

            content = document.page_content.strip()

            if content:

                extracted.append(
                    f"\n[Document: {path.name} | Page: {page}]\n"
                    f"{content}"
                )

    return "\n\n".join(extracted)


def split_text(text, max_chars=6000):

    text = text.strip()

    return [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
        if text[i:i + max_chars].strip()
    ]


def invoke_llm(prompt):

    llm = load_llm()

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        return "\n".join(
            str(part.get("text", part))
            if isinstance(part, dict)
            else str(part)
            for part in content
        )

    return str(content)


# ============================================================
# 6. PDF INGESTION PIPELINE
# ============================================================

def index_documents(paths):

    successful = []
    failed = []
    total_chunks = 0

    if not paths:

        return successful, failed, total_chunks

    progress = st.progress(0)

    status = st.empty()

    for i, path in enumerate(paths):

        try:

            status.info(
                f"📖 Reading {path.name}..."
            )

            # Step 1: Extract PDF pages
            documents = load_pdf(str(path))

            if not documents:
                raise ValueError(
                    "No pages could be extracted."
                )

            # Step 2: Check readable text
            readable_text = [
                doc for doc in documents
                if doc.page_content.strip()
            ]

            if not readable_text:

                raise ValueError(
                    "No readable text found. "
                    "This may be a scanned PDF requiring OCR."
                )

            # Step 3: Split into chunks
            chunks = chunk_documents(documents)

            if not chunks:

                raise ValueError(
                    "Document chunking returned no content."
                )

            # Step 4: Generate document identity
            doc_id = get_doc_id(str(path))

            # Step 5: Store vectors in Pinecone
            stored = store_chunks(chunks)

            if stored <= 0:

                raise ValueError(
                    "No chunks were stored in Pinecone."
                )

            # Step 6: Track successfully indexed document
            successful.append({
                "name": path.name,
                "doc_id": doc_id,
                "chunks": stored,
                "pages": len(documents)
            })

            total_chunks += stored

            logger.info(
                "Indexed %s successfully. Chunks: %s",
                path.name,
                stored
            )

        except Exception as error:

            logger.exception(
                "Failed to process PDF: %s",
                path.name
            )

            failed.append({
                "name": path.name,
                "error": str(error)
            })

        progress.progress(
            (i + 1) / len(paths)
        )

    status.empty()
    progress.empty()

    return successful, failed, total_chunks


# ============================================================
# 7. TOPIC EXTRACTION
# ============================================================



def generate_topics(text):

    # Keep each LLM request small
    sections = split_text(
        text,
        max_chars=1200
    )

    results = []

    progress = st.progress(0)
    status = st.empty()

    for i, section in enumerate(sections):

        status.info(
            f"Extracting topics: {i + 1}/{len(sections)}"
        )

        prompt = f"""
        Extract the document topics from this text.

        Return only:
        - Main headings
        - Subtopics
        - Important concepts

        Maximum 120 words.

        Do not summarize.
        Do not repeat topics.
        Do not add outside knowledge.
        Use simple Markdown bullets.

        TEXT:
        {section}
        """

        llm = load_llm()

        # Apply a small output limit to this call
        response = llm.invoke(prompt)

        results.append(response.content)

        progress.progress(
            (i + 1) / len(sections)
        )

    progress.empty()
    status.empty()

    return "\n\n".join(results)



# ============================================================
# 8. DOCUMENT SUMMARIZATION
# ============================================================

def generate_summary(text, quick=False):

    sections = split_text(
        text,
        max_chars=6000
    )

    section_summaries = []

    progress = st.progress(0)

    for i, section in enumerate(sections):

        prompt = CHAPTER_SUMMARY_PROMPT.format(
            text=section
        )

        result = invoke_llm(prompt)

        section_summaries.append(result)

        progress.progress(
            (i + 1) / len(sections)
        )

    progress.empty()

    combined = "\n\n".join(
        section_summaries
    )

    if quick:

        final_prompt = f"""
        Create a concise executive summary of the
        following document.

        Include only:
        - Main idea
        - Most important topics
        - Key findings
        - Important takeaways

        Keep it easy to understand.

        CONTENT:

        {combined}
        """

    else:

        final_prompt = SUMMARY_PROMPT.format(
            text=combined
        )

    return invoke_llm(
        final_prompt
    )


# ============================================================
# 9. RETRIEVAL AUGMENTED GENERATION
# ============================================================

def retrieve_context(question, selected_paths, top_k=6):

    if not selected_paths:

        raise ValueError(
            "Please select at least one PDF."
        )

    doc_ids = get_selected_doc_ids(
        selected_paths
    )

    # Only search selected documents
    retrieved = search_chunks(
        question=question,
        top_k=top_k,
        doc_ids=doc_ids
    )

    # Remove empty results
    retrieved = [
        item for item in retrieved
        if item.get("text", "").strip()
    ]

    return retrieved


def build_context(retrieved):

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f"""
            DOCUMENT: {item.get('file_name', 'Unknown')}

            PAGE: {item.get('page', 'Unknown')}

            CONTENT:
            {item.get('text', '')}
            """
        )

    return "\n\n".join(
        context_parts
    )


def format_sources(retrieved):

    sources = {}

    for item in retrieved:

        key = (
            item.get("file_name"),
            item.get("page")
        )

        sources[key] = True

    output = []

    for filename, page in sources:

        output.append(
            f"- **{filename}** — Page {page}"
        )

    return "\n".join(output)


# ============================================================
# 10. HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.markdown(
        """
        <h1 style="font-size:45px;margin-bottom:0;">
            DocuMind.
        </h1>
        <p class="documind-subtitle">
            Your personal AI-powered knowledge space.
        </p>
        """,
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        """
        <div style="text-align:right;font-size:38px;">
            📚 ✨
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 11. SIDEBAR - DOCUMENT MANAGEMENT
# ============================================================

with st.sidebar:

    st.markdown("## 📚 Your Library")

    st.caption(
        "Bring all your documents into one place."
    )

    # ---------------- UPLOAD ----------------

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF documents."
    )

    if st.button(
        "＋ Add PDFs to Library",
        use_container_width=True
    ):

        if uploaded_files:

            try:

                saved, unchanged = save_uploaded_pdfs(
                    uploaded_files
                )

                if saved:

                    st.success(
                        f"{saved} PDF(s) saved successfully."
                    )

                if unchanged:

                    st.info(
                        f"{unchanged} unchanged PDF(s) skipped."
                    )

                st.rerun()

            except Exception as error:

                logger.exception(
                    "Upload failed"
                )

                st.error(
                    f"Upload failed: {error}"
                )

        else:

            st.warning(
                "Please select PDFs first."
            )

    st.divider()

    # ---------------- FOLDER SCAN ----------------

    st.markdown("### 📂 Folder Import")

    st.caption(
        "Scan your uploaded_pdfs folder and "
        "automatically process its documents."
    )

    if st.button(
        "🔄 Scan & Process PDFs",
        use_container_width=True
    ):

        paths = get_pdf_files()

        if not paths:

            st.warning(
                "No PDF files found. Add PDFs to your library first."
            )

        else:

            with st.spinner(
                "Reading PDFs and creating your knowledge base..."
            ):

                successful, failed, total = index_documents(
                    paths
                )

                for item in successful:

                    st.session_state.indexed_docs[
                        item["doc_id"]
                    ] = item

                st.session_state.last_ingestion = (
                    len(successful)
                )

            if successful:

                st.success(
                    f"{len(successful)} PDF(s) processed "
                    f"and {total} chunks stored!"
                )

            if failed:

                for item in failed:

                    st.error(
                        f"{item['name']}: {item['error']}"
                    )

            if successful:

                st.rerun()

    st.divider()

    # ---------------- DOCUMENT LIST ----------------

    st.markdown("### 📖 Available Documents")

    pdf_paths = get_pdf_files()

    pdf_names = [
        path.name for path in pdf_paths
    ]

    if pdf_names:

        st.caption(
            f"{len(pdf_names)} documents in your library"
        )

        selected_names = st.multiselect(
            "Select documents",
            options=pdf_names,
            default=pdf_names,
            key="document_selector"
        )

    else:

        st.info(
            "Your library is empty. Upload your first PDF."
        )

        selected_names = []

    st.divider()

    # ---------------- INDEXING STATUS ----------------

    st.markdown("### ⚙️ Knowledge Base")

    selected_paths = get_selected_paths(
        selected_names
    )

    selected_ids = set(
        get_selected_doc_ids(selected_paths)
    )

    indexed_ids = set(
        st.session_state.indexed_docs.keys()
    )

    indexed_selected = (
        selected_ids.intersection(indexed_ids)
    )

    if selected_ids and selected_ids.issubset(indexed_ids):

        st.success(
            f"{len(indexed_selected)} selected document(s) indexed."
        )

    elif selected_ids:

        st.warning(
            "Some selected documents need indexing."
        )

    else:

        st.info(
            "Select documents to begin."
        )

    # ---------------- PROCESS SELECTED ----------------

    if st.button(
        "🚀 Process Selected PDFs",
        use_container_width=True
    ):

        if not selected_paths:

            st.warning(
                "Select at least one PDF."
            )

        else:

            with st.spinner(
                "Processing selected documents..."
            ):

                successful, failed, total = index_documents(
                    selected_paths
                )

                for item in successful:

                    st.session_state.indexed_docs[
                        item["doc_id"]
                    ] = item

            if successful:

                st.success(
                    f"{len(successful)} PDF(s) indexed. "
                    f"{total} chunks stored."
                )

            for item in failed:

                st.error(
                    f"{item['name']}: {item['error']}"
                )

            if successful:

                st.rerun()

    st.divider()

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 12. MAIN DASHBOARD METRICS
# ============================================================

pdf_paths = get_pdf_files()

pdf_names = [
    path.name for path in pdf_paths
]

indexed_ids = set(
    st.session_state.indexed_docs.keys()
)

all_current_ids = {
    get_doc_id(str(path))
    for path in pdf_paths
}

indexed_current = indexed_ids.intersection(
    all_current_ids
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "📚 PDF Documents",
        len(pdf_names)
    )

with col2:

    st.metric(
        "⚡ Indexed Documents",
        len(indexed_current)
    )

with col3:

    st.metric(
        "💬 Chat Messages",
        len(st.session_state.messages)
    )


# ============================================================
# 13. APPLICATION NAVIGATION
# ============================================================

overview, topics, summary, chat, study = st.tabs([
    "🏠 Overview",
    "📑 Explore Topics",
    "✨ Summarize",
    "💬 Chat with PDFs",
    "🧠 Study & Quiz"
])


# ============================================================
# 14. OVERVIEW
# ============================================================

with overview:

    st.markdown(
        "## Welcome to your knowledge space"
    )

    st.markdown(
        "Transform lengthy documents into organized, "
        "searchable and understandable knowledge."
    )

    st.image(
        "https://images.unsplash.com/photo-1507842217343-583bb7270b66"
        "?auto=format&fit=crop&w=1600&q=85",
        use_container_width=True
    )

    st.markdown("### What would you like to do?")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="documind-card">
            <h3>📑 Explore Topics</h3>
            <p>
            Automatically discover the main headings,
            subheadings and concepts in your documents.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="documind-card">
            <h3>💬 Chat with PDFs</h3>
            <p>
            Ask natural-language questions and receive
            answers supported by document context.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="documind-card">
            <h3>✨ Smart Summaries</h3>
            <p>
            Understand long documents with structured,
            concise and detailed summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="documind-card">
            <h3>🧠 Study & Quiz</h3>
            <p>
            Create practice questions and revise
            important concepts from your PDFs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Your Document Collection")

    if pdf_names:

        for path in pdf_paths:

            doc_id = get_doc_id(str(path))

            status = (
                "✅ Indexed"
                if doc_id in indexed_ids
                else "⏳ Not indexed"
            )

            st.markdown(
                f"📄 **{path.name}** — {status}"
            )

    else:

        st.info(
            "Your document collection is empty."
        )


# ============================================================
# 15. EXPLORE TOPICS
# ============================================================

with topics:

    st.markdown(
        "## 📑 Discover What's Inside Your PDFs"
    )

    st.caption(
        "Let AI analyze the structure and identify important topics."
    )

    if not selected_paths:

        st.warning(
            "Please upload and select a PDF first."
        )

    else:

        topic_path = st.selectbox(
            "Choose a document",
            selected_paths,
            format_func=lambda x: x.name,
            key="topic_selector"
        )

        topic_id = get_doc_id(
            str(topic_path)
        )

        if st.button(
            "🔍 Discover Topics",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing your document..."
                ):

                    text = extract_document_text([
                        topic_path
                    ])

                    if not text.strip():

                        raise ValueError(
                            "No readable text found in this PDF."
                        )

                    result = generate_topics(text)

                    st.session_state.topic_results[
                        topic_id
                    ] = result

            except Exception as error:

                logger.exception(
                    "Topic extraction failed"
                )

                st.error(
                    f"Topic extraction failed: {error}"
                )

        if topic_id in st.session_state.topic_results:

            st.divider()

            st.markdown("### Document Structure")

            st.markdown(
                st.session_state.topic_results[topic_id]
            )

            st.download_button(
                "⬇️ Download Topic Outline",
                data=st.session_state.topic_results[
                    topic_id
                ],
                file_name=f"{topic_path.stem}_topics.md",
                mime="text/markdown"
            )


# ============================================================
# 16. SUMMARIZATION
# ============================================================

with summary:

    st.markdown(
        "## ✨ Smart Document Summarizer"
    )

    st.caption(
        "Convert lengthy PDFs into useful knowledge."
    )

    if not selected_paths:

        st.warning(
            "Select a PDF to generate a summary."
        )

    else:

        summary_path = st.selectbox(
            "Select a document",
            selected_paths,
            format_func=lambda x: x.name,
            key="summary_selector"
        )

        summary_id = get_doc_id(
            str(summary_path)
        )

        summary_type = st.radio(
            "Choose summary style",
            [
                "Detailed Summary",
                "Quick Summary"
            ],
            horizontal=True
        )

        if st.button(
            "✨ Generate Summary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Reading and summarizing your PDF..."
                ):

                    text = extract_document_text([
                        summary_path
                    ])

                    if not text.strip():

                        raise ValueError(
                            "No readable text found."
                        )

                    result = generate_summary(
                        text,
                        quick=(
                            summary_type == "Quick Summary"
                        )
                    )

                    cache_key = (
                        summary_id,
                        summary_type
                    )

                    st.session_state.summary_results[
                        cache_key
                    ] = result

            except Exception as error:

                logger.exception(
                    "Summarization failed"
                )

                st.error(
                    f"Summary generation failed: {error}"
                )

        cache_key = (
            summary_id,
            summary_type
        )

        if cache_key in st.session_state.summary_results:

            st.divider()

            st.markdown(
                "### Your Document Summary"
            )

            result = st.session_state.summary_results[
                cache_key
            ]

            st.markdown(result)

            st.download_button(
                "⬇️ Download Summary",
                data=result,
                file_name=f"{summary_path.stem}_summary.md",
                mime="text/markdown"
            )


# ============================================================
# 17. CHAT WITH PDFS
# ============================================================

with chat:

    st.markdown(
        "## 💬 Chat with Your Documents"
    )

    st.caption(
        "Ask questions and get answers grounded in your PDFs."
    )

    st.info(
        "Select the documents you want to search in the sidebar. "
        "Those documents must be indexed first."
    )

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            answer = ""

            try:

                if not selected_paths:

                    raise ValueError(
                        "Select at least one PDF in the sidebar."
                    )

                selected_ids = set(
                    get_selected_doc_ids(selected_paths)
                )

                not_indexed = (
                    selected_ids
                    - set(st.session_state.indexed_docs.keys())
                )

                if not_indexed:

                    raise ValueError(
                        "Some selected documents are not indexed. "
                        "Click Process Selected PDFs first."
                    )

                with st.spinner(
                    "🔎 Searching your documents..."
                ):

                    # Step 1: Retrieve relevant chunks
                    retrieved = retrieve_context(
                        question,
                        selected_paths,
                        top_k=6
                    )

                    if not retrieved:

                        answer = (
                            "I couldn't find relevant information "
                            "in your selected PDFs. Try rephrasing "
                            "your question or selecting another document."
                        )

                    else:

                        # Step 2: Build context
                        context = build_context(
                            retrieved
                        )

                        # Step 3: Build RAG prompt
                        prompt = SYSTEM_PROMPT.format(
                            context=context,
                            question=question
                        )

                        # Step 4: Generate answer
                        answer = invoke_llm(
                            prompt
                        )

                        # Step 5: Attach sources
                        sources = format_sources(
                            retrieved
                        )

                        answer += (
                            "\n\n---\n"
                            "### 📚 Sources\n"
                            + sources
                        )

                    st.markdown(answer)

            except Exception as error:

                logger.exception(
                    "Chat operation failed"
                )

                answer = (
                    f"Unable to answer your question: {error}"
                )

                st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# ============================================================
# 18. STUDY AND QUIZ
# ============================================================

with study:

    st.markdown(
        "## 🧠 Your Personal Study Assistant"
    )

    st.caption(
        "Create revision material directly from your documents."
    )

    if not selected_paths:

        st.warning(
            "Select at least one PDF."
        )

    else:

        quiz_count = st.slider(
            "Number of questions",
            min_value=3,
            max_value=15,
            value=5
        )

        if st.button(
            "🎯 Generate Practice Quiz",
            use_container_width=True
        ):

            try:

                selected_ids = set(
                    get_selected_doc_ids(selected_paths)
                )

                if not selected_ids.issubset(
                    st.session_state.indexed_docs.keys()
                ):

                    raise ValueError(
                        "Please index all selected PDFs first."
                    )

                with st.spinner(
                    "Creating your practice questions..."
                ):

                    retrieved = search_chunks(
                        question=(
                            "Important topics, concepts, "
                            "definitions and key facts"
                        ),
                        top_k=15,
                        doc_ids=list(selected_ids)
                    )

                    context = build_context(
                        retrieved
                    )

                    if not context.strip():

                        raise ValueError(
                            "No document content was retrieved."
                        )

                    prompt = QUIZ_PROMPT.format(
                        number=quiz_count,
                        context=context
                    )

                    result = invoke_llm(
                        prompt
                    )

                    st.session_state.quiz_results[
                        tuple(sorted(selected_ids))
                    ] = result

            except Exception as error:

                logger.exception(
                    "Quiz generation failed"
                )

                st.error(
                    f"Quiz generation failed: {error}"
                )

        quiz_key = tuple(
            sorted(
                get_selected_doc_ids(selected_paths)
            )
        )

        if quiz_key in st.session_state.quiz_results:

            st.divider()

            st.markdown(
                "### 📝 Your Practice Questions"
            )

            st.markdown(
                st.session_state.quiz_results[quiz_key]
            )

            st.download_button(
                "⬇️ Download Quiz",
                data=st.session_state.quiz_results[
                    quiz_key
                ],
                file_name="documind_practice_quiz.md",
                mime="text/markdown"
            )


# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#958773;">
        <p>📚 DocuMind AI</p>
        <p class="small-note">
            Turning documents into knowledge.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
