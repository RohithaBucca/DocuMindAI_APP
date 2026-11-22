
import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_doc_id(pdf_path):
    """Generate a stable ID based on PDF contents."""

    hasher = hashlib.sha256()

    with open(pdf_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def load_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    doc_id = get_doc_id(pdf_path)
    file_name = os.path.basename(pdf_path)

    for document in documents:

        # Keep the page number as a 1-based page number
        document.metadata["page"] = (
            int(document.metadata.get("page", 0)) + 1
        )

        document.metadata["doc_id"] = doc_id
        document.metadata["file_name"] = file_name

        # Avoid exposing local machine paths
        document.metadata.pop("source", None)

    return documents


def load_pdfs_from_folder(folder_path):

    os.makedirs(folder_path, exist_ok=True)

    all_documents = []

    pdf_files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(".pdf")
    ]

    for pdf_path in pdf_files:

        try:
            documents = load_pdf(pdf_path)
            all_documents.extend(documents)

        except Exception as error:
            print(
                f"Could not process "
                f"{os.path.basename(pdf_path)}: {error}"
            )

    return all_documents


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    # Remove empty chunks
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    return chunks


def process_pdf(pdf_path):

    documents = load_pdf(pdf_path)

    chunks = chunk_documents(documents)

    return chunks


def process_folder(folder_path):

    documents = load_pdfs_from_folder(folder_path)

    chunks = chunk_documents(documents)

    return documents, chunks
