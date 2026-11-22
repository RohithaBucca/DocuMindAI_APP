
import os
import uuid
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from Model_load import load_embedding_model


load_dotenv()

INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "documind-knowledge"
)

NAMESPACE = "pdf-library"

EMBEDDING_DIMENSION = 384


def get_pinecone_index():

    api_key = os.getenv("PINECONE_API")

    if not api_key:
        raise ValueError("PINECONE_API is missing")

    pc = Pinecone(api_key=api_key)

    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv(
                    "PINECONE_REGION",
                    "us-east-1"
                )
            )
        )

        # Wait until the index is ready
        for _ in range(60):

            description = pc.describe_index(INDEX_NAME)

            if description.status["ready"]:
                break

            time.sleep(2)

        else:
            raise TimeoutError(
                "Pinecone index did not become ready."
            )

    return pc.Index(INDEX_NAME)


def store_chunks(chunks):

    model = load_embedding_model()

    index = get_pinecone_index()

    batch_size = 100
    stored_count = 0

    records = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        if not text:
            continue

        embedding = model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

        doc_id = chunk.metadata["doc_id"]
        file_name = chunk.metadata["file_name"]
        page = int(chunk.metadata["page"])

        chunk_number = int(
            chunk.metadata.get("chunk", 0)
        )

        vector_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{doc_id}:{page}:{chunk_number}:{text}"
        ))

        metadata = {
            "text": text,
            "doc_id": doc_id,
            "file_name": file_name,
            "page": page
        }

        records.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })

    for i in range(0, len(records), batch_size):

        batch = records[i:i + batch_size]

        index.upsert(
            vectors=batch,
            namespace=NAMESPACE
        )

        stored_count += len(batch)

    return stored_count


def search_chunks(question, top_k=5, doc_ids=None):

    model = load_embedding_model()

    index = get_pinecone_index()

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    query_arguments = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": NAMESPACE
    }

    # Restrict search to selected PDFs if requested
    if doc_ids:

        query_arguments["filter"] = {
            "doc_id": {"$in": list(doc_ids)}
        }

    results = index.query(**query_arguments)

    retrieved_chunks = []

    for match in results.matches:

        metadata = match.metadata or {}

        retrieved_chunks.append({
            "text": metadata.get("text", ""),
            "file_name": metadata.get(
                "file_name", "Unknown document"
            ),
            "page": metadata.get("page", "?"),
            "score": match.score
        })

    return retrieved_chunks
