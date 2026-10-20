
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tempfile


# Local embedding model: no OpenAI embedding API required
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def create_vectorstore(documents, persist_directory=None):
    return Chroma.from_documents( # chroma.from_documents means it will create a vector store from the documents provided
        documents=documents,
        embedding=embeddings_model,
        persist_directory=persist_directory, # used to store the vector store in disk, if not provided it will be stored in memory
    )


def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = create_vectorstore(SAMPLE_DOCS, tmpdir)
        try:
            print("Documents:", vectorstore._collection.count()) # used to count the number of documents in the vector store

            query = "What is LangChain?" # query to search for
            results = vectorstore.similarity_search(query, k=2) # perform similarity search for the query

            for i, doc in enumerate(results, start=1):#enumerate means to get the index and the value
                print(f"\nResult {i}: {doc.page_content}")
                print("Source:", doc.metadata["source"])
        finally:
            vectorstore._client.close()
            del vectorstore


def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = create_vectorstore(SAMPLE_DOCS, tmpdir)

        query = "Explain vector stores."
        results = vectorstore.similarity_search_with_score(query, k=3)

        for i, (doc, distance) in enumerate(results, start=1):
            print(f"\nResult {i}: {doc.page_content}")
            print("Distance:", distance)
            print("Source:", doc.metadata["source"])


#this filtering helps to narrow down results on the basis of metadata for eg : if we only want results 
# related to  "topic" metadata of type : 'database' so, it will only return the documents that have the metadata 
# "topic" of type "database"

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = create_vectorstore(SAMPLE_DOCS, tmpdir)

        query = "What databases are available?"

        print("\nWithout filtering:")
        for doc in vectorstore.similarity_search(query, k=5):
            print(doc.page_content)

        print("\nOnly database-topic documents:")
        for doc in vectorstore.similarity_search(
            query,
            k=5,
            filter={"topic": "database"},
        ):
            print(doc.page_content)


def as_retriever():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = create_vectorstore(SAMPLE_DOCS, tmpdir)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        print("\nSimilarity retriever:")
        for doc in retriever.invoke("How do I build AI applications?"):
            print(doc.page_content)

        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 5},
        )

        print("\nMMR retriever:")
        for doc in mmr_retriever.invoke("vector databases and embeddings"):
            print(doc.page_content)


def persist_chroma():
    persist_dir = "./chroma_db"

    vectorstore = create_vectorstore(
        SAMPLE_DOCS,
        persist_dir,
    )

    print("Stored documents:", vectorstore._collection.count())
    del vectorstore

    reloaded = Chroma(
        collection_name="langchain",
        embedding_function=embeddings_model,
        persist_directory=persist_dir,
    )

    print("Reloaded documents:", reloaded._collection.count())

    results = reloaded.similarity_search("LangChain", k=2)
    for doc in results:
        print(doc.page_content)


def exercise_vector_store_setup():
    def create_retriever(
        texts: list[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k: int = 3,
    ):
        docs = [Document(page_content=text) for text in texts]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_docs = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
        )

        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    sample_texts = [
        "Python is used for web development, data science, machine learning, and automation.",
        "JavaScript powers web applications. React and Vue help developers build user interfaces.",
        "Rust is a systems programming language focused on safety and performance.",
    ]

    retriever = create_retriever(
        sample_texts,
        chunk_size=200,
        chunk_overlap=20,
        k=2,
    )

    queries = [
        "What's good for web development?",
        "Which language focuses on safety?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        for doc in retriever.invoke(query):
            print("-", doc.page_content)


if __name__ == "__main__":
    # Uncomment one function at a time.
    # chroma_basics()
    # similarity_search_with_scores()
    # metadata_filtering()
    # as_retriever()
    # persist_chroma()
    exercise_vector_store_setup()
