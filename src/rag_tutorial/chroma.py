
import chromadb

client = chromadb.Client()

collection = client.create_collection(name="my_documents")

collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Python is a popular programming language used for AI and machine learning.",
        "ChromaDB is a vector database used to store and retrieve document embeddings.",
        "Django is a Python web framework used to build backend applications."
    ]
)

query = "Which database stores vector embeddings?"

results = collection.query(
    query_texts=[query],
    n_results=2,
    include=["documents", "distances"]
)

print("Query:", query)
print("\nRetrieved documents:")

for i, document in enumerate(results["documents"][0]):
    print(f"\n{i + 1}. {document}")
    print("ID:", results["ids"][0][i])
    print("Distance:", results["distances"][0][i])
