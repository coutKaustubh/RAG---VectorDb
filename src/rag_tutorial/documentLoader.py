import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from dotenv import load_dotenv

load_dotenv()


def load_text_file():
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8"
    ) as temp_file:

        temp_file.write("""
This is a short story about the importance of data in AI.

In a small village, there lived a wise old woman who was known for her
ability to predict the future. People would come from far and wide to
seek her advice, and she would always provide them with the right answers.
""")

        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path, encoding="utf-8")
        documents = loader.load()

        print(f"\nLoaded {len(documents)} document(s)")

        for doc in documents:
            print("\nDocument content:")
            print(doc.page_content)

            print("\nMetadata:")
            print(doc.metadata)

    finally:
        os.remove(temp_file_path)


# if __name__ == "__main__":
#     load_text_file()



def load_pdf_file(pdf_path :str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"\nLoaded {len(documents)} document(s)")

    for doc in documents:
        print("\nDocument content:")
        print(doc.page_content)

        # print("\nMetadata:")
        # print(doc.metadata)

if __name__ == "__main__":
    load_pdf_file(r"E:\Projects\RAG- tutorial\src\documents\RESUME.pdf")
    