from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

FD_FOLDERS = "/mnt/linux_storage/projects/Historical_AI/rag/texts-l"
EMB_MODEL = "embeddinggemma:latest"

class HistoricalVector:
    def __init__(self):
        self.add_docs = not os.path.exists("./chroma_db")
        self.embedding_func = OllamaEmbeddings(model=EMB_MODEL)
        self.vector_stores = None
        if self.add_docs:
            self.vector_stores = self.create_vectors()
            self.add_docs = False
        else:
            self.vector_stores = Chroma(persist_directory="./chroma_db",
                                         embedding_function=self.embedding_func)
        self.retriever = self.vector_stores.as_retriever(search_kwargs={"k": 3})

    def create_vectors(self): 
        add_docs = not os.path.exists("./chroma_db")
        documents = []
        ids = []


        for i, file in enumerate(os.listdir(FD_FOLDERS)):
            with open(os.path.join(FD_FOLDERS, file), "r") as f:
                text = f.read()
                for j, line in enumerate(text.splitlines()):
                    if len(line) > 10:
                        print(f"Document {i}.{j}: {line[:15]}\n")
                        doc = Document(page_content=line, id=f"{i}.{j}")
                        documents.append(doc)
                        ids.append(doc.id)

        return Chroma.from_documents(documents=documents, 
                                    ids=ids,
                                    embedding=OllamaEmbeddings(model=EMB_MODEL), 
                                    persist_directory="./chroma_db")

    def search(self, query):
        return self.retriever.invoke(query)

if __name__ == "__main__":
    historical_vector = HistoricalVector()
    historical_context = historical_vector.search("A group conversation between a native Algerian, a Morrocan, and a french settler")
    print(historical_context)