# %%
from haystack.utils import ComponentDevice, Device
import pandas as pd
import pickle




# %%
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

# 

# %%
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
import pickle



# %%
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever

# retriever = InMemoryEmbeddingRetriever(document_store=document_store)
# retriever = InMemoryBM25Retriever(document_store=document_store)

class QA_system():

    def reader_retriever(self):
    # Load the pickled model
        with open('document_store_res.pkl', 'rb') as f:
            document_store_new = pickle.load(f)

        retriever_model_path = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        retriever = InMemoryEmbeddingRetriever(document_store=document_store_new)

        return retriever

    def createDocumentEmbeddings(self, numDocuments):
        # read abstracts from pmc files for document store embeddings   
        df = pd.read_csv('pmcFiles_en.csv', usecols=['pmcid', 'abstract'], nrows=numDocuments)
        for i in range(df.shape[0]):
            print(df.loc[i][1])

        documents = [Document(id=df.loc[i][0], content=str(df.loc[i][1])) for i in range(df.shape[0])]
        model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

        self.document_store = InMemoryDocumentStore()

        indexing_pipeline = Pipeline()

        indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model, device=self.device), name="embedder")
        indexing_pipeline.add_component(instance=DocumentWriter(document_store=self.document_store), name="writer")
        indexing_pipeline.connect("embedder.documents", "writer.documents")

        indexing_pipeline.run({"documents": documents}, debug=True)

        # for saving a new document store full of article embeddings
        with open("document_store_res.pkl", "wb") as f:
            pickle.dump(self.document_store, f)

    def initialize(self):
        self.device = ComponentDevice.from_single(Device.gpu(id=0))
        self.retriever = self.reader_retriever()

        self.reader = ExtractiveReader(device=self.device, no_answer=False)
        self.reader.warm_up()

        self.extractive_qa_pipeline = Pipeline()

        self.model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

        self.extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=self.model, device=self.device), name="embedder")
        self.extractive_qa_pipeline.add_component(instance=self.retriever, name="retriever")
        self.extractive_qa_pipeline.add_component(instance=self.reader, name="reader")

        self.extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
        

    def query(self, query):
        results = self.extractive_qa_pipeline.run(
            data={"embedder": {"text": query}, "retriever": {"top_k": 5}, "reader": {"query": query, "top_k": 3}}
        )
        return results