import os
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes.other import Shaper
from haystack.nodes import PromptNode, PromptTemplate
from haystack.pipelines import Pipeline

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

from haystack.utils import convert_files_to_docs


# Set logging level to INFO
import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

#faiss_index_path = "faiss_document_store.db"


# Shaper helps expand the `query` variable into a list of identical queries (length of documents)
# and store the list of queries in the `questions` variable
# (the variable used in the question answering template)
def get_pipeline(doc_dir):

    # Registering a new prompt template called "concept-exemplar"
    prompt_template = \
    """
    Identify a non-overlapping spans of text in the given
    context that resonates with the given concept. By resonate we mean that the
    meaning of the concept is captured in the span. The span exemplifies what
    the concept means. The identified span MUST be present verbatim in the context. \n\nConcept: 'family support'\nContext: Abubakar and his
    wife are expecting his mother to help his wife with the new born. It is
    evening and she has not arrived yet. Then he decided to call the neighbor's
    wife to Come and help the baby and the new born woman to boil hot water and
    baths the baby and made the baby to sleep before the mother come
    back.\nSpan: are expecting his mother to help\n\n\nConcept: $concepts
    \nContext: $documents\nSpan:
    """
    template = PromptTemplate(name="concept-exemplar",prompt_text=prompt_template)
    prompt_node = PromptNode("text-davinci-003", api_key=api_key)
    prompt_node.add_prompt_template(template)

    # Set concept-exemplar as my default
    exemplifier = prompt_node.set_default_prompt_template("concept-exemplar")


    shaper = Shaper(func="value_to_list", inputs={"value": "query", "target_list":"documents"}, outputs=["concepts"])


    if os.path.exists("faiss_document_store.db"):
        print("FAISS document store already exists")
        document_store = FAISSDocumentStore(
            faiss_index_path="faiss_document_store.faiss",
            faiss_config_path="faiss_document_store.json")

        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    else:
        print("New Document Store created")
        document_store = FAISSDocumentStore(faiss_index_factory_str='Flat')

        docs = convert_files_to_docs(dir_path=doc_dir)

        document_store.write_documents(docs)

        # 4. Set up retriever
        # bm25_retriever = BM25Retriever(document_store=document_store)
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
        )
        # Important:
        # Now that we initialized the Retriever, we need to call update_embeddings() to iterate over all
        # previously indexed documents and update their embedding representation.
        # While this can be a time consuming operation (depending on the corpus size), it only needs to be done once.
        # At query time, we only need to embed the query and compare it to the existing document embeddings, which is very fast.
        document_store.update_embeddings(retriever)

        document_store.save("faiss_document_store.faiss")


  #  # 1. Setup document store
  #  if os.path.exists("faiss_document_store.json"):
  #      print("Path exists")
  #      document_store = FAISSDocumentStore(
  #          faiss_index_path="faiss_document_store.faiss",
  #          faiss_config_path="faiss_document_store.json")
  #  else:
  #      print("path does not exist")
  #      document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)
  #      document_store.save("faiss_document_store.faiss")

        #document_store.save("faiss_index")
       # document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
    #document_store = InMemoryDocumentStore(use_bm25=True)

    # 2. Put files in habitus folder into a list for indexing
    #files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]

    # 3. Set up text indexing pipline and index all files in folder
    #indexing_pipeline = TextIndexingPipeline(document_store)
    #indexing_pipeline.run_batch(file_paths=files_to_index)

        # New combined pipeline
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(component=shaper, name="shaper", inputs=["Retriever"])
    pipe.add_node(component=exemplifier, name="exemplifier", inputs=['shaper'])

    return pipe

