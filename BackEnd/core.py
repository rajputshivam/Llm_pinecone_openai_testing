from typing import Any

from langchain_openai import OpenAIEmbeddings,ChatOpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import Pinecone as PineConeLangChain
# from pinecone import Pinecone
# from langchain.vectorstores import Pinecone
# from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# pc = Pinecone(
#     api_key=os.environ.get(["PINECONE_API_KEY"])
# )

# pc = pinecone.init(api_key=os.environ["PINECONE_API_KEY"])
os.environ['PINECONE_API_KEY'] = "614e1ab1-8eea-44aa-ae1f-44f0c384b451"

from consts import INDEX_NAME


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True)

    return qa({"query": query})


if __name__ == '__main__':
    print(run_llm(query="what is RetrievalQA chain"))
