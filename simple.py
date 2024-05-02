# RETRIEVE
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

# GRADE
from langchain.prompts import PromptTemplate

# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
#     of a retrieved document to a user question. If the document contains keywords related to the user question, 
#     grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
#     Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
#      <|eot_id|><|start_header_id|>user<|end_header_id|>
#     Here is the retrieved document: \n\n {document} \n\n
#     Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """,
#     input_variables=["question", "document"],
# )
# retrieval_grader = prompt | llm | JsonOutputParser()


# GENERATE
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
local_llm = 'llama3'

llm = ChatOllama(model=local_llm, temperature=0)
rag_chain = prompt | llm | StrOutputParser()

# GRAPH
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):

    question : str
    generation : str
    documents : List[str]

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

# def grade_documents(state):
#     print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#     question = state["question"]
#     documents = state["documents"]
#     
#     # Score each doc
#     filtered_docs = []
#     web_search = "No"
#     for d in documents:
#         score = retrieval_grader.invoke({"question": question, "document": d.page_content})
#         grade = score['score']
#         # Document relevant
#         if grade.lower() == "yes":
#             print("---GRADE: DOCUMENT RELEVANT---")
#             filtered_docs.append(d)
#         # Document not relevant
#         else:
#             print("---GRADE: DOCUMENT NOT RELEVANT---")
#             # We do not include the document in filtered_docs
#             # We set a flag to indicate that we want to run web search
#             web_search = "Yes"
#             continue
#     return {"documents": filtered_docs, "question": question, "web_search": web_search}
    

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


# RUN

state = {"question": "What is the meaning of life"}
state = retrieve(state)
print(state["documents"])
state = generate(state)
print(state["generation"])

