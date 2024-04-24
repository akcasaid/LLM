# -*- coding: utf-8 -*-
"""
Retrieval_Augmented_Generation.py
"""

!pip install -q langchain
!pip install -q torch
!pip install -q transformers
!pip install -q sentence-transformers
!pip install -q datasets
!pip install -q faiss-cpu
!pip install -q openai

from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
from transformers import pipeline, BertTokenizer, BertModel
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import torch

dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "context"

# Create loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# load the data and display the first 15 entries
data = loader.load()
data[:2]

# Create an instance of RecursiveCharacterTextSplitter class
# with specific parameters

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = text_splitter.split_documents(data)

docs[0]

model_path = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name = model_path,      # provide pretrained model path
    model_kwargs = model_kwargs,  # pass model config options
    encode_kwargs = encode_kwargs # pass encoding options
)

text = 'I love METU YZT!'
query_result = embeddings.embed_query(text)
print(query_result[:3])  # see the first three dimensions
print(len(query_result)) # see the vector size

def calculate_semantic_similarity(sentence1, sentence2):

    embeddings1 = embeddings.embed_query(sentence1)
    embeddings2 = embeddings.embed_query(sentence2)

    embeddings1 = torch.tensor(embeddings1)
    embeddings2 = torch.tensor(embeddings2)

    # TODO: Calculate cosine similarity between the embeddings, remember the formula
    dot_product = torch.dot(embeddings1, embeddings2)
    norm1 = torch.norm(embeddings1)
    norm2 = torch.norm(embeddings2)
    similarity = dot_product / (norm1 * norm2)
    ########

    similarity = torch.clamp(similarity, -1.0, 1.0) # for guaranteeing the interval, optional

    return similarity

def calculate_semantic_similarity_answer(sentence1, sentence2):

    embeddings1 = embeddings.embed_query(sentence1)
    embeddings2 = embeddings.embed_query(sentence2)

    embeddings1 = torch.tensor(embeddings1)
    embeddings2 = torch.tensor(embeddings2)

    # TODO: Calculate cosine similarity between the embeddings, remember the formula
    dot_product = torch.dot(embeddings1, embeddings2)
    norm1 = torch.norm(embeddings1)
    norm2 = torch.norm(embeddings2)
    similarity = dot_product / (norm1 * norm2)
    ########

    similarity = torch.clamp(similarity, -1.0, 1.0) # for guaranteeing the interval, optional

    return similarity

sentence1 = "I love METU YZT!"
sentence2 = "METU YZT is my favorite!"

sentence3 = "I don't like orange."
sentence4 = "I don't like orange."

semantic_similarity1 = calculate_semantic_similarity(sentence1, sentence2)
print(f"Semantic Similarity: {semantic_similarity1}")

semantic_similarity2 = calculate_semantic_similarity(sentence3, sentence4)
print(f"Semantic Similarity: {semantic_similarity2}")

docs = docs[:10000]
vectorDB = FAISS.from_documents(docs, embeddings)

# this part may take some time, be patient :)

question = "When was lamborghini sold to the Volkswagen Group?"
searchDocs = vectorDB.similarity_search(question)
print(searchDocs[0])

model_name = "Intel/dynamic_tinybert"

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          padding = True,
                                          truncation = True,
                                          max_length = 512)

question_answerer = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=tokenizer,
    return_tensors='pt'
)

question_answerer(question = "What is the best society in METU?",
                  context = "There are more than 50 societies in our university, but YZT is by far the best.")

retriever = vectorDB.as_retriever()

docs = retriever.get_relevant_documents("When was Lamborghini sold to the Volkswagen Group?")
print(docs[0])

retriever = vectorDB.as_retriever(search_kwagrs={"k":4})

# completion llm
llm = ChatOpenAI(
    openai_api_key="api_key",
    model_name='gpt-3.5-turbo',
    temperature=0.1 # allow the model to be more generative
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

question = "Who is Thomas Jefferson?"
result = qa.run(question)
print(result)

docs = retriever.get_relevant_documents(question)
print(docs[0])
