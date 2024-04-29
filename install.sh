#!/bin/sh

python -m venv .
./bin/pip install -U langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph tavily-python 

./bin/pip install bs4
./bin/pip install gpt4all
./bin/pip install os

# Emmbbeds = text to vector
./bin/pip install langchain-nomic 

ollama pull llama3
