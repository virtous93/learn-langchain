import os
import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The key value is left blank, so you must set it when executing.
os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt_template = (
    "You are a knowledgeable assistant. Given the following context, answer the question:\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_prompt(context, question):
    return prompt_template.format(context=context, question=question)

# Retrieving context and formatting it with the question.
def create_input(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    context = format_docs(retrieved_docs)
    return {'context': context, 'question': question}

inputs = create_input("What is Task Decomposition?")
formatted_prompt = format_prompt(inputs['context'], inputs['question'])

output = llm.invoke(formatted_prompt)
parsed_output = StrOutputParser().invoke(output)
print(parsed_output)
