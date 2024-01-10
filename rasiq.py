
import os
import openai

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ.get("OPENAI_API_KEY")

#load llm
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model='gpt-4',temperature=0)

from langchain.document_loaders import TextLoader
loader = TextLoader("ChatWithData/BL3122.txt")
document = loader.load()
print(len(document))
print(type(document))

content = " "
i = 0
while i<len(document):
    pages = (document[i].page_content)
    content = content + pages
    ("length of page",i,len(pages))
    ("page", i, "=", pages)
    ("---------------------------------")
    i+=1
print((len(content)))
print(type(content))

from langchain.text_splitter import RecursiveCharacterTextSplitter
chunk_size = 1000
chunk_overlap = 0
length_function = len

r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,)
chunks = r_splitter.split_text(content)
print(len(chunks))

from langchain.embeddings import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

#prompttemplate and instruction to llm
template = '''From the {pages} answer {questions}
'''
pages = chunks
questions = input("Ask anything: ")

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template=template, input_variables=['pages','questions'])

#vector store and retrieval chain

from langchain.vectorstores import FAISS
from langchain.chains import LLMChain

vectordb = FAISS.from_texts(texts=chunks, embedding=embedding)
#launching the QA chain
chain1 = LLMChain(llm=llm, prompt=prompt)
qa = chain1.run({'pages': chunks, 'questions': questions})
print(qa)