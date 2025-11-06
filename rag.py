import os
import dotenv
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import HTMLSemanticPreservingSplitter

# 设置大模型APIkey
dotenv.load_dotenv()
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")





# 文档切分
from bs4 import Tag
from langchain_text_splitters import HTMLSemanticPreservingSplitter

# 使用requests获取HTML内容
url = "https://www.sidefx.com/docs/houdini/index.html"
response = requests.get(url)
html_string = response.text  # 这是HTML源码字符串
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]


# def code_handler(element: Tag) -> str:
#     data_lang = element.get("data-lang")
#     code_format = f"<code:{data_lang}>{element.get_text()}</code>"

#     return code_format


splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=headers_to_split_on,
    separators=["\n\n", "\n", ". ", "! ", "? "],
    max_chunk_size=50,
    preserve_images=True,
    preserve_videos=True,
    elements_to_preserve=["table", "ul", "ol", "code"],
    denylist_tags=["script", "style", "head"],
)

documents = splitter.split_text(html_string)


# 加载嵌入模型
from langchain_openai import OpenAIEmbeddings
import chromadb

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 向量存储
from langchain_chroma import Chroma

client = chromadb.PersistentClient(path="./chroma_langchain_db")
vector_store_from_client = Chroma(
    client=client,
    collection_name="collection_name",
    embedding_function=embeddings
)

vector_store_from_client.add_documents(documents=documents)
results = vector_store_from_client.similarity_search(
    "How to use the network",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# 定义提示词模板
prompt = ChatPromptTemplate.from_template('''
    你是一个问答机器人，你的任务是根据下面给定的已知信息回答用户提出的问题：
    已知信息：
    {context} 
    用户的问题是：
    {question}
    如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请先回答“根据已知的信息，我无法回答您的问题。我只能通过推理回答：”
    然后通过自己的推理推理，给出你的回答。
    如果已知信息包含用户问题的答案，请直接回答用户的问题，绝不可以编造任何不确定的信息。
    ''')

# def query_vector(info):
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     docs = retriever.get_relevant_documents(info["question"])

# 初始化大模型
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)


# 定义输出解析器
output_parser = StrOutputParser()

# 测试大模型问答
chain = prompt | model | output_parser
# {"context" : query_vector, "question" : lambda x : x["question"]}
# respons = chain.stream({"question": "In Houdini, How can i create a lightning?"})
# for r in respons:
#     print(r, end='', flush=True)