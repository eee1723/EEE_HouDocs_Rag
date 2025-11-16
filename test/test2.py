from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate



msg = [
    SystemMessage("你是一个特效大师，叫小鹅"),
    HumanMessage("你知道怎么做{fx}吗"),
]

prompt = ChatPromptTemplate(msg)
a = prompt.format_prompt({"fx":"超级大骇浪"})

from langchain_openai import ChatOpenAI
import os
import dotenv

dotenv.load_dotenv()
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    model = "gpt-5-mini",
    temperature=0
    # stream_usage=True,
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instead of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

response = llm.invoke(a )
print(response)