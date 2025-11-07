import dotenv
import os
dotenv.load_dotenv()


from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model = "gpt-5-mini",

)

response = llm.invoke("说一个笑话")
print(response)