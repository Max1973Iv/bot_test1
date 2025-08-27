import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import re
import requests
from openai import AsyncOpenAI
from dotenv import load_dotenv
#
# Инициализация переменных окружения
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#
# Запрос к OpenAI
async def answer(system,            # системный промпт
                          user_query,        # запрос пользователя
                          model='gpt-4o-mini'):
#
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query}]
    response = await AsyncOpenAI().chat.completions.create(
       model=model,
        messages=messages)
    return response.choices[0].message.content
#