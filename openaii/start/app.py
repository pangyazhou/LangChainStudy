from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

"""
基础功能入门
"""

llm = ChatOpenAI(openai_api_key="sk-xxx")


# 简单使用
def simple():
    llm.invoke("how can langsmith help with testing?")


# 使用提示词
def prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    chain = prompt | llm
    result = chain.invoke({"input": "how can langsmith help with testing?"})
    print(result)


# 使用输出解析器
def output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    outparser = StrOutputParser()
    chain = prompt | llm | outparser
    result = chain.invoke({"input": "how can langsmith help with testing?"})
    print(result)


if __name__ == '__main__':
    # simple()
    # prompts()
    output_parser()
