"""
langchain流式输出示例
"""
import asyncio
import time

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from collections import deque

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt_template = "给我讲一个关于{topic}的笑话"
prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()
json_parser = JsonOutputParser()

chunks = []


class StreamCallBackHander:
    def __init__(self):
        self.tokens = deque()

    def append_token(self, token):
        self.tokens.append(token)

    def get_tokens(self):
        returnValue = []
        while self.tokens:
            returnValue.append(self.tokens.popleft())
        return returnValue


class CustomSse:
    def __init__(self, callback: StreamCallBackHander):
        self.callback = callback

    def event_source(self):
        while True:
            tk = self.callback.get_tokens()
            if len(tk) > 0:
                yield f'{"".join(tk)}'


# LCEL实现流式输出
def stream_lcel(topic: str, callback: StreamCallBackHander):
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    for chunk in chain.stream(topic):
        callback.append_token(chunk)
        # chunks.append(chunk)
        print(chunk, end="", flush=True)



# 流式输入
# 显示结果无
async def stream_input():
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | json_parser)
    async for text in chain.astream("中国足球"):
        print(text, flush=True)


# 非流式组件
def nonstream_retriever():
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    vectorstore = FAISS.from_texts(
        ["harrison worked at kensho", "harrison likes spicy food"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # chunks = [chunk for chunk in retriever.stream("where did harrison work?")]
    # print(chunks)
    retrieval_chain = (
            {
                "context": retriever.with_config(run_name="Docs"),
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
            | StrOutputParser()
    )
    for chunk in retrieval_chain.stream(
            "Where did harrison work? " "Write 3 made up sentences about this place."
    ):
        print(chunk, end="|", flush=True)


# 程序入口
if __name__ == "__main__":
    # stream_lcel()
    # asyncio.run(stream_input())
    # nonstream_retriever()
    pass
