"""
流式事件示例
"""
import asyncio

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt_template = "给我讲一个关于{topic}的笑话"
prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()
json_parser = JsonOutputParser()


# 简单示例
async def start():
    events = []
    async for event in model.astream_events("hello", version="v1"):
        events.append(event)
    print(events[:3])


# 解析json流
async def stream_json():
    chain = model | json_parser
    events = [
        event
        async for event in chain.astream_events(
            'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict '
            'with an outer key of "countries" which contains a list of countries. Each country should have the key '
            '`name` and `population`',
            version="v1",
        )
    ]
    print(events[:3])


# 程序入口
if __name__ == "__main__":
    # asyncio.run(start())
    asyncio.run(stream_json())
    pass