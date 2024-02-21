"""
展示如何使用 Enum 输出解析器。
"""
from enum import Enum

from langchain.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI()


class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    BLACK = "black"


def enum_parser_invoke():
    parser = EnumOutputParser(enum=Colors)

    prompt = PromptTemplate.from_template(
        """What color eyes does this person have?

    > Person: {person}

    Instructions: {instructions}"""
    ).partial(instructions=parser.get_format_instructions())
    chain = prompt | model | parser
    output = chain.invoke({"person": "Frank Sinatra"})
    print(output)



# 程序入口
if __name__ == "__main__":
    enum_parser_invoke()
    pass