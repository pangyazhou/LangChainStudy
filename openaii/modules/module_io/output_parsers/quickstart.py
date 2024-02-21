"""
输出解析器是帮助构建语言模型响应的类。
输出解析器必须实现两个主要方法：

1. “Get format instructions”：返回一个字符串的方法，其中包含有关如何格式化语言模型输出的指令。
2. “Parse”：一种接收字符串（假设是语言模型的响应）并将其解析为某种结构的方法。
"""
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field, validator



# 语言模型对象
model = ChatOpenAI()


# 输出解析器的主要类型， PydanticOutputParser.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field

def pydantic_output_parser_invoke():
    parser = PydanticOutputParser(pydantic_object=Joke)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # And a query intended to prompt a language model to populate the data structure.
    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"query": "Tell me a joke."})
    result = parser.invoke(output)
    print(result)


# 程序入口
if __name__ == "__main__":
    pydantic_output_parser_invoke()
    pass