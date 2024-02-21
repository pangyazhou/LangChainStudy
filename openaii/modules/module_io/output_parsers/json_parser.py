"""
该输出解析器允许用户指定任意 JSON 模式并查询 LLM 以获取符合该模式的输出。
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

model = ChatOpenAI()


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


def json_parser_invoke():
    # And a query intented to prompt a language model to populate the data structure.
    joke_query = "Tell me a joke."

    # Set up a parser + inject instructions into the prompt template.
    # parser = JsonOutputParser(pydantic_object=Joke)
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    output = chain.invoke({"query": joke_query})
    print(output)
    # 该输出解析器支持流式输出
    for s in chain.stream({"query": joke_query}):
        print(s)




# 程序入口
if __name__ == "__main__":
    json_parser_invoke()
    pass