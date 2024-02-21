"""
可用于将 LLM 输出解析为日期时间格式。
"""
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI()


def datetime_output_parser_invoke():
    output_parser = DatetimeOutputParser()
    template = """Answer the users question:

    {question}

    {format_instructions}"""
    prompt = PromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    print(prompt)
    chain = prompt | model | output_parser
    output = chain.invoke({"question": "when was bitcoin founded?"})
    print(output)       # 2009-01-03 18:15:05

# 程序入口
if __name__ == "__main__":
    datetime_output_parser_invoke()
    pass