"""
当您想要返回以逗号分隔的项目列表时，可以使用此输出解析器
"""
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 语言模型对象
model = ChatOpenAI()

def comma_separated_output_parser():
    output_parser = CommaSeparatedListOutputParser()

    format_instructions = output_parser.get_format_instructions()
    print(format_instructions)          # Your response should be a list of comma separated values, eg: `foo, bar, baz`
    prompt = PromptTemplate(
        template="List five {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions},
    )
    chain = prompt | model | output_parser
    result = chain.invoke({"subject": "ice cream flavors"})
    print(result)       # ['Vanilla', 'Chocolate', 'Strawberry', 'Mint Chocolate Chip', 'Cookies and Cream']

    for s in chain.stream({"subject": "ice cream flavors"}):
        print(s)

# 程序入口
if __name__ == "__main__":
    comma_separated_output_parser()
    pass