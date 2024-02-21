"""
分步格式化提示词
1. 使用字符串分步格式化
2. 使用函数(返回字符串)分步格式化
"""
from datetime import datetime

from langchain_core.prompts import PromptTemplate


# 使用字符串分步格式化
def partial_with_strings():
    prompt = PromptTemplate(template="{foo}, {bar}", input_variables=["foo", "bar"])
    partial_prompt = prompt.partial(foo="hello")
    print(partial_prompt.format(bar="world"))


# 使用函数分步格式化
def partial_with_function():
    prompt = PromptTemplate(
        template="Tell me a {adjective} joke about the day {date}",
        input_variables=["adjective", "date"],
    )
    partial_prompt = prompt.partial(date=_get_datetime())
    print(partial_prompt.format(adjective="funny"))


# 日期获取函数
def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")


# 程序入口
if __name__ == "__main__":
    # partial_with_strings()
    partial_with_function()
    pass