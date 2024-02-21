"""
How to use ChatModels that support function calling

某些聊天模型（例如 OpenAI）具有函数调用 API，可让您描述函数及其参数，并让模型返回一个 JSON 对象，其中包含要调用的函数以及该函数的输入。
函数调用对于构建使用工具的链和代理以及更普遍地从模型中获取结构化输出非常有用。
"""


# 乘法函数
import json
from typing import Type, Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool, convert_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


model = ChatOpenAI(model="gpt-3.5-turbo")



def multiply(a: int, b: int) -> int:
    """Multiply two integers together.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


def python_function_invoke():
    result = json.dumps(convert_to_openai_tool(multiply), indent=2)
    print(result)


class MultiplyPydantic(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


def pydantic_class_invoke():
    result = json.dumps(convert_to_openai_tool(MultiplyPydantic), indent=2)
    print(result)


"""
LangChain Tool
"""
class MultiplySchema(BaseModel):
    """Multiply tool schema."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseTool):
    args_schema: Type[BaseModel] = MultiplySchema
    name: str = "multiply"
    description: str = "Multiply two integers together."

    def _run(self, a: int, b: int, **kwargs: Any) -> Any:
        return a * b


def langchain_tool_invoke():
    print(json.dumps(convert_to_openai_tool(Multiply()), indent=2))


"""
绑定函数到模型
"""
def binding_functions_invoke():
    # 调用模型时指定绑定的函数
    result = model.invoke("what's 5 times three", tools=[convert_to_openai_tool(multiply)])
    print(result)

    # 函数绑定到模型工具
    model_with_tool = model.bind(tools=[convert_to_openai_tool(multiply)])
    result = model_with_tool.invoke("what's 5 times three")
    print(result)

    # 强制使用tool_choice参数调用工具
    model_with_tool = model.bind(
        tools=[convert_to_openai_tool(multiply)],
        tool_choice={"type": "function", "function": {"name": "multiply"}}
    )
    result = model_with_tool.invoke(
        "don't answer my question. no do answer my question. no don't. what's five times four"
    )
    print(result)

    # bind_tools函数
    llm_with_tool = model.bind_tools([multiply], tool_choice="multiply")
    result = llm_with_tool.invoke("what's 5 times three")
    print(result)

    # 绑定functions
    llm_with_functions = model.bind(
        functions=[convert_to_openai_function(multiply)], function_call={"name": "multiply"}
    )
    result = llm_with_functions.invoke("what's 3 times a million")
    print(result)

    # bind_functions函数
    llm_with_functions = model.bind_functions([multiply], function_call="multiply")
    result = llm_with_functions.invoke("what's 3 times a million")
    print(result)




# 程序入口
if __name__ == "__main__":
    # python_function_invoke()
    # pydantic_class_invoke()
    # langchain_tool_invoke()
    binding_functions_invoke()
    pass