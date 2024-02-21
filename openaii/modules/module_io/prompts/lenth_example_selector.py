"""
LengthBasedExampleSelector
长度匹配示例选择器
此示例选择器根据长度选择要使用的示例。
当您担心构建的提示会超过上下文窗口的长度时，这非常有用。
对于较长的输入，它将选择较少的示例来包含，而对于较短的输入，它将选择更多的示例。

样例选择器： 长度选择器
指定样例长度限制，输入长度小的时候，提示词多一些样例。输入长度大的时候，提示词少一些样例
"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The maximum length that the formatted examples should be.
    # Length is measured by the get_text_length function below.
    max_length=25,
    # The function used to get the length of a string, which is used
    # to determine which examples to include. It is commented out because
    # it is provided as a default value if none is specified.
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)


# 长度样例提示词选择器使用示例
def prompt_invoke():
    # An example with small input, so it selects all examples.
    # 给出一个短的输入， 提示词选择全部样例
    print(dynamic_prompt.format(adjective="big"))
    # An example with long input, so it selects only one example.
    # 给出一个长的输入， 提示词只选择一个样例
    long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
    print(dynamic_prompt.format(adjective=long_string))
    # You can add an example to an example selector as well.
    # 可以给样例选择器添加样例
    new_example = {"input": "big", "output": "small"}
    dynamic_prompt.example_selector.add_example(new_example)
    print(dynamic_prompt.format(adjective="enthusiastic"))

    chain = dynamic_prompt | model
    result = chain.invoke({"adjective": "enthusiastic"})
    print(result)


if __name__ == "__main__":
    prompt_invoke()
    pass