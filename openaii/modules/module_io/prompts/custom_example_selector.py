"""
自定义示例选择器
如果您有大量示例，您可能需要选择要包含在提示中的示例。
示例选择器是负责执行此操作的类。
"""

# 示例列表   英语翻译成意大利语
from typing import Dict, List, Any

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "hi", "output": "ciao"},
    {"input": "bye", "output": "arrivaderci"},
    {"input": "soccer", "output": "calcio"},
]


# 自定义示例选择器  根据单词长度选择示例
class CustomExampleSelector(BaseExampleSelector):
    # 构造方法
    def __init__(self, examples: List[dict]):
        self.examples = examples

    def add_example(self, example: Dict[str, str]) -> Any:
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        # 这里假设输入是字符串
        new_word = input_variables["input"]
        new_word_length = len(new_word)

        # 存储最佳匹配及其与输入的长度差
        best_match = None
        smallest_diff = float("inf")

        # 遍历示例
        for example in self.examples:
            # 计算每个示例与输入的长度差
            current_diff = abs(len(example["input"]) - new_word_length)

            # 寻找最小长度差的示例
            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example
        return [best_match]


def selector_invoke():
    example_selector = CustomExampleSelector(examples)
    example = example_selector.select_examples({"input": "okay"})
    print(example)

    example_selector.add_example({"input": "hand", "output": "mano"})
    example = example_selector.select_examples({"input": "okay"})
    print(example)

    # 用于提示词
    example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="Input: {input} -> Output:",
        prefix="Translate the following words from English to Italian",
        input_variables=["input"]
    )
    print(prompt.format(input="word"))
    pass


# 程序入口
if __name__ == "__main__":
    selector_invoke()
    pass
