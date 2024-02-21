"""
NGramOverlapExampleSelector
根据 ngram 重叠分数，根据与输入最相似的示例来选择示例并对其进行排序。
ngram 重叠分数是 0.0 到 1.0 之间的浮点数（含 0.0 和 1.0）。

选择器允许设置阈值分数。ngram 重叠分数小于或等于阈值的示例被排除。
默认情况下，阈值设置为 -1.0，因此不会排除任何示例，只会对它们重新排序。
将阈值设置为 0.0 将排除与输入没有 ngram 重叠的示例。

样例中的样例进行与输入的相关度排序
"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector


# Examples of a fictional translation task.
examples = [
    {"input": "See Spot run.", "output": "Ver correr a Spot."},
    {"input": "My dog barks.", "output": "Mi perro ladra."},
    {"input": "Spot can run.", "output": "Spot puede correr."},
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

example_selector = NGramOverlapExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The threshold, at which selector stops.
    # It is set to -1.0 by default.
    threshold=-1.0,
    # For negative threshold:
    # Selector sorts examples by ngram overlap score, and excludes none.
    # For threshold greater than 1.0:
    # Selector excludes all examples, and returns an empty list.
    # For threshold equal to 0.0:
    # Selector sorts examples by ngram overlap score,
    # and excludes those with no ngram overlap with input.
)
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the Spanish translation of every input",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)


def prompt_invoke():
    # An example input with large ngram overlap with "Spot can run."
    # and no overlap with "My dog barks."
    print(dynamic_prompt.format(sentence="Spot can run fast."))

    # You can add examples to NGramOverlapExampleSelector as well.
    new_example = {"input": "Spot plays fetch.", "output": "Spot juega a buscar."}
    example_selector.add_example(new_example)
    print(dynamic_prompt.format(sentence="Spot can run fast."))

    # You can set a threshold at which examples are excluded.
    # For example, setting threshold equal to 0.0
    # excludes examples with no ngram overlaps with input.
    # Since "My dog barks." has no ngram overlaps with "Spot can run fast."
    # it is excluded.
    # 不相关的不展示
    example_selector.threshold = 0.0
    print(dynamic_prompt.format(sentence="Spot can run fast."))

    # Setting small nonzero threshold
    # 相识度低于一定程度不显示
    example_selector.threshold = 0.09
    print(dynamic_prompt.format(sentence="Spot can play fetch."))

    # Setting threshold greater than 1.0
    # 大于1.0 所有都不显示
    example_selector.threshold = 1.0 + 1e-9
    print(dynamic_prompt.format(sentence="Spot can play fetch."))
    pass


if __name__ == "__main__":
    prompt_invoke()
    pass