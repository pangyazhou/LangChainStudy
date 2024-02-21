"""
SemanticSimilarityExampleSelector
该对象根据与输入的相似性来选择示例。它通过查找与输入具有最大余弦相似度的嵌入示例来实现这一点
"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    # 使用本地向量库
    Chroma,
    # The number of examples to produce.
    k=1,
)
similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)


def prompt_invoke():
    # Input is a feeling, so should select the happy/sad example
    # Input: happy
    # Output: sad
    print(similar_prompt.format(adjective="worried"))
    # Input is a measurement, so should select the tall/short example
    # Input: tall
    # Output: short
    print(similar_prompt.format(adjective="large"))
    # You can add new examples to the SemanticSimilarityExampleSelector as well
    similar_prompt.example_selector.add_example(
        {"input": "enthusiastic", "output": "apathetic"}
    )
    # Input: enthusiastic
    # Output: apathetic
    print(similar_prompt.format(adjective="passionate"))
    pass


if __name__ == "__main__":
    prompt_invoke()
    pass