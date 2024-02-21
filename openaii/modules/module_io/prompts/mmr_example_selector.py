"""
MaxMarginalRelevanceExampleSelector
根据与输入最相似的示例的组合来选择示例，
同时还针对多样性进行优化。
它通过查找与输入具有最大余弦相似度的嵌入示例来实现这一点，然后迭代地添加它们，
同时惩罚它们与已选择示例的接近程度
"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


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

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    # 嵌入模型 使用OpenAI提供的，环境中需要有 OPENAI_API_KEY=xxxx
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    # 向量数据库  本地临时库
    FAISS,
    # The number of examples to produce.
    k=2,
)
mmr_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)


def prompt_invoke():
    # Input is a feeling, so should select the happy/sad example as the first one
    print(mmr_prompt.format(adjective="worried"))
    pass


if __name__ == "__main__":
    prompt_invoke()
    pass