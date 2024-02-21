"""
创建使用少量示例的提示模板。可以从一组示例或示例选择器对象构建少量提示模板
"""
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

"""
使用示例集
创建一个少量示例的列表。每个示例都应该是一个字典，其中键是输入变量，值是这些输入变量的值。
"""
example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template="Question: {question}\n{answer}"
    )
examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]
# 基于示例集创建提示词模板
def example_set_invoke():
    # 创建模板
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template="Question: {question}\n{answer}"
    )
    print(example_prompt.format(**examples[0]))
    # 填充模板
    prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        suffix="Question: {input}",
        input_variables=["input"],
    )
    print(prompt.format(input="Who was the father of Mary Ball Washington?"))
    pass


"""
使用示例选择器
使用前面的示例集投喂SemanticSimilarityExampleSelector示例选择器，
需要嵌入模型计算相似度，使用向量数据库执行临近搜索
"""
def example_selector_invoke():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        OpenAIEmbeddings(),
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma,
        # This is the number of examples to produce.
        k=1,
    )
    # Select the most similar example to the input.
    question = "Who was the father of Mary Ball Washington?"
    selected_examples = example_selector.select_examples({"question": question})
    print(f"Examples most similar to the input: {question}")
    print(selected_examples)

    # 填充模板
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )
    print(prompt.format(input="Who was the father of Mary Ball Washington?"))
    pass


# 程序入口
if __name__ == "__main__":
    # example_set_invoke()
    example_selector_invoke()