"""
如何在聊天模型中使用少量示例
"""
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

model = ChatOpenAI()

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]
# 固定示例
# 使用提供的全部示例
def fixed_example_invoke():
    # This is a prompts template used to format each individual example.
    # 提示词模板 用于格式化每个示例
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    print(few_shot_prompt.format())

    # 集成进最终的提示词中并在模型中使用
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a wondrous wizard of math."),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    chain = final_prompt | model
    result = chain.invoke({"input": "What's the square of a triangle?"})
    print(result)


# 动态使用示例
# 根据示例选择器基于输入选择将要使用的示例
def dynamic_example_invoke():
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
        {"input": "2+4", "output": "6"},
        {"input": "What did the cow say to the moon?", "output": "nothing at all"},
        {
            "input": "Write me a poem about the moon",
            "output": "One for the moon, and one for me, who are we to talk about the moon?",
        },
    ]
    # 连接嵌入模型与向量数据库
    to_vectorize = [" ".join(example.values()) for example in examples]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
    # 选择相似度最高的2个示例
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    # The prompts template will load examples by passing the input do the `select_examples` method
    # 选择示例
    selected_examples = example_selector.select_examples({"input": "horse"})
    print(selected_examples)
    # Define the few-shot prompts.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input"],
        example_selector=example_selector,
        # Define how each example will be formatted.
        # In this case, each example will become 2 messages:
        # 1 human, and 1 AI
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )
    print(few_shot_prompt.format(input="What's 3+3?"))

    # 基于提示词使用模型
    chain = few_shot_prompt | model
    result = chain.invoke({"input": "tell me  3+3?"})
    print(result)


# 程序入口
if __name__ == "__main__":
    # fixed_example_invoke()
    dynamic_example_invoke()
    pass