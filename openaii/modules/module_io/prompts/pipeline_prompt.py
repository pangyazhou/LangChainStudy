"""
本程序介绍了如何将多个提示组合在一起。
当您想要重复使用部分提示时，这会很有用。
这可以通过 PipelinePrompt 来完成
"""

from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate


def pipeline_prompt_invoke():
    full_template = """{introduction}

    {example}

    {start}"""
    full_prompt = PromptTemplate.from_template(full_template)

    introduction_template = """You are impersonating {person}."""
    introduction_prompt = PromptTemplate.from_template(introduction_template)

    example_template = """Here's an example of an interaction:

    Q: {example_q}
    A: {example_a}"""
    example_prompt = PromptTemplate.from_template(example_template)

    start_template = """Now, do this for real!

    Q: {input}
    A:"""
    start_prompt = PromptTemplate.from_template(start_template)

    input_prompts = [
        ("introduction", introduction_prompt),
        ("example", example_prompt),
        ("start", start_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts
    )

    print(pipeline_prompt.input_variables)
    prompt = pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
    print(prompt)





# 程序入口
if __name__ == "__main__":
    pipeline_prompt_invoke()
    pass