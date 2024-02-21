"""
本程序文件将涵盖使用语言模型的基础知识。
它将介绍两种不同类型的模型 - LLM 和 ChatModel。
然后，它将介绍如何使用 PromptTemplates 格式化这些模型的输入，以及如何使用输出解析器处理输出
"""
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

"""
系统环境中已经配置了 OPENAI_API_KEY=sk-xxx
如果您不想设置环境变量，可以openai_api_key在启动 OpenAI LLM 类时直接通过命名参数传递密钥:
llm = ChatOpenAI(openai_api_key="sk-xxx")
1. LLM 对象将字符串作为输入和输出字符串
2. ChatModel 对象将消息列表作为输入并输出消息
"""
# LLM模型对象
llm = OpenAI()
# ChatModel模型对象
chat_model = ChatOpenAI()
# 提问问题
text = "给我讲一个关于中国足球的笑话"
messages = [HumanMessage(content=text)]


# LLM模型调用示例
# 有一天，中国足球队比赛输得很惨，教练非常生气，他大声喊道：“你们这群球员，脚下都长了两个左脚吗？怎么就连个球都踢不进去！”球员们尴尬地低下头，其中一位球员小声回答：“教练，其实我们脚下并没有两个左脚，只是我们每个人都有一个左脚和一个右脚，但是我们每次踢球都像是用的两个左脚。”教练无语地摇摇头：“这还不简单，以后每个人都只用一个左脚踢球，那就不会出现这种情况了！”
def llm_invoke():
    result = llm.invoke(text)
    print(result)


# ChatModel模型调用示例
# content='有一天，中国足球队和巴西足球队进行友谊赛，结果中国队被巴西队狂虐10-0。赛后，中国队教练非常气愤地对队员们说：“你们怎么这么没用！连一球都进不了，给我好好反思！”队员们面面相觑，一个小队员突然开口说：“教练，我有个建议，我们下次不如把球门换成方形的！”教练愣了一下，然后问：“为什么要换成方形的？”小队员回答：“这样我们就可以说，我们队比巴西队还牛逼，连球都能踢进方形的球门！”'
def chat_model_invoke():
    result = chat_model.invoke(messages)
    print(result)


"""
提示词模板 (Prompt Templates)
通常，应用会将<用户输入>添加到较大的文本中，称为提示模板，该文本提供有关当前特定任务的附加上下文。
"""
# 提示词示例
# What is a good name for a company that makes colorful socks?
def prompt_invoke():
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    prompt_value = prompt.format(product="colorful socks")
    print(prompt_value)


# 消息列表提示词示例
# [SystemMessage(content='You are a helpful assistant that translates English to French.'), HumanMessage(content='I love programming.')]
def chat_prompt_invoke():
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    human_template = "{text}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chat_prompt_value = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
    print(chat_prompt_value)


"""
输出解析器 (Output Parsers)
OutputParser将语言模型的原始输出转换为可以在下游使用的格式
1. 将LLM输出文本转换为结构化信息（例如 JSON）
2. 将ChatMessage转换为字符串
3. 将除消息之外的调用返回的额外信息（如 OpenAI 函数调用）转换为字符串。
"""
# 输出解析器示例
# ['hi', 'bye']
def output_parser_invoke():
    output_parser = CommaSeparatedListOutputParser()
    result = output_parser.parse("hi, bye")
    print(result)


"""
构建应用链
我们现在可以将所有这些组合成一条链。该链将获取输入变量，将这些变量传递给提示模板以创建提示，将提示传递给语言模型，然后通过（可选）输出解析器传递输出。
"""
# 构建调用链
# ['blue', 'red', 'green', 'yellow', 'orange']
def chain_invoke():
    template = "Generate a list of 5 {text}.\n\n{format_instructions}"
    chat_prompt = ChatPromptTemplate.from_template(template)        # 提示词模板
    output_parser = CommaSeparatedListOutputParser()                # 输出解析器
    chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
    chain = chat_prompt | chat_model | output_parser                # 构建chain
    result = chain.invoke({"text": "colors"})                       # 调用语言模型
    print(result)


# 程序入口
if __name__ == "__main__":
    # llm_invoke()
    # chat_model_invoke()
    # prompt_invoke()
    # chat_prompt_invoke()
    # output_parser_invoke()
    chain_invoke()
