"""
语言模型的提示是用户提供的一组指令或输入，用于指导模型的响应，帮助模型理解上下文并生成相关且连贯的基于语言的输出，例如回答问题、完成句子或参与某项活动。对话。
本程序介绍LangChain中常用的提词器模板及使用示例
How to use few-shot examples with LLMs
How to use few-shot examples with chat models
How to use example selectors
How to partial prompts
How to work with message prompts
How to compose prompts together
How to create a pipeline prompts
"""
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI()


"""
PromptTemplate使用字符串作为输入，字符串作为输出
支持任意数量的参数
PromptTemplate, ChatPromptTemplate 实现Runnable接口，意味着它们支持invoke,ainvoke,stream,astream,batch,abatch,astream_log调用
"""
# PromptTemplate示例
# Tell me a funny joke about chickens.
def prompt_template_invoke():
    prompt_template = PromptTemplate.from_template(
        "Tell me a {adjective} joke about {content}."
    )
    prompt_value = prompt_template.format(adjective="funny", content="chickens")
    print(prompt_value)
    # invoke 调用
    prompt_value = prompt_template.invoke({"adjective": "funny", "content": "chickens"})
    print(prompt_value)         # text='Tell me a funny joke about chickens.'
    # 转化为message
    msg = prompt_value.to_messages()
    print(msg)                  # [HumanMessage(content='Tell me a funny joke about chickens.')]



"""
ChatPromptTemplate使用chat messages 列表作为输入
每个chat message关联内容与附加参数role。
OpenAI的API中，一个chat message包括AI role， human role, system role
"""
# ChatPromptTemplate示例
def chat_prompt_template_invoke():
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"),
        ]
    )
    messages = chat_prompt.format_messages(name="Bob", user_input="What is your name")
    print(messages)
    # invoke 调用
    chat_prompt_value = chat_prompt.invoke({"name": "Bob", "user_input": "What is your name"})
    print(chat_prompt_value.to_messages())    # [SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), HumanMessage(content='Hello, how are you doing?'), AIMessage(content="I'm doing well, thanks!"), HumanMessage(content='What is your name')]
    print(chat_prompt_value.to_string())
    # System: You are a helpful AI bot. Your name is Bob.
    # Human: Hello, how are you doing?
    # AI: I'm doing well, thanks!
    # Human: What is your name
    print(chat_prompt_value.json())    # {"messages": [{"content": "You are a helpful AI bot. Your name is Bob.", "additional_kwargs": {}, "type": "system"}, {"content": "Hello, how are you doing?", "additional_kwargs": {}, "type": "human", "example": false}, {"content": "I'm doing well, thanks!", "additional_kwargs": {}, "type": "ai", "example": false}, {"content": "What is your name", "additional_kwargs": {}, "type": "human", "example": false}]}


"""
字符串提示词
"""
def string_prompt_invoke():
    prompt = (
        PromptTemplate.from_template("Tell me a joke about {topic}")
        + ", make it funny"
        + "\n\nand in {language}"
    )
    print(prompt)   # input_variables=['language', 'topic'] template='Tell me a joke about {topic}, make it funny\n\nand in {language}'
    prompt_value = prompt.format(topic="sports", language="chinese")
    print(prompt_value)     # Tell me a joke about sports, make it funny\n\nand in chines
    # 用于ChatOpenAI模型
    chain = LLMChain(llm=model, prompt=prompt)
    result = chain.invoke({"topic": "足球", "language": "chinese"})
    print(result)       # {'topic': '足球', 'language': 'chinese', 'text': '为什么足球比赛中的草地总是那么幸福？因为它总是被球员踩！'}


"""
Chat Prompt
您可以轻松创建将其与其他消息或消息模板相结合的管道
当没有要格式化的变量时使用 Message
当有要格式化的变量时使用 MessageTemplate
您还可以仅使用一个字符串  (注意：这将自动推断为 HumanMessagePromptTemplate)
"""
def chat_prompt_invoke():
    prompt = SystemMessage(content="You are a nice pirate")
    print(prompt)
    new_prompt = (
        prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
    )
    print(new_prompt)
    prompt_value = new_prompt.format_messages(input="i said hi")
    print(prompt_value)

    chain = LLMChain(llm=model, prompt=new_prompt)
    result = chain.invoke({"input": "i said hi"})
    print(result)           # {'input': 'i said hi', 'text': 'Hello! How can I help you today?'}




# 程序入口
if __name__ == "__main__":
    # prompt_template_invoke()
    # chat_prompt_template_invoke()
    # string_prompt_invoke()
    chat_prompt_invoke()

    pass