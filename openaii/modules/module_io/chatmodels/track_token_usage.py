"""
How to track token usage in a ChatModel call
介绍了如何跟踪特定调用的令牌使用情况。
目前仅针对 OpenAI API 实现。
"""
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI


model = ChatOpenAI(model="gpt-3.5-turbo")



# 最终token调用
# Tokens Used: 158
# 	Prompt Tokens: 14
# 	Completion Tokens: 144
# Successful Requests: 1
# Total Cost (USD): $0.00030900000000000003
def track_token_invoke():
    with get_openai_callback() as cb:
        model.invoke("给我讲一个关于中国足球的笑话")
        # model.invoke("给我讲一个笑话")
        print(cb)



# 程序入口
if __name__ == "__main__":
    track_token_invoke()
    pass
