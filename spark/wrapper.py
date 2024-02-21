from langchain_xfyun.chat_models import ChatSpark
from langchain_xfyun.prompts import ChatPromptTemplate
from langchain_xfyun.chains import LLMChain

#以下密钥信息从控制台获取
appid = "fba38362"     #填写控制台中获取的 APPID 信息
api_secret = "xxx"   #填写控制台中获取的 APISecret 信息
api_key ="xxx"    #填写控制台中获取的 APIKey 信息

llm = ChatSpark(app_id=appid, api_key=api_key,
                api_secret=api_secret)

prompt = ChatPromptTemplate.from_template(
    "我有一个生产[{product}]商品的公司，请帮我取一个最合适的公司名称。只输出答案本身"
)

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

product = "魔方"
ans = chain.run(product)
print(ans)