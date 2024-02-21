from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from spark.LLMService.SparkLLM import SparkLLM
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


llm = SparkLLM()


# 超长文本总结
def longtext():
    # 导入文本
    loader = UnstructuredFileLoader("E:\\tmp\\text.txt")
    # 文本转为Document对象
    document = loader.load()
    print(f'documents:{len(document)}')

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )

    # 切分文本
    split_documents = text_splitter.split_documents(document)
    print(f'documents:{len(split_documents)}')

    # 加载llm模型
    # 创建总结链
    chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

    # 执行总结
    chain.invoke(split_documents[:5])


# 问答机器人
def robot():
    # 导入文本
    loader = UnstructuredFileLoader("E:\\tmp\\shy\\三禾一员工手册.pdf")
    # 文本转为Document对象
    document = loader.load()
    #print(f'documents:{len(document)}')

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0
    )

    # 切分文本
    split_documents = text_splitter.split_documents(document)
    #print(f'documents:{len(split_documents)}')

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(split_documents, embeddings)

    # 创建问答对象
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    # 进行问答
    result = qa({"query": "三禾一公司怎么样"})
    print(result)


def prompt():
    # llm.api_secret = "ODRhNzU1NWM5NTkxOGRiYTgxNjkxM2Ez"
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    chain.invoke({"input": "如何入职科大讯飞"})
    # print(result)


# 信息检索
def retrieve():
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()


if __name__ == '__main__':
    # longtext()
    # robot()
    prompt()
    pass
