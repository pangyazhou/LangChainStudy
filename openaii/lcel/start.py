# Requires:
# pip install langchain docarray tiktoken

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-3.5-turbo")


def simple():
    prompt = ChatPromptTemplate.from_template("给我讲一个关于 {topic} 的笑话")
    # prompt_value = prompts.invoke({"topic": "中国足球"})
    # print(prompt_value)
    # print(prompt_value.to_messages())
    # print(prompt_value.to_string())

    # message = llm.invoke(prompt_value)
    # print(message)

    output_parser = StrOutputParser()
    # text = output_parser.invoke(message)
    # print(text)

    chain = prompt | llm | output_parser

    text = chain.invoke({"topic": "中国足球"})
    print(text)


# RAG检索
def rag_search():
    #  pip install pydantic==1.10.8
    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | llm | output_parser

    message = chain.invoke("where did harrison work?")
    print(message)


# 程序入口
if __name__ == "__main__":
    simple()
    # rag_search()
