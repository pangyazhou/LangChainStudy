"""
这涵盖了如何将Markdown文档加载为我们可以在下游使用的文档格式
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader


# 加载一个md文档名打印出来
def md_loader_invoke():
    markdown_path = "example_data/deployment.md"
    # mode="elements" 按照段落分割？
    # mode="single"   单个文档
    # mode = "paged"  按照页分割
    loader = UnstructuredMarkdownLoader(markdown_path, mode="paged")
    docs = loader.load()
    for doc in docs:
        # print(doc)
        print("type: ", doc.type)
        print("metadata: ", doc.metadata)
        print("content: ", doc.page_content)


# 程序入口
if __name__ == "__main__":
    md_loader_invoke()
    pass