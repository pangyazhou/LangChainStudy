"""
涵盖了如何将HTML文档加载为我们可以在下游使用的文档格式。
"""
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader


def html_loader_invoke():
    loader = UnstructuredHTMLLoader("example_data/fake-content.html")
    docs = loader.load()
    print(len(docs))
    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)


# 使用BSHTMLLoader加载HTML文档
# 标题写入到metadata中
def bs_html_loader_invoke():
    loader = BSHTMLLoader("example_data/fake-content.html")
    docs = loader.load()
    print(len(docs))
    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)     # {'source': 'example_data/fake-content.html', 'title': 'HTML鍏ラ棬绀轰緥'}


# 程序入口
if __name__ == "__main__":
    # html_loader_invoke()
    bs_html_loader_invoke()
    pass