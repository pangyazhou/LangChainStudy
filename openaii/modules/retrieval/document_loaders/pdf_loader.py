"""
这涵盖了如何将PDF文档加载为我们下游使用的文档格式。
"""
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader, PDFMinerLoader, \
    PyPDFium2Loader, PyMuPDFLoader, PyPDFDirectoryLoader, PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# 加载一个PDF文档，按照页码加载入文档数组
# 这种方法的优点是可以使用页码检索文档。
def pdf_loader_invoke():
    pdf_path = "example_data/from_one_to_infinity.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    for doc in docs:
        print(doc)


# 使用嵌入模型测试
def test_with_embedding():
    pdf_path = "example_data/Ceph集群部署.pdf"
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    docs = faiss_index.similarity_search("怎么把OSD从RADOS集群中移除？", k=2)
    for doc in docs:
        print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


# 从PDF文档中提取图像
def extracting_images_invoke():
    loader = PyPDFLoader("example_data/2103.15348.pdf", extract_images=True)
    pages = loader.load()
    print(pages[4].page_content)
    pass


# 加载非机构化PDF
# 出现错误ImportError: cannot import name 'open_filename' from 'pdfminer.utils'， 执行下面两行命令
# pip uninstall pdfminer
# pip install pdfminer.six
def unstructured_loader_invoke():
    loader = UnstructuredPDFLoader("example_data/2103.15348.pdf")
    pages = loader.load()
    for page in pages:
        print(page)
    pass


# 抓取远程PDF
def fetch_remote_pdf_loader_invoke():
    loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)


# PDFMiner是一个可以从PDF文档中提取信息的工具。
# 与其他PDF相关的工具不同，它注重的完全是获取和分析文本数据。
# PDFMiner允许你获取某一页中文本的准确位置和一些诸如字体、行数的信息。
# 它包括一个PDF转换器，可以把PDF文件转换成HTML等格式。
# 它还有一个扩展的PDF解析器，可以用于除文本分析以外的其他用途。
def pdfminer_loader_invoke():
    loader = PDFMinerLoader("example_data/2103.15348.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)


# pypdfium2是对PDFium的Python 3绑定，PDFium是由Foxit编写并由Google维护的自由许可的PDF渲染库
def pypdfium2_loader_invoke():
    loader = PyPDFium2Loader("example_data/2103.15348.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)


# PyMuPDF是MuPDF的一个Python绑定--"一个轻量级的PDF和XPS查看器"。使用PyMuPDF可以将一个PDF文件转换为多种图像格式。
def pymupdf_loader_invoke():
    loader = PyMuPDFLoader("example_data/2103.15348.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)


# 加载PDF目录
def pdy_dir_loader_invoke():
    loader = PyPDFDirectoryLoader("example_data/")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)


# 与 PyMuPDF 一样，输出文档包含有关 PDF 及其页面的详细元数据，并每页返回一个文档。
def pdfplumber_loader_invoke():
    loader = PDFPlumberLoader("example_data/from_one_to_infinity.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)


# 程序入口
if __name__ == "__main__":
    # pdf_loader_invoke()
    # test_with_embedding()
    # extracting_images_invoke()
    # unstructured_loader_invoke()
    # fetch_remote_pdf_loader_invoke()
    # pdfminer_loader_invoke()
    # pypdfium2_loader_invoke()
    # pymupdf_loader_invoke()
    # pdy_dir_loader_invoke()
    pdfplumber_loader_invoke()
    pass