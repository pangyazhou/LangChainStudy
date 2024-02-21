"""
涵盖了如何加载目录中的所有文档
使用glob参数来控制加载哪些文件
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PythonLoader


def file_directory_loader_invoke():
    # 指定加载目录
    # glob="**/*.csv" 加载CSV格式的文件
    # show_progress=True 显示加载进度
    # use_multithreading=True 多线程加载
    # loader_cls=TextLoader 指定文件加载器 （默认使用UnstructuredLoader）
    loader = DirectoryLoader('C:\\tmp\\loader', glob="**/*.py",
                             show_progress=True, use_multithreading=True, loader_cls=PythonLoader)
    docs = loader.load()
    print(len(docs))
    for doc in docs:
        print(doc)


def auto_detect_file_encoding_invoke():
    path = "C:\\tmp\\loader"
    # silent_errors=True  跳过错误编码文件
    # 自动检测文档编码， 排除错误编码
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader,
                             silent_errors=True, loader_kwargs={"autodetect_encoding": True})
    docs = loader.load()
    doc_sources = [doc.metadata['source'] for doc in docs]
    print(doc_sources)


# 程序入口
if __name__ == "__main__":
    # file_directory_loader_invoke()
    auto_detect_file_encoding_invoke()
    pass