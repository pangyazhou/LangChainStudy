"""
CSV格式文件加载器
"""

from langchain_community.document_loaders.csv_loader import CSVLoader


def csv_loader_invoke():
    loader = CSVLoader(file_path='example_data/library_borrow_records.csv', source_column="Title")
    data = loader.load()
    for doc in data:
        print(doc)

# 程序入口
if __name__ == "__main__":
    csv_loader_invoke()
    pass