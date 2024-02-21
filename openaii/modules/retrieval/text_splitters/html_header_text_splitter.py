"""
HTML文本分割器
"""
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]


# 使用HTML标题分割
def html_splitter_invoke():
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    html_header_splits = html_splitter.split_text(html_string)
    for text in html_header_splits:
        print(text)


# 通过管道传输到另一个拆分器，并从 Web URL 加载 html
def pipelined_splitter_invoke():
    url = "https://plato.stanford.edu/entries/goedel/"
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # for local file use html_splitter.split_text_from_file(<path_to_file>)
    html_header_splits = html_splitter.split_text_from_url(url)

    chunk_size = 500
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(html_header_splits)
    for text in splits:
        print(text)


# 程序入口
if __name__ == "__main__":
    # html_splitter_invoke()
    pipelined_splitter_invoke()
    pass