"""
分块通常旨在将具有共同上下文的文本放在一起
Markdown 文件是按标题组织的。在特定标头组中创建块是一个直观的想法
使用 MarkdownHeaderTextSplitter.这将Markdown文件按一组指定的标头拆分
"""
from langchain.text_splitter import MarkdownHeaderTextSplitter


def md_header_splitter_invoke():
    markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    for doc in md_header_splits:
        print(doc)


# 组合使用分割器
def combine_splitter_invoke():
    markdown_document = "# Intro \n\n    ## History \n\n Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] \n\n Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. \n\n ## Rise and divergence \n\n As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for \n\n additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. \n\n #### Standardization \n\n From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. \n\n ## Implementations \n\n Implementations of Markdown are available for over a dozen programming languages."

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # Char-level splits
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunk_size = 250
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    for doc in splits:
        print(doc)


# 程序入口
if __name__ == "__main__":
    # md_header_splitter_invoke()
    combine_splitter_invoke()
    pass