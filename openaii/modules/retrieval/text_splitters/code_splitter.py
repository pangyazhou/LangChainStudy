"""
CodeTextSplitter
允许您使用支持的多种语言拆分代码
"""
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)


def code_splitter_invoke():
    # ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 'python', 'rst', 'ruby', 'rust', 'scala',
    # 'swift', 'markdown', 'latex', 'html', 'sol', 'csharp', 'cobol']
    # ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
    # Full list of supported languages
    print([e.value for e in Language])
    # You can also see the separators used for a given language
    print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))


def python_code_splitter_invoke():
    PYTHON_CODE = """
    def hello_world():
        print("Hello, World!")

    # Call the function
    hello_world()
    """
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=50, chunk_overlap=0
    )
    python_docs = python_splitter.create_documents([PYTHON_CODE])
    for doc in python_docs:
        print(doc)


def js_code_splitter_invoke():
    JS_CODE = """
    function helloWorld() {
      console.log("Hello, World!");
    }

    // Call the function
    helloWorld();
    """

    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=60, chunk_overlap=0
    )
    js_docs = js_splitter.create_documents([JS_CODE])
    for doc in js_docs:
        print(doc)


def ts_code_splitter_invoke():
    TS_CODE = """
    function helloWorld(): void {
      console.log("Hello, World!");
    }

    // Call the function
    helloWorld();
    """

    ts_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.TS, chunk_size=60, chunk_overlap=0
    )
    ts_docs = ts_splitter.create_documents([TS_CODE])
    for doc in ts_docs:
        print(doc)


def md_code_splitter_invoke():
    markdown_text = """
    # 🦜️🔗 LangChain

    ⚡ Building applications with LLMs through composability ⚡

    ## Quick Install

    ```bash
    # Hopefully this code block isn't split
    pip install langchain
    ```

    As an open-source project in a rapidly developing field, we are extremely open to contributions.
    """
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
    )
    md_docs = md_splitter.create_documents([markdown_text])
    for doc in md_docs:
        print(doc)


def latex_code_splitter_invoke():
    latex_text = """
    \documentclass{article}

    \\begin{document}

    \maketitle

    \section{Introduction}
    Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.

    \subsection{History of LLMs}
    The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

    \subsection{Applications of LLMs}
    LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

    \end{document}
    """
    latex_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
    )
    latex_docs = latex_splitter.create_documents([latex_text])
    for doc in latex_docs:
        print(doc)


def html_code_splitter_invoke():
    html_text = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>🦜️🔗 LangChain</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                h1 {
                    color: darkblue;
                }
            </style>
        </head>
        <body>
            <div>
                <h1>🦜️🔗 LangChain</h1>
                <p>⚡ Building applications with LLMs through composability ⚡</p>
            </div>
            <div>
                As an open-source project in a rapidly developing field, we are extremely open to contributions.
            </div>
        </body>
    </html>
    """
    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=60, chunk_overlap=0
    )
    html_docs = html_splitter.create_documents([html_text])
    for doc in html_docs:
        print(doc)


def c_code_splitter_invoke():
    C_CODE = """
    using System;
    class Program
    {
        static void Main()
        {
            int age = 30; // Change the age value as needed

            // Categorize the age without any console output
            if (age < 18)
            {
                // Age is under 18
            }
            else if (age >= 18 && age < 65)
            {
                // Age is an adult
            }
            else
            {
                // Age is a senior citizen
            }
        }
    }
    """
    c_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CSHARP, chunk_size=128, chunk_overlap=0
    )
    c_docs = c_splitter.create_documents([C_CODE])
    for doc in c_docs:
        print(doc)


# 程序入口
if __name__ == "__main__":
    # code_splitter_invoke()
    # python_code_splitter_invoke()
    # js_code_splitter_invoke()
    # ts_code_splitter_invoke()
    # md_code_splitter_invoke()
    # latex_code_splitter_invoke()
    # html_code_splitter_invoke()
    c_code_splitter_invoke()
    pass


