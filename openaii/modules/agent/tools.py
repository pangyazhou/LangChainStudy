import requests
from langchain.agents import load_tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import get_all_tool_names


def ddg_search_invoke():
    search = DuckDuckGoSearchRun()
    search.run("Obama's first name?")


def tools_load():
    tools = load_tools(['searchapi'])
    search = tools[0]
    search.run("2018年世界杯冠军是谁？")


def show_tools():
    print(get_all_tool_names())

def https_request():
    response = requests.get('https://duckduckgo.com')
    print(response)


if __name__ == "__main__":
    # ddg_search_invoke()
    # https_request()
    tools_load()
    # show_tools()

