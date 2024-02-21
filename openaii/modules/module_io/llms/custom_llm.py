"""
介绍了如何创建自定义 LLM 包装器，以防您想使用自己的 LLM 或与 LangChain 支持的包装器不同的包装器。
需要实现两个接口：
1. _call 接受字符串、一些可选停用词并返回字符串的方法。
2. _llm_type 返回字符串的属性。仅用于记录目的。
3._identifying_params 用于帮助打印此类的属性。（可选）
"""

# 实现一个非常简单的自定义LLM， 它只返回输入的前n个字符
from typing import Optional, List, Any, Mapping

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
#from langchain.llms.base import LLM


class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ):
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}


def custom_llm_invoke():
    llm = CustomLLM(n=10)
    result = llm.invoke("This is a foobar thing")
    print(result)
    print(llm)



# 程序入口
if __name__ == "__main__":
    custom_llm_invoke()
    pass