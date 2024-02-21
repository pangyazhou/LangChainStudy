import logging
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from spark.LLMService import SparkApi

logger = logging.getLogger(__name__)


class SparkLLM(LLM):
    """Define the custom LLM wrapper for Xunfei SparkLLM to get support of LangChain
    """
    # 基于langchain的LLM基类定制讯飞星火大模型类提供对星火大模型的调用
    """Endpoint URL to use.此URL指向部署的调用星火大模型的FastAPI接口地址"""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""
    # max_token: int = 4000
    """Max token allowed to pass to the model.在真实应用中考虑启用"""
    # temperature: float = 0.75
    """LLM model temperature from 0 to 10.在真实应用中考虑启用"""
    # history: List[List] = []
    """History of the conversation.在真实应用中可以考虑是否启用"""
    # top_p: float = 0.85
    """Top P for nucleus sampling from 0 to 1.在真实应用中考虑启用"""
    # with_history: bool = False
    """Whether to use history or not.在真实应用中考虑启用"""
    # 以下密钥信息从控制台获取
    appid = "fba38362"  # 填写控制台中获取的 APPID 信息
    api_secret = "xxx"  # 填写控制台中获取的 APISecret 信息
    api_key = "xxx"  # 填写控制台中获取的 APIKey 信息

    # 用于配置大模型版本，默认“general/generalv2”
    # domain = "general"   # v1.5版本
    # domain = "generalv2"    # v2.0版本
    domain = "generalv3"  # v3.0版本
    # 云端环境的服务地址
    # Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
    # Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
    spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"  # v3.0环境的地址


    @property
    def _llm_type(self) -> str:
        return "SparkLLM"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # call api
        logger.debug(prompt)
        invocation_llm = InvocationLLM(self.appid, self.api_key, self.api_secret, self.spark_url, self.domain)
        response = invocation_llm.call_llm(prompt)
        logger.debug(f"SparkLLM response: {response}")
        return response


class InvocationLLM:
    def __init__(self, appid, api_key, api_secret, spark_url, domain):
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.spark_url = spark_url
        self.domain = domain
        pass

    def invoke(self):
        pass

    def getText(self, role, content):
        text = []
        json = {"role": role, "content": content}
        text.append(json)
        return text

    def getlength(self, text):
        length = 0
        for content in text:
            temp = content["content"]
            leng = len(temp)
            length += leng
        return length

    def checklen(self, text):
        while self.getlength(text) > 8000:
            del text[0]
        return text

    def call_llm(self, query: str):
        question = self.checklen(self.getText("user", query))
        SparkApi.answer = ""
        SparkApi.main(self.appid, self.api_key, self.api_secret, self.spark_url, self.domain, question)
        return SparkApi.answer
