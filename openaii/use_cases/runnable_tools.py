"""
调试工具
"""

from typing import Dict, Optional
from langchain_core.runnables.utils import Input
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.load import Serializable


class StdOutputRunnable(Serializable, Runnable[Input, Input]):
    @property
    def lc_serializable(self) -> bool:
        return True

    def invoke(self, input: Dict, config: Optional[RunnableConfig] = None) -> Input:
        print(input)
        return self._call_with_config(lambda x: x, input, config)