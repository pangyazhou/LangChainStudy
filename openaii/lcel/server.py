# -*- coding: utf-8 -*-
import random
import asyncio
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from openaii.lcel.stream import stream_lcel, CustomSse, StreamCallBackHander

app = FastAPI()
# 跨域设置，因为测试需要前端访问，所以允许所有域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

callback = StreamCallBackHander()
customSse = CustomSse(callback)


def generate_events():
    while True:
        yield f'data: {time.ctime()}\n\n'
        time.sleep(1)


@app.get("/")
def root():
    print("welcome to ")
    return "welcome to openaii"


@app.get('/query')
def query(inputValue: str):
    print("inputValue: ", inputValue)
    stream_lcel(inputValue, callback)
    return "query"


@app.get('/events')
async def stream(request: Request):
    def new_count():
        return random.randint(1, 100)

    async def event_generator():
        index = 0
        while True:
            index += 1
            if await request.is_disconnected():
                break
            # 测试取随机数据，每次取一个随机数
            if count := new_count():
                print(count)
                yield {'data': count}

            await asyncio.sleep(1)

    return EventSourceResponse(customSse.event_source())


if __name__ == '__main__':
    uvicorn.run('server:app', reload=True)
