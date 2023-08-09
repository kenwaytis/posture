FROM registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0

WORKDIR /home/posture

COPY ./download_model.py download_model.py

RUN python download_model.py

RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic==1.10.8 \
        loguru
COPY . . 
