FROM paidax/dev-containers:modelscope-v0.8

WORKDIR /home/posture

COPY ./download_model.py download_model.py

RUN python download_model.py

RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic==1.10.8 \
        loguru \
        pai-easycv>=0.11.0 \
        wget \
        dataclasses \
        seaborn \
        json-tricks \
        jsonplus \
        cityscapesscripts \
        pytorch-metric-learning \
        xtcocotools \
        appdirs \
        typing
        
COPY . . 
