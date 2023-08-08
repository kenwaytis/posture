FROM paidax/dev-containers:modelscope-v0.8

WORKDIR /home/posture

COPY ./download_model.py download_model.py

RUN python download_model.py

COPY . . 
