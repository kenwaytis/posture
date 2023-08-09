import cv2
import json
import base64
import requests
from typing import List
import numpy as np
from pydantic import BaseModel
from loguru import logger
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import RedirectResponse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


model_id = 'damo/cv_yolox-pai_hand-detection'
hand_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

app = FastAPI()

class Image(BaseModel):
    images: List[str]
    threshold: int = 200

@app.on_event("startup")
async def startup_event():
    with open("test1.png", "rb") as image_file:
        # 对图片进行Base64编码
        encoded_string_1 = base64.b64encode(image_file.read()).decode('utf-8')
    with open("test2.png", "rb") as image_file:
        # 对图片进行Base64编码
        encoded_string_2 = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
        "images": [encoded_string_1,encoded_string_2],
        "threshold": 200
    }
    response = requests.post("http://localhost:10056/predict", json=data).text
    logger.info(f"Startup Event Response: {response}")

@app.post("/predict")
async def prediction(items: Image):
    try:
        flag = 0
        results = []
        for img_b64 in items.images:
            if flag == 2:
                break
            img_binary = base64.b64decode(img_b64)
            buffer = np.frombuffer(img_binary,dtype=np.uint8)
            img = cv2.imdecode(buffer,flags=cv2.IMREAD_COLOR)
            result = hand_detection(img)
            logger.debug(result)
            if len(result['boxes']) == 1:
                results.append(
                    [(
                    int((result['boxes'][0][0]+result['boxes'][0][2]))/2,
                    int((result['boxes'][0][1]+result['boxes'][0][3]))/2
                    ),
                    (
                    0.0,
                    0.0
                    )]
                )
            else:
                results.append(
                    [(
                    int((result['boxes'][0][0]+result['boxes'][0][2]))/2,
                    int((result['boxes'][0][1]+result['boxes'][0][3]))/2
                    ),
                    (
                    int((result['boxes'][1][0]+result['boxes'][1][2]))/2,
                    int((result['boxes'][1][1]+result['boxes'][1][3]))/2
                    )]
                )
            flag += 1

        left_sqr = int((pow((results[0][0][0]-results[1][0][0]), 2) + pow(results[0][0][1]-results[1][0][1], 2)) ** 0.5)
        right_sqr = int((pow((results[0][1][0]-results[1][1][0]), 2) + pow(results[0][1][1]-results[1][1][1], 2)) ** 0.5)
        threshold = items.threshold
        logger.info(
            f"left_sqr: {left_sqr}\n"
            f"right_sqr: {right_sqr}\n"
            f"threshold: {threshold}\n"
        )
        if left_sqr > threshold or right_sqr > threshold:
            return {"is_move": 1}
        return {"is_move": 0}
    except Exception as e:
        logger.info(e)
        return{"is_move": 0}

@app.get("/health")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.get("/health/inference")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK
    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)