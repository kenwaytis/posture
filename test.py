import base64
import requests

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # 对图片进行Base64编码
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def send_request(images, threshold):
    # 构建POST数据
    data = {
        "images": images,
        "threshold": threshold
    }
    response = requests.post("http://localhost:10056/predict", json=data)
    return response.text

def main():
    # 读取两个图片文件并转换为Base64
    base64_1 = image_to_base64("test3.png")
    base64_2 = image_to_base64("test4.png")

    # 发送请求并打印结果
    result = send_request([base64_1, base64_2], 200)
    print(result)

if __name__ == "__main__":
    main()
