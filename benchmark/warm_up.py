import argparse

import requests


def run(ip: str, port: str, model: str):
    url = f"http://{ip}:{port}/v1/chat/completions"
    data = {"model": model, "messages": [{"role": "user", "content": "Hello"}]}

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print("LLM engine has been successfully warmed up")
    else:
        print(f"Failed to warm up LLM engine. Error code is {response.status_code}")


args = argparse.ArgumentParser()

args.add_argument("--ip", type=str, required=True)

args.add_argument("--port", type=str, required=True)

args.add_argument("--model", type=str, required=True)

if __name__ == "__main__":

    args = args.parse_args()

    ip = args.ip
    port = args.port
    model = args.model

    run(ip, port, model)
