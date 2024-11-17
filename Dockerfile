FROM registry.gitee-ai.local/base/iluvatar-corex:3.2.0-sd3-bi100

RUN apt-get update && apt-get install -y net-tools ffmpeg libnss3 libpcre3 libpcre3-dev build-essential software-properties-common python3.9 python3-pip git git-lfs curl wget

WORKDIR /root/smart-sales

COPY . .

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "00🏡智能营销.py", "--server.port=7860"]