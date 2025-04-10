FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/LinguifyTTS

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install python3-dev cmake python3-pip git
# install torch takes a long time, cache it in case we may change requirements.txt
# RUN git clone --depth 1 https://github.com/josephchay/linguify.git@
ADD LinguifyTTS.tar .
RUN mv LinguifyTTS_dockerfile LinguifyTTS
RUN cd LinguifyTTS && pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN cd LinguifyTTS/runtime/python && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. LinguifyTTS.proto
CMD ["/bin/bash", "-c", "cd /opt/LinguifyTTS/LinguifyTTS/runtime/python && . ./path/sh && python3 server.py --port 50000 --max_conc 4 --model_dir speech_tts/LinguifyTTS-300M && sleep infinity"]
