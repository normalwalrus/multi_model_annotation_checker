# docker build -t asr-eval/whisper-hf:1.0.0 .
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get clean \
    && apt-get update \ 
    && apt-get install -y gcc g++ libsndfile1 ffmpeg sox wget git \
    && rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
RUN python3 -m pip install --upgrade --no-cache-dir pip wheel \ 
    && python3 -m pip install --no-cache-dir Cython==3.0.6

ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -c "from qwen_asr import Qwen3ASRModel; Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-1.7B')"

WORKDIR /asr-eval
ENTRYPOINT ["bash"]
