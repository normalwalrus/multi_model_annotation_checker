# docker build -t asr-eval/fastconformer:1.0.0 .
# development code are mounted onto the container, so as to facilitate code changes without rebuilding docker image

ARG PYTORCH_VERSION=2.0.1
ARG CUDA_VERSION=11.7
ARG CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONFAULTHANDLER 1
ENV TZ=Asia/Singapore
ENV TRANSFORMERS_CACHE="/nemo/models/transformers_cache"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install --no-install-recommends -y gcc g++ libsndfile1 ffmpeg wget git && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && \
    rm -rf /var/cache/apt/archives/

ARG NEMO_VERSION=2.2.0

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir Cython==0.29.35 && \
    pip3 install --no-cache-dir nemo_toolkit[asr]==${NEMO_VERSION} && \
    pip3 install --no-cache-dir huggingface_hub==0.24.0

WORKDIR /workspace

ADD requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# # install kenlm decoding binaries
# RUN git clone "https://github.com/NVIDIA/NeMo.git"
# RUN mv /workspace/NeMo /workspace/nemo
# RUN cd /workspace/nemo/scripts/asr_language_modeling/ngram_lm && ./install_beamsearch_decoders.sh

# Cache the model for faster evaluation
RUN python3 -c 'import nemo.collections.asr as nemo_asr ; model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")' 
# ; model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")'

WORKDIR /asr-eval
ENTRYPOINT ["bash"]
