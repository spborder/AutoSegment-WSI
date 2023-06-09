FROM python:3

LABEL maintainer="Sam Border CMI Lab <samuel.border@medicine.ufl.edu"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    openslide-tools \
    python3-openslide

ENV RUNTYPE="LOCAL"

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git

WORKDIR /

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8050

COPY . .

