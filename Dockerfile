FROM continuumio/miniconda:4.7.12

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /yass

ENV PATH /opt/conda/bin:$PATH

COPY . /yass

RUN make install

