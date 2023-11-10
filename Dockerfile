# FROM "python:3.10.13-alpine3.18"
FROM "ubuntu:22.04"
RUN apt update && apt install -y curl gcc gfortran libblas-dev liblapack-dev libc-dev g++ python3.10 python3.10-dev python3-pip python3-venv python3-virtualenv libffi-dev libssl-dev cmake make
# RUN apk update && apk add curl gcc gfortran openblas openblas-dev libc-dev g++ python3-dev libffi-dev openssl-dev cmake make
RUN curl -sSL https://install.python-poetry.org | python3 -
COPY . ./llm-feature-extractor
COPY ./pyvenv.cfg /usr
WORKDIR ./llm-feature-extractor
ENV PATH="${PATH}:/root/.local/bin"
RUN pip3 install virtualenv && poetry config virtualenvs.create false && \
    poetry update && poetry install --no-interaction --with dev && \
    poe install-sklearn && \
    poe install-xgboost && \
    poe install-torch && \
    poe install-transformers && \
    poe install-captum
CMD ["/bin/sh"]
