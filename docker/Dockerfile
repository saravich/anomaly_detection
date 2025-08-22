FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/

COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir -U pip setuptools wheel && pip install -e .[dev]

COPY src /app/src
COPY configs /app/configs
COPY docker /app/docker
COPY Makefile ./

ENTRYPOINT ["python", "-m", "adlib.cli.infer"]
