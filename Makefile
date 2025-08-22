dev:
\tpip install -U pip && pip install -e .[dev]
\tpre-commit install

demo-data:
\tpython -m adlib.data --make-synth data/synth

docker-build:
\tdocker build -t anomaly-detection -f docker/Dockerfile .

docker-demo:
\tdocker run --rm -v $(PWD):/app anomaly-detection --help
