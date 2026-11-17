# Sandbox for generated strategy code.   docker build -t augurix-sandbox .
# Library versions are pinned to what the model saw in training (cutoff January 2025):
# generated code written for pandas 2 / numpy 1 breaks on pandas 3 / numpy 2.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.3 pyarrow==17.0.0
ENV PYTHONDONTWRITEBYTECODE=1
USER nobody
