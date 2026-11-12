# Small image for the offline demo. Default embedder is the built-in hashing one:
# no model weights are downloaded at build time or at run time.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
CMD ["python", "-m", "hybrid_rag", "demo"]
