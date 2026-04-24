FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8100

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8100"]
