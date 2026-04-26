FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

COPY . .

RUN pip install --no-cache-dir . && chmod +x start.sh

EXPOSE 8100

CMD ["./start.sh"]
