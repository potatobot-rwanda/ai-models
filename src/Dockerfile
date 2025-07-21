# Example Dockerfile for a FastAPI app with Uvicorn
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "ner_api:app", "--host", "0.0.0.0", "--port", "80"]
