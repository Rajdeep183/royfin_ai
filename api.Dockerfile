FROM python:3.11-slim
WORKDIR /app
COPY model/ ./model/
COPY cloud/functions/ ./cloud/functions/
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt fastapi uvicorn
COPY api_main.py .
EXPOSE 8000
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
