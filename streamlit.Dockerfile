FROM python:3.11-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install --upgrade pip && pip install -r requirements_streamlit.txt
COPY streamlit_app.py .
COPY model/ ./model/
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
