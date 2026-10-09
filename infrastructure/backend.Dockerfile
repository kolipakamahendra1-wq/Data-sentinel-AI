FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend backend
COPY agents agents
COPY data data
COPY drift drift
COPY ml ml
COPY monitoring monitoring
COPY profiling profiling
ENV DATASENTINEL_STORAGE=/storage
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
