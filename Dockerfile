FROM python:3.10

WORKDIR /app

# Copy requirements dan install
COPY  . .

RUN pip install --no-cache-dir -r requirements.txt 

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]