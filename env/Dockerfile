FROM python:3.9
WORKDIR /app
COPY env/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "inference.py"]
