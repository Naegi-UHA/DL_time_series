FROM tensorflow/tensorflow:2.18.0
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
COPY . /app
EXPOSE 80
CMD ["python", "main.py"]
