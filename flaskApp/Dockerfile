FROM python:3.7.2-stretch
WORKDIR /app
ADD . /app
RUN pip install uwsgi --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
CMD ["uwsgi", "app.ini"]