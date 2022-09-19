FROM ubuntu:20.04

FROM python:3.8.0

RUN mkdir -p /home/WIL_model

WORKDIR /home/WIL_model

ADD . /home/WIL_model

RUN touch /home/WIL_model/.env

ADD ./.env /home/WIL_model/.env

RUN pip install -r requirements.txt

RUN pip install gunicorn

EXPOSE 8000

CMD ["gunicorn","--preload", "WIL_model.wsgi", "--bind","0.0.0.0:8000"]
