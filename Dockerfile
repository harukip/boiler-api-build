FROM tensorflow/tensorflow:2.3.0-gpu
MAINTAINER yuhao

RUN apt update
RUN apt install -y python3-lxml
ADD ./boilerplate-api /api
WORKDIR /api
RUN pip install -r requirement.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]