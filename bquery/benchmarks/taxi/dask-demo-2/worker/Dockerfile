FROM python:3.8

RUN mkdir /opt/app
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

ENTRYPOINT ["dask-worker"]