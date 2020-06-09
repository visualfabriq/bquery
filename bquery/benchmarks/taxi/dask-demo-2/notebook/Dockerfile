FROM python:3.8

RUN mkdir /opt/app
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt install -y nodejs
RUN apt install -y graphviz
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install dask-labextension
RUN jupyter serverextension enable dask_labextension
COPY examples /opt/app/examples
RUN chmod -R 755 /opt/app
WORKDIR /opt/app
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]