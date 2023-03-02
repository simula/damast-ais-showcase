FROM python:3.10-slim
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

RUN apt-get update && \
    apt-get install -q -y python3-pip

RUN python3 -m pip install --upgrade pip setuptools

RUN python3 -m pip install vaex dash numpy

WORKDIR /root/script
COPY data/ais_20200101.hdf5 .
COPY data/mmsi2vesseltype.hdf5 .
COPY main.py .
COPY web_application.py .
EXPOSE 8888/tcp
ENTRYPOINT ["python3", "main.py", "--file=/root/script/ais_20200101.hdf5", "--mmsi-file=/root/script/mmsi2vesseltype.hdf5", "--port=8888"]
