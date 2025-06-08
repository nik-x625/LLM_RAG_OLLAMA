FROM python:3.11-slim
RUN apt-get update -y
RUN apt-get -y install ntp ssh vim net-tools python3 python3-pip python3-dev wget tzdata git sudo
RUN apt-get -y install tcpdump tcpflow pylint iputils-ping curl unzip telnet redis lsb-release snapd
#RUN apt-get -y nodejs npm
#RUN apt-get -y install mosquitto
#RUN apt-get -y install build-essential libssl-dev libffi-dev python3-setuptools python3-venv
#RUN apt-get -y install nginx pipx

#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

#RUN pip3 install flask flask_login redis rq pymongo flake8 pyyaml markdown sqlalchemy --break-system-packages
#RUN pip3 install flask_principal slugify feedwerk shortuuid psutil clickhouse_connect requests numpy --break-system-packages
#RUN useradd flask

RUN echo "Europe/Berlin" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

RUN pip install --upgrade pip
RUN pip install jinja2==3.0 pyyaml qdrant-client requests --break-system-packages

# added when PDF import feature added to the codebase
RUN pip install pymupdf sentence-transformers pdf2image pytesseract langchain --break-system-packages
RUN apt-get -y install poppler-utils tesseract-ocr

# This is necessary besides the main timezone command
RUN ln -sf /usr/share/zoneinfo/Europe/Berlin /etc/localtime

# To keep container alive
CMD ["tail", "-f", "/dev/null"]

# How to use?
# Put the content in Dockerfile
# run 'docker build -t test_image ./'  
# run 'docker build -t test_image ./ --platform linux/amd64'       if you want to allocate the architecture
# run 'docker run -d -p8080:8080 -p8081:8081 -p8585:8585 -v "$(pwd)":/opt/ test_image'