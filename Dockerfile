FROM python:3.11-slim
RUN apt-get update -y
RUN apt-get -y install ntp ssh vim net-tools python3 python3-pip python3-dev wget tzdata git sudo
RUN apt-get -y install tcpdump tcpflow pylint iputils-ping curl unzip telnet redis lsb-release snapd

RUN echo "Europe/Berlin" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

RUN pip install --upgrade pip
RUN pip install pyyaml qdrant-client requests --break-system-packages

# added when PDF import feature added to the codebase
RUN pip install pymupdf sentence-transformers pdf2image pytesseract langchain --break-system-packages
RUN apt-get -y install poppler-utils tesseract-ocr

# This is necessary besides the main timezone command
RUN ln -sf /usr/share/zoneinfo/Europe/Berlin /etc/localtime

# To keep container alive
CMD ["tail", "-f", "/dev/null"]
