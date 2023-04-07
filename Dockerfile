FROM pytorch/pytorch

COPY ./FastFlow /FastFlow
WORKDIR /FastFlow

RUN pip install -r ./requirements.txt
RUN apt update
RUN apt upgrade -y
RUN apt install -y cmake build-essential

CMD ["/bin/bash"]