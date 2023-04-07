FROM pytorch/pytorch

COPY ./FastFlow /FastFlow
WORKDIR /FastFlow

RUN pip install -r ./requirements.txt
RUN apt update
RUN apt upgrade -y
RUN apt install -y cmake build-essential libgl1-mesa-dev libglib2.0-0

CMD ["/bin/bash"]