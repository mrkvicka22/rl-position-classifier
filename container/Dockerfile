FROM python:3.9

RUN mkdir -p /workspace/
WORKDIR /workspace/
ADD requirements.txt /workspace/
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN git clone https://github.com/nevercast/rl-position-classifier.git --depth 1
ADD .netrc /root/.netrc
ADD startup.sh /workspace/startup.sh
ADD launch_agent.py /workspace/launch_agent.py

ENTRYPOINT ["/workspace/startup.sh"]