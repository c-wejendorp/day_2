# base image
FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY day_2/ day_2/
COPY data/ data/
COPY models/ models/
COPY Makefile Makefile

#set working directory and install dependencies
WORKDIR /

# run make requirements to install dependencies with no cache
#RUN make requirements

# install dependencies with no cache
# the pip install . looks for the setup.py file in the current directory and installs the package
RUN pip install . --no-cache-dir


# entrypoint is what is run when the container is run. 
# Here the flag -u here makes sure that any output from our script e.g. any print(...) statements gets redirected to our terminal. 
# ENTRYPOINT ["python", "-u", "day_2/train_model.py","train"]
ENTRYPOINT ["python", "-u", "day_2/train_model.py"]
CMD ["train"]

