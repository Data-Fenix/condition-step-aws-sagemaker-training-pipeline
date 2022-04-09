FROM ubuntu:20.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update OS
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 

RUN apt-get update && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && \
    apt-get -y install gcc g++ make 

ARG PIP=pip3

WORKDIR /
           
#RUN pip3 install pandas
#RUN pip3 install sklearn
#RUN pip3 install boto3
#RUN pip3 install xgboost==0.80
##RUN pip3 install pickle
#RUN pip3 install pyarrow


ARG PIP=pip3

RUN ${PIP} install --no-cache --upgrade \
            xgboost \
            pandas \
            boto3 \
            sklearn \
            pyarrow \
            matplotlib \
            pytz \
            seaborn \
            shap \
            numpy==1.19.5 \
            numba==0.54 \
            fastparquet
    
# Add a Python script and configure Docker to run it
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "evaluation.py"]
