FROM jupyter/scipy-notebook

WORKDIR '/app/breast_cancer_wisconsin'
COPY ./Notebooks ./notebooks
COPY ./datasets ./datasets
COPY ./utils ./utils

