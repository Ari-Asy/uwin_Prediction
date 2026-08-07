FROM quay.io/jupyter/scipy-notebook:2024-10-28

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && fix-permissions "/home/${NB_USER}"

ENV PYTHONPATH=/home/jovyan/project/src