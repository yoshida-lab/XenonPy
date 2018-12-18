FROM pytorch/pytorch

WORKDIR /etc
RUN echo "backend : Agg" >> matplotlibrc

RUN /opt/conda/bin/conda install -y jupyter scikit-learn pytest pytest-cov matplotlib seaborn pandas &&\
    /opt/conda/bin/conda install -y -c rdkit rdkit &&\
    /opt/conda/bin/conda clean -ya &&\
    pip install pymatgen ruamel.yaml jupyterlab pybtex openpyxl plotly

WORKDIR /opt/xenonpy
COPY . .
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf && \
    pip install -v . && pytest tests -v

EXPOSE 8888

WORKDIR /workspace
CMD [ "jupyter" , "lab", "--ip=0.0.0.0", "--no-browser", "--port=8888", "--allow-root"]