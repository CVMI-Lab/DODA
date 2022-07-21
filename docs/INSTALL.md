# Installation

### Requirements
All the codes are tested in the following environment:
- Python 3.6+
- PyTorch 1.5
- CUDA 10.2
- [spconv v1.2](https://github.com/traveller59/spconv)


### Install dependent libraries


a. Clone this repository.
```shell
git clone https://github.com/CVMI-Lab/DODA.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
    ```
    pip install -r requirements.txt 
    ```

* Install [`spconv v1.2`](https://github.com/traveller59/spconv) following official guidance.

* Install [`pointgroup_ops`](../lib/pointgroup_ops):
    ```
    conda install -c bioconda google-sparsehash
    cd lib/pointgroup_ops
    python3 setup.py develop
    export LD_LIBRARY_PATH=/your_path_to_python_env/:LD_LIBRARY_PATH
    ```
    if you encounter any problems in th installing process, please refer to [PointGroup](https://github.com/dvlab-research/PointGroup) for assistance.
* Install [`pointops`](../lib/pointops2):
    ```
    cd lib/pointops2
    python3 setup.py develop
    ```