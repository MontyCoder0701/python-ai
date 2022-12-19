# Fly AI Practice Codes
Practice codes for Fly AI

## Setup with virtual environment
Download Conda <https://www.anaconda.com/products/distribution/start-coding-immediately>

```sh
conda create --name flyai python=3.8   
```
```sh
conda activate flyai   
```

## Jupyter lab setup
```sh
pip install ipykernel
```
```sh
python -m ipykernel install --user --name flyai  --display-name flyai          
```
```sh
jupyter lab
```
The website should be running on <http://localhost:8888/lab?token=ed9ba11b30d318147f02a1fe1ed76a844776c1b46fba988a>