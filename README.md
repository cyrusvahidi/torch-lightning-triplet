# torch-lightning-triplet
* Triplet Loss implemented with PyTorch Lightning 

## Installation
`git clone`

#### Create a virtual environment
`python3 -m venv <venv_name>`
`source venv/bin/activate`

#### Install the dependencies
`cd torch-lightning-triplet`
`pip install -r requirements.txt`

## Train the model
`python -m src.scripts.train_triplet` 

see command-line arguments in `src.utils.parse_args`

* Embeddings visualisations on the test set before and after training are output to `/plot`


## TODO
[] implement hard triplet loss
