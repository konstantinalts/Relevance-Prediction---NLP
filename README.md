# Relevance Prediction - NLP Assignment
Authors:
- Christou Christou
- Latsiou Konstantina

## Setup
We propose creating a virtual machine and installing the dependencies libraries in it.

#### Create a Virtual environment
Create a virtual environment 'venv'

`python -m venv venv`

Activate on Unix (Mac/Linux)

`. venv/bin/activate`

Activate on Windows

`venv\Scripts\activate`


#### Install required libraries
Install with pip

`pip install -r requirements.txt`

Install with Anacoda

`conda install --file requirements.txt`

## Running the code

1. Create a subfolder called 'data', in the same directory as the python files.
2. Place the 3 source files inside the 'data' subfolder.
3. Run the `main.py` file. **ATTENTION!** The file will sequentially preprocess the data, create the different text representations and run the different learning models for all the representations. This will require over an hour to run (run time is based on 2GHz Intel Core i5, with 8GB 1867 MHz LPDDR3 machine.)
4. Due to pickling, about 5GB of extra disk space is required.
