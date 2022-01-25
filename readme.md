# gas_val_to_csv_for_ERDDAP

This repository is used to convert the text files with .txt extension from ASVCO2 gas validation into a csv format for ERDDAP.

## Installation

Make sure that python is installed and added to the PATH. and use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

### Create a virtual environment and activate it

In Windows:
Create a local subfolder, then open a command prompt (cmd) and use the command below.
```cmd
C:\my_local_subfolder> python -m venv .\.venv
```
The above command will create a virtual environment in the subfolder \'.venv\' within my_local_subfolder.
Next, activate the virtual environment with the activate .bat script.
```cmd
C:\my_local_subfolder> .\.venv\Scripts\acivate
```

In Linux:
Create a local subfolder, then open a bash prompt in that subfolder.
```bash
my_name@my_PC:~/my_local_subfolder$ python3 -m venv ./venv
```
The above command will create a virtual environment in the subfolder \'.venv\' within my_local_subfolder.
Next, activate the virtual environment.
```bash
my_name@my_PC:~/my_local_subfolder$ source ./venv/bin/activate
```

### Install Python packages into the virtual environment

In Windows:
```cmd
(.venv)C:\my_local_subfolder> python -m pip install -r requirements.txt
```
In Linux:
```bash
(venv)my_name@my_PC:~/my_local_subfolder$ python3 -m pip install -r requirements.txt
```

## Usage

Edit the \"if \__name__ == '\__main__'\:\" portion of the code to point to the corresponding folders and files. Then run it.

## Contributing
This is intended for a very limited usage within the ASVCO2 project. The resulting output data is required to conform to expectations. For any changes, please open an issue first to discuss what you would like to change.

## License
TBD
[//]: <> ([MIT](https://choosealicense.com/licenses/mit/))
