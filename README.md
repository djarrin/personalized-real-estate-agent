# Personalized Real Estate Agent



## Local Dev Installation Guide

This project uses **Conda** for environment management and **Makefile** commands for easy setup. Follow the steps below to get started.

### **1. Install Conda (if not already installed)**
If you don’t have Conda installed, download and install **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** or **[Anaconda](https://www.anaconda.com/products/distribution)**.

Check if Conda is installed:
```bash
conda --version
```

### **2. Create the Conda Environment**
Run the following command to create a new Conda environment with Python:
```bash
make create_env
```
This will create a Conda environment named **`my_env`** (or another name if modified in the Makefile) with Python installed.

### **3. Activate the Environment**
Once created, activate the Conda environment:
```bash
conda activate my_env
```

### **4. Install Rust (If not already installed)**
Some packages, like `tiktoken`, require Rust for installation. Run:
```bash
make install_rust
```
This will check if Rust is installed and install it if necessary.

### **5. Install Required Python Packages**
Now, install all dependencies listed in `requirements.txt`:
```bash
make install_reqs
```
This will:
- Ensure **`pip`**, `setuptools`, and `wheel` are up-to-date.
- Install all dependencies from `requirements.txt`.


