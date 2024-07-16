# Python Virtual Environment Setup Guide


## Prerequisites
- Python installed on your system.
  - [Download Python](https://www.python.org/downloads/)

## Steps

### Windows

1. **Open Command Prompt**
   - Press `Win + R`, type `cmd`, and press `Enter`.

2. **Navigate to your project directory**
   ```sh
   cd path\to\your\project
   ```

3. **Create a virtual environment**
   ```sh
   python -m venv venv
   ```

4. **Activate the virtual environment**
   ```sh
   venv\Scripts\activate
   ```

   You should see `(venv)` at the beginning of the command line.

5. **Install packages from `requirements.txt`**
   ```sh
   pip install -r requirements.txt
   ```

6. **Deactivate the virtual environment**
   ```sh
   deactivate
   ```

### macOS

1. **Open Terminal**
   - Press `Command + Space`, type `Terminal`, and press `Enter`.

2. **Navigate to your project directory**
   ```sh
   cd path/to/your/project
   ```

3. **Create a virtual environment**
   ```sh
   python3 -m venv venv
   ```

4. **Activate the virtual environment**
   ```sh
   source venv/bin/activate
   ```

   You should see `(venv)` at the beginning of the command line.

5. **Install packages from `requirements.txt`**
   ```sh
   pip install -r requirements.txt
   ```

6. **Deactivate the virtual environment**
   ```sh
   deactivate
   ```