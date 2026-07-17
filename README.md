<div align="center">
    
<h1>Forest Gain Foundations: Vision foundation model for distinguishing forest growth typology and restoration site informatics</h1>

</div align="center">

# Table of Contents
- [Preparation](#preparation)

## Preparation
    
- ### **Environment Installation**:
    <details open>
    #### **Step 1**: Download or clone the repository.
    ```python
    git clone https://github.com/JamesBrockUoB/ForestGrowthFoundations.git
    cd ./ForestChat/DataCollection
    ```
    
    #### **Step 2**: Create a virtual environment named `forest_growth_env` and activate it.
    Requires [pyenv](https://github.com/pyenv/pyenv) with the
    [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugin installed.

    1. Install Python 3.12 (if not already installed):

    ```bash
      pyenv install 3.12.12
    ```

    2. Create and activate the project virtualenv:

    ```bash
      pyenv virtualenv 3.12.12 forest-gain-venv
      pyenv activate forest-gain-venv
    ```

      (Or, to activate automatically whenever you're in this directory,
      run `pyenv local forest-gain-venv` once instead of step 2's
      `pyenv activate` — pyenv will then activate it for you on `cd`.)

    3. Install dependencies:

    ```bash
      cd DataCollection/
      pip install -r requirements.txt
    ```

    4. Deactivate when done:

    ```bash
      pyenv deactivate
    ```

    To confirm the environment is active: `python --version` should report
    `3.12.12`, and `pyenv version` should show `forest-gain-venv`.

    #### **Step 3**: Authorise GEE.

    - **EE User Login (`ee.Authenticate()` (one-time) + `ee.Initialize()`)
    - Both local and HPC run the same authentication logic — only credential *locations* differ

    #### One-time Earth Engine OAuth

    Run locally (or any interactive machine):

    ```
    python
    import ee

    ee.Authenticate()
    ee.Initialize()
    ```

    This will open a browser login flow, and store credentials locally at:
    ```~/.config/earthengine/credentials```

    If using a HPC system - upload this file in an easy-to-access area for usage later.

    ### **Step 4**: Setup .env file.
    Create a file in the project root folder called `.env` with the following variables:

      - GEE_PROJECT - Your GEE project name
      - OUTPUT_DIR - data/
      - BATCH_SIZE - 5
      - AOI_STEP - 0.25
      - TILE_PIXELS - 256
      - NUM_WORKERS - 4
      - TILE_SCALE - 10
      - DRIVE_FOLDER - Your output folder for data to be collected in GDrive
      - DRIVE_REMOTE - gdrive
      - HPC_REMOTE - Remote cluster connection and destination for file porting - the path to the repo's data folder. ideally, set an alias for HPC connection
      - GOOGLE_APPLICATION_CREDENTIALS - Your service account credentials path + file name
      - DRIVE_AUTH_CREDENTIALS - Your Earth Engine credentials for GDrive authorisations
      - SEARCH_MODE - `asset` for limiting search area to geographic extent of deadtrees.earth product, or `global` for the whole world
      - PERIOD - The time period for collecting and processing imagery for. `p1` for 2017-2020, or `p2` for 2020-2024. Note that there are no pseudo-labels for forest typology for `p2`
      - AEE_SOURCE - Data source for downloading AEE embeddings - either `gee` or `geoai`. GEE is faster but uses compute quota, whereas the GeoAI library is quota-free but slower

    </details>