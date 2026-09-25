<div align="center">
    
<h1>Forest Gain Foundations: Vision foundation model for identifying forest gain areas and tile-level gain typology</h1>

</div align="center">

# Table of Contents
- [Preparation](#preparation)

## Preparation
    
- ### **Environment Installation**:
    <details open>
    #### **Step 1**: Download or clone the repository.
    ```python
    git clone https://github.com/JamesBrockUoB/ForestGainFoundations.git
    cd ./ForestGainFoundations
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
      sh install_deps.sh
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

    #### **Step 4**: Authorise rclone.

    Ideally, locally authorise rclone and then copy the rclone.conf file onto HPC to `~/.config/rclone/rclone.conf`

    Steps to authorise rclone:
      - `rclone config`
      - `n` - for a new remote
      - Name: `gdrive` (must match settings.drive_remote which will be derived from the .env vars defined in the next step)
      - Client ID / secret - can skip or provide personal GCP OAuth client set up (see here for creating service account and client IDs for OAuth 2: https://developers.google.com/workspace/guides/create-credentials#oauth-client-id)
      - Scope: `1` (full access)
      - Service Account Credentials - can use credentials here instead of a client ID.
      - Choose `No` for autoconfig if on HPC due to limited browser based auth - this will provide a URL which can be followed on local machine
        from which the verification code can be pasted back into the HPC prompt.
      - For your GCP project associated with the project, ensure the Google Drive API is enabled if not using a default provided client ID/secret

    ### **Step 5**: Setup .env file.
    Create a file in the project root folder called `.env` with the following variables:

      - GEE_PROJECT - Your GEE project name
      - OUTPUT_DIR - data/ (or full path to the /data folder in DataCollection)
      - BATCH_SIZE - 100 - AOI batch size processing
      - AOI_STEP - 0.1
      - TILE_PIXELS - 256
      - NUM_WORKERS - 2 - used on HPC for scaling AOI generation and filtering - still limited by GEE concurrency, so set to 2 to be cautious
      - TILE_SCALE - 10
      - DRIVE_FOLDER - Your output folder for data to be collected in GDrive (e.g. forest_gain_tiles)
      - DRIVE_REMOTE - gdrive
      - HPC_REMOTE - Remote cluster connection and destination for file porting - the path to the repo's data folder. ideally, set an alias for HPC connection (if working on local machine and uploading to HPC, then use in the form hpc_alias:path, if working on HPC, just use path to data folder, like OUTPUT_DIR)
      - USE_HPC - 0 if local, 1 if processing from a HPC resource
      - EE_CREDENTIALS_PATH - Your Earth Engine credentials for GDrive authorisations
      - SEARCH_MODE - `asset` for limiting search area to geographic extent of deadtrees.earth product, or `global` for the whole world
      WANDB_USERNAME - Weights and Biases username for model training
    </details>