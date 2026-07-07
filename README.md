<div align="center">
    
<h1>Forest Gain Foundations: Vision foundation model for distinguishing forest growth typology and restoration site informatics</h1>

</div align="center">

# Table of Contents
- [Preparation](#preparation)

## Preparation
    
- ### **Environment Installation**:
    <details open>
    
    #### **Step 1**: Create a virtual environment named `forest_growth_env` and activate it.
    ```python
    conda create -n forest_growth_env python=3.11.9
    conda activate forest_growth_env
    ```
    
    #### **Step 2**: Download or clone the repository.
    ```python
    git clone https://github.com/JamesBrockUoB/ForestGrowthFoundations.git
    cd ./ForestChat/DataCollection
    ```
    
    #### **Step 3**: Install dependencies.
    ```python
    pip install -r requirements.txt
    ```

    #### **Step 4**: Authorise GEE.

    This pipeline uses a hybrid authentication model:
    - **Service Account → Earth Engine computation (required everywhere)**
    - **EE User Login (`ee.Authenticate`) → Google Drive exports**
    - Both local and HPC run the same authentication logic — only credential *locations* differ

    #### One-time Earth Engine OAuth (Drive access)

    Required once per user account to enable `Export.image.toDrive`.

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

    #### Service Account Authentication (EE Compute)

    This step is required to enable Earth Engine processing in both **local and HPC environments**.

    The service account is used for:
    - image processing
    - batch exports
    - all non-interactive Earth Engine computation

    It is **separate from Google Drive authentication**.

    ---

    #### 1. Create a Google Cloud Service Account

    Go to the Google Cloud Console:

    https://console.cloud.google.com/

    #### Steps:

    1. Select or create a Google Cloud project
    2. Navigate to: **IAM & Admin → Service Accounts**
    3. Create a new service account (e.g. `gee-worker`)

    ---

    #### 2. Grant Required Permissions

    You must ensure the service account has Earth Engine access.

    In general:
    - Earth Engine API must be enabled for the project:
    - The service account must be associated with a project that has Earth Engine access

    No additional configuration is typically required beyond this.

    ---

    #### 3. Generate Service Account Key

    Create and download a **JSON key file**:

    - Go to the service account
    - Open the **Keys** section
    - Create a new key (JSON format)
    - Download the file

    This file is your authentication credential for Earth Engine compute.

    ---

    #### 4. Store the Key Securely

    Place the file somewhere stable:

    #### Local machine

    ~/.config/gee/service-account.json


    #### HPC system

    Place in a location near the project repo

    ### **Step 5**: Setup .env file.
    Create a file in the project root folder called `.env` with the following variables:

      - GEE_PROJECT - Your GEE project name
      - OUTPUT_DIR - data/
      - BATCH_SIZE - 25
      - AOI_STEP - 0.25
      - TILE_PIXELS - 128
      - NUM_WORKERS - 4
      - TILE_SCALE - 10
      - DRIVE_FOLDER - Your output folder for data to be collected in GDrive
      - DRIVE_REMOTE - gdrive
      - HPC_REMOTE - Remote cluster connection and destination for file porting
      - GOOGLE_APPLICATION_CREDENTIALS - Your service account credentials path + file name
      - DRIVE_AUTH_CREDENTIALS - Your Earth Engine credentials for GDrive authorisations
      - SEARCH_MODE - `asset` for limiting search area to geographic extent of deadtrees.earth product, or `global` for the whole world
      - PERIOD - The time period for collecting and processing imagery for. `p1` for 2017-2020, or `p2` for 2020-2024. Note that there are no pseudo-labels for forest typology for `p2`

    </details>