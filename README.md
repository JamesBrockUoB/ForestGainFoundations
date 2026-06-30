Project for identifying forest growth from EO data time series and downstream
reforestation site informatics.

<div align="center">
    
<h1>Forest Growth Foundations: Vision foundation model for distinguishing forest growth typology and restoration site informatics.</h1>


## Table of Contents
- [Preparation](#preparation)

### Preparation
    
- **Environment Installation**:
    <details open>
    
    **Step 1**: Create a virtual environment named `forest_growth_env` and activate it.
    ```python
    conda create -n forest_growth_env python=3.11.9
    conda activate forest_growth_env
    ```
    
    **Step 2**: Download or clone the repository.
    ```python
    git clone https://github.com/JamesBrockUoB/ForestGrowthFoundations.git
    cd ./ForestChat/DataCollection
    ```
    
    **Step 3**: Install dependencies.
    ```python
    pip install -r requirements.txt
    ```
    </details>

    **Step 4**: Setup .env file.
    Create a file in the project root folder called `.env` with the following variables:
      - GEE_PROJECT - Your GEE project name
      - OUTPUT_DIR - data/
      - OUTPUT_FILE - aois/aoi_filter_checkpoint.json
      - BATCH_SIZE - 500
      - AOI_STEP - 0.25
      - TILE_PIXELS - 128
      - NUM_WORKERS - 4
      - TILE_SCALE - 10
      - DRIVE_FOLDER - Your output folder for data to be collected in GDrive
      - DRIVE_REMOTE - gdrive
      - HPC_REMOTE - Remote cluster connection and destination for file porting