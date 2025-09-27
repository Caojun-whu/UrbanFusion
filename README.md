______________________________________________________________________

<div align="center">

# UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.xxxx.xxxx-B31B1B.svg)](https://arxiv.org/abs/)
<!--[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)-->

</div>
Dominik J. Mühlematter¹, Lin Che¹, Ye Hong¹, Martin Raubal¹, Nina Wiedemann¹²  

¹ ETH Zürich  
² Intel Corporation
## Abstract
<img src="assets/architecture_V1.png" alt="Method overview">

Forecasting urban phenomena such as housing prices and public health indicators requires the effective integration of various geospatial data. Current methods primarily utilize task-specific models, while recent foundation models for spatial representations often support only limited modalities and lack multimodal fusion capabilities. To overcome these challenges, we present UrbanFusion, a Geo-Foundation Model (GeoFM) that features Stochastic Multimodal Fusion (SMF). The framework employs modality-specific encoders to process different types of inputs, including street view imagery, remote sensing data, cartographic maps, and points of interest (POIs) data. These multimodal inputs are integrated via a Transformer-based fusion module that learns unified representations. An extensive evaluation across 41 tasks in 56 cities worldwide demonstrates UrbanFusion’s strong generalization and predictive performance compared to state-of-the-art GeoAI models. Specifically, it 1) outperforms prior foundation models on location-encoding, 2) allows multimodal input during inference, and 3) generalizes well to regions unseen during training. UrbanFusion can flexibly utilize any subset of available modalities for a given location during both pretraining and inference, enabling broad applicability across diverse data availability scenarios.

## Installation
#### Pip

```bash
# Clone project
git clone https://github.com/DominikM198/UrbanFusion
cd UrbanFusion

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate.bat  # On Windows

# Install PyTorch 2.6 (choose one of the commands below: CPU or CUDA)
#    See also: https://pytorch.org/get-started/
# --- Linux / macOS ---
# Example: CUDA 12.6
# pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
# Example: CPU-only
# pip install torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
# --- Windows (PowerShell or cmd) ---
# Example: CUDA 12.6
# pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
# Example: CPU-only
# pip install torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install package
pip install -e .
```

## Minimal Usage Example 
Using pretrained models for location encoding is straightforward. The example below demonstrates how to load the model and generate representations based solely on geographic coordinates (latitude and longitude), without requiring any additional input modalities.
```python
import torch
from huggingface_hub import hf_hub_download
from srl.multi_modal_encoder.load import get_urbanfusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Coordinates: batch of 32 (lat, lon) pairs
coords = torch.randn(32, 2).to(device)

# Placeholders for other modalities (SV, RS, OSM, POI)
placeholder = torch.empty(32).to(device)
inputs = [coords, placeholder, placeholder, placeholder, placeholder]

# Mask all but coordinates (indices: 0=coords, 1=SV, 2=RS, 3=OSM, 4=POI)
mask_indices = [1, 2, 3, 4]

# Load pretrained UrbanFusion model
ckpt = hf_hub_download("DominikM198/UrbanFusion", "UrbanFusion/UrbanFusion.ckpt")
model = get_urbanfusion(ckpt, device=device).eval()

# Encode inputs (output shape: [32, 768])
with torch.no_grad():
    embeddings = model(inputs, mask_indices=mask_indices, return_representations=True).cpu()
```
For a more comprehensive guide—including instructions on applying the model to downstream tasks and incorporating additional modalities (with options for downloading, preprocessing, and using contextual prompts with or without precomputed features)—see the following tutorials:

- [`UrbanFusion_coordinates_only.ipynb`](https://github.com/DominikM198/UrbanFusion/blob/main/tutorials/UrbanFusion_coordinates_only.ipynb)  
- [`UrbanFusion_multimodal.ipynb`](https://github.com/DominikM198/UrbanFusion/blob/main/tutorials/UrbanFusion_multimodal.ipynb)  


## Project Structure

The directory structure of new project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── encoder                  <- Modality-specific encoders configs
│   ├── experiment               <- Experiment configs (UrbanFusion + Baselines)
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs (e.g., local paths)
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs (UrbanFusion + Baselines)
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── h5_files                  <- Pretraining data (typically stored on a large, fast drive for efficiency)
│   └── svi_data
│       └── place-pulse-2.0           <- H5 dataset folder (place precomputed H5 files here)
│           ├── SVI                       <- PP2-M street-view images (not required if H5 is provided)
│           ├── sentinel2                 <- PP2-M sentinel-2 images (not required if H5 is provided)
│           └── OSM_basemaps              <- OSM basemaps (not required if H5 is provided)
│
└── svi_data
│   └── place-pulse-2.0            <- PP2-M tables and statistics (.csv, .tsv, .npy)
│       ├── downstreamtask_data        <- Downstreamtask datasets, configs, and results
│       ├── pdfm                       <- PDFM embeddings and geojson's
│       └── encoder_weights            <- Weights of OSM-MAE, GPS2Vec, SatCLIP, CSP
│
└── tutorials                 <- Tutorial notebooks
│
└── Notebooks                 <- Notebooks for preprocessing, and analysis of results
│
├── logs                      <- Logs generated by hydra and lightning loggers
│
├── scripts                   <- Scripts for training and evaluation
│   ├── downstream_tasks         <- Downstream tasks scripts
│   ├── eval_utils               <- Evaluation utils scripts
│   ├── loss                     <- UrbanFusion loss functions
│   ├── lr_schedule              <- Learning rate schedule scripts
│   ├── preprocessing            <- Preprocessing scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── srl                       <- Source code
│   ├── baselines                <- Baselines
│   ├── data                     <- Dataloading
│   ├── encoders                 <- Modality-specific encoders
│   ├── multi_modal_encoder      <- Mutlimodal UrbanFusion encoder
│   ├── OSM_Map_MAE              <- OSM-MAE model implementation
│   └── utils                    <- Utils
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

## Setup for Pretraining
#### PP2-M Dataset

UrbanFusion is pretrained on our **Place Pulse 2.0 – Multimodal (PP2-M)** dataset, which extends the original Place Pulse 2.0 [1] by including not only street-view images, but also Sentinel-2 satellite imagery, cartographic basemaps, and point-of-interest (POI) features. More details are available in our paper. The dataset is publicly released on [Hugging Face 🤗](https://huggingface.co/datasets/DominikM198/PP2-M). First, there is the original data, as well as an H5 file named `01_06_2025_legendre_polys_10_bge-small-en-v1.5.h5`, which includes precomputed features from the frozen backbone models for faster training. This file was generated by the script [`precompute_modality_features.py`](https://github.com/DominikM198/UrbanFusion/blob/main/scripts/preprocessing/Place_pulse_2/precompute_modality_features.py).

In addition, the same directory includes precomputed statistics of the pretraining data —  `mean_vector.npy`, `std_vector.npy`, and `scaling_vector.npy` — used to normalize predictions for the latent modality reconstruction loss. These statistics were computed using the H5 file mentioned above and the script: [`modality_reconstruction_statistics.py`](https://github.com/DominikM198/UrbanFusion/blob/main/scripts/preprocessing/Place_pulse_2/modality_reconstruction_statistics.py).

Both *UrbanFusion* and the baseline configurations perform  model pretraining, and generation and storage of out-of-sample location representations for downstream tasks.  For pretraining, in addition to the PP2-M datasets, the following two files are required at the specified paths:
- svi_data/place-pulse-2.0/downstreamtask_data/London_Energy_Usage/London_Energy_Usage_locations.csv
- svi_data/place-pulse-2.0/downstreamtask_data/London_Housing_Prices/London_Housing_Prices_locations.csv
  
Refer to the *Downstream tasks* section in the dataset documentation (subsections **Housing Prices** and **Energy Consumption**) for instructions on preparing this data. If you want to train the model **without saving predictions for Housing Prices and Energy Consumption**, remove the following line from the experiment configuration:

```yaml
Data:
  coordinate_predictions:
```

The original modality folders (`SVI/`, `sentinel2/`, `osm_basemaps/`, and `osm_pois/`) **are not required** if you are using the precomputed H5 file. If you use the H5 file setup, make sure to download all necessary files and maintain the relative paths defined in:  
- [`configs/experiment/placepulse2/UrbanFusionV2_1.yaml`](https://github.com/DominikM198/UrbanFusion/blob/main/configs/experiment/placepulse2/UrbanFusionV2_1.yaml).
- [`configs/data/place_pulse2.yaml`](https://github.com/DominikM198/UrbanFusion/blob/main/configs/data/place_pulse2.yaml).
  
Absolute paths can be modified in:  
- [`configs/local/paths_local.yaml`](https://github.com/DominikM198/UrbanFusion/blob/main/configs/local/paths_local.yaml).


Alternatively, the original download and preprocessing scripts used to construct the dataset from scratch can be found in [`scripts/preprocessing/Place_pulse_2/`](https://github.com/DominikM198/UrbanFusion/blob/main/scripts/preprocessing/Place_pulse_2/)..





#### Pretrained Models
We release the pretrained weights of **UrbanFusion** on [Hugging Face 🤗](https://huggingface.co/DominikM198/UrbanFusion) to support reproducibility, along with out-of-sample representations of the PP2-M model used for downstream task evaluation.

#### OpenStreetMap Basemap Encoder
We fine-tuned a Vision Transformer (ViT) using the Masked Autoencoder (MAE) framework on cartographic basemaps from OpenStreetMap. The resulting model is available on [Hugging Face 🤗](https://huggingface.co/DominikM198/OSM-MAE/tree/main). For pretraining UrbanFusion, please place the model weights in the following directory:  
[SpatialFoundationModel/svi_data/place-pulse-2.0/encoder_weights](https://github.com/DominikM198/UrbanFusion/tree/main/svi_data/place-pulse-2.0/encoder_weights).


#### Baselines
In our study, we evaluate existing models for spatial representation learning across a wide range of urban prediction tasks. For a fair comparison, we pretrained all methods on the same PP2-M dataset whenever possible. We also report results using the original pretrained weights, although direct comparisons can be challenging due to differences in dataset size and coverage across learning frameworks.  

Specifically, we evaluate the following models:

- **GeoCLIP** [2]: We use the official codebase and evaluate both the original pretrained weights and weights further pretrained on PP2-M. Setup instructions are provided [here](srl/baselines/geoclip/model/gps_gallery) and [here](srl/baselines/geoclip/model/gps_gallery/README.md).
- **SatCLIP** [3]: We use the official codebase and evaluate both the original pretrained weights and those further pretrained on PP2-M, considering \(L = 10\) and \(L = 40\), along with an ablation study at \(L = 100\). Setup instructions are available [here](https://github.com/DominikM198/UrbanFusion/tree/main/svi_data/place-pulse-2.0/encoder_weights). Our experiments focus on the ViT variants.
- **GAIR** [4]: We implemented our own version of GAIR pretrained on PP2-M based on the corresponding paper, with two modifications: (i) we omit the INR module, and (ii) we use different encoders for the modalities. Specifically, we adopt the same location, remote sensing, and street-view encoders as in UrbanFusion, GeoCLIP, and SatCLIP, to facilitate fairer comparison across methods. Setup instructions are provided [here](srl/baselines/GAIR/model/gps_gallery).
- **CSP** [5]: We use the location encoders from CSP, pretrained on iNaturalist2018 and fMoW. Setup instructions can be found [here](https://github.com/DominikM198/UrbanFusion/tree/main/svi_data/place-pulse-2.0/encoder_weights).
- **GPS2Vec** [6][7]: Although not a single foundation model but rather a collection of local models, we include GPS2Vec and GPS2Vec+ as additional baselines. We refer to them as GPS2Vec (tag) and GPS2Vec (visual), respectively. Setup instructions are available [here](https://github.com/DominikM198/UrbanFusion/tree/main/svi_data/place-pulse-2.0/encoder_weights).
- **PDFM** [8]: We use Google’s Population Dynamics Foundation Model (PDFM), pretrained on multimodal data at the county and ZIP code levels in the United States. While the model itself is not open source, embeddings can be requested. Instructions are provided [here](https://github.com/DominikM198/UrbanFusion/tree/main/srl/baselines/PDFM/pdfm_embeddings/v0/us).

## Training and Evaluation of UrbanFusion

After downloading and preparing the data as described in **Setup for Pretraining**, both UrbanFusion and the baseline models can be trained and evaluated.

### Training UrbanFusion

To train a model, use an experiment configuration from the `configs/experiment/` directory. For example, to pretrain UrbanFusion:

    python scripts/train.py experiment=placepulse2/UrbanFusionV2_1

You can override any configuration parameter directly from the command line. For instance, to run training on CPU:

    python scripts/train.py experiment=placepulse2/UrbanFusionV2_1 trainer=cpu

The model weights, out-of-sample predictions and representations, as well as training logs, are saved to the log_dir [`configs/paths/default.yaml`](https://github.com/DominikM198/UrbanFusion/blob/main/configs/paths/default.yaml), unless overridden in [`configs/local/paths_local.yaml`](https://github.com/DominikM198/UrbanFusion/blob/main/configs/local/paths_local.yaml). More information about the software framework and its usage can be found in the [Lightning Hydra Template repository](https://github.com/ashleve/lightning-hydra-template).


### Evaluation and Generation of Representations for New Locations
Given pretrained weights, we can also load the model and evaluate it on the test data. This will also save predictions for out-of-sample locations:

    python scripts/eval.py experiment=placepulse2/UrbanFusionV2_1 ckpt_path="/path/to/ckpt/name.ckpt"

If representations based solely on coordinates are to be created, it is sufficient to add the path to the CSV under the key data: coordinate_predictions: in the corresponding experiment configuration. The only requirement is that the CSV contains geographical coordinates in columns named latitude and longitude.

Please note that UrbanFusion will generate meaningful representations only if the provided locations fall within the training region. For locations outside this region, it is recommended to collect additional modalities to serve as contextual prompts.
### Final Experiment Configurations

The following configurations were used for the final experiments. All files are located in the `configs/experiment/placepulse2/` directory:

- **UrbanFusionV2_1**  
  Original UrbanFusion model.

- **UrbanFusionV2ablation_1**  
  Ablation study: Training UrbanFusion on data with missing modalities. Each location has coordinates and 1–4 other modalities.

- **UrbanFusionV2ablation_2**  
  Ablation study: Training UrbanFusion on data where each location has only coordinates and a single modality.

- **UrbanFusionV2ablation_3**  
  Ablation study: Training UrbanFusion using only the contrastive location alignment objective.

- **UrbanFusionV2ablation_4**  
  Ablation study: Training UrbanFusion using only the latent modality reconstruction objective.

### Baselines

The `baselines/` folder contains all configuration files for pretraining and evaluation of baseline models.

### SyntheticPID Experiments

The `syntheticPID/` folder contains experiments evaluating UrbanFusion and baseline models on synthetic datasets.

## Downstream Tasks
We evaluate UrbanFusion and the baselines across a wide range of urban prediction tasks, including both regression and classification. In this process, we utilize the generated representations as input features to predict the target variable. We consider three different settings: 

1. **Coordinate-Only Encoding**: This setting uses only coordinates as input to the foundation models to generate representations. 

2. **Multimodal Encoding**: In this setting, any subset of supported modalities can be used as contextual prompts.

3. **Cross-Regional Generalization**: Here, we apply the model to regions that were not included in the training data.

For our downstream learners, we employ ridge regression, logistic regression, and multi-layer perceptrons (MLPs).

#### Datasets

**Housing Prices**  
The dataset is available on [Kaggle](https://www.kaggle.com/datasets/jakewright/house-price-data).  
Preprocessing is handled in [`scripts/downstream_tasks/preprocessing_housing_energy.ipynb`](https://github.com/DominikM198/UrbanFusion/blob/main/scripts/downstream_tasks/preprocessing_housing_energy.ipynb).  
This dataset provides housing price data used for regression tasks.

**Energy Consumption**  
Download the dataset from the [UK Government website](https://www.gov.uk/government/statistics/postcode-level-electricity-statistics-2023).  
Additionally, [UK postcode latitude and longitude data](https://www.freemaptools.com/download-uk-postcode-lat-lng.htm) is required.  
Preprocessing is handled in [`scripts/downstream_tasks/preprocessing_housing_energy.ipynb`](https://github.com/DominikM198/UrbanFusion/blob/main/scripts/downstream_tasks/preprocessing_housing_energy.ipynb).  
This dataset supports modeling energy consumption at the postcode level.

**Crime Incidence**  
Data for the year 2021 is available on [OSF](https://osf.io/zyaqn).  
This dataset includes spatial crime statistics for downstream tasks.

**Urban Perception**  
The PP2-M dataset can be found on [Hugging Face 🤗](https://huggingface.co/datasets/DominikM198/PP2-M).
This dataset includes visual and perceptual data from urban scenes.

**ZIP Code Tasks**  
Download `conus27.csv` and `zcta_poverty.csv` from the [Google Research Population Dynamics benchmarks](https://github.com/google-research/population-dynamics/tree/master/data/benchmarks).  
The file `zcta.geojson` can be requested from the same [repository](https://github.com/google-research/population-dynamics/tree/master).  
These files contain demographic and geographic information at the ZIP code level.

**Landcover**  
Download from the [USGS National Land Cover Database](https://www.usgs.gov/centers/eros/science/annual-national-land-cover-database).  
This dataset provides yearly land cover classifications across the United States.

**Landuse**  
Obtain data from the [Copernicus Urban Atlas 2018](https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-2018) for the following cities:  
Praha, Berlin, München, København, Madrid, Barcelona, Helsinki, Paris, Grad Zagreb, Dublin, Roma, Milano, Amsterdam, Warszawa, Lisboa, Bucuresti, Stockholm, Bratislava, London, Glasgow.  
This dataset contains detailed land use classifications for selected European cities.

> 💡 **Note:** After downloading the datasets, you may need to adapt paths in the downstream scripts. See the relevant notebooks and scripts under [`scripts/downstream_tasks/`](https://github.com/DominikM198/UrbanFusion/tree/main/scripts/downstream_tasks).

#### Evaluate Downstream Task

The configurations for the downstream tasks can be found in the folder [`svi_data/place-pulse-2.0/downstreamtask_data/results`](https://github.com/DominikM198/UrbanFusion/tree/main/svi_data/place-pulse-2.0/downstreamtask_data/results). These configurations define how each downstream evaluation should be run.

Further instructions for evaluating downstream tasks are provided in the relevant scripts under [`scripts/downstream_tasks/`](https://github.com/DominikM198/UrbanFusion/tree/main/scripts/downstream_tasks). Paths may need to be adapted before execution, depending on your environment or dataset locations.


## Citation
If you find our work useful or interesting or use our code, please cite our paper as follows
```latex
@article{muehlematter2025urbanfusion,
  title   = {UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations},
  author  = {Dominik J. Mühlematter and Lin Che and Ye Hong and Martin Raubal and Nina Wiedemann},
  year    = {2025},
  journal = {arXiv preprint arXiv:xxxx.xxxxx}
}

```

## References

[1] Dubey, A., Naik, N., Parikh, D., Raskar, R., and Hidalgo, C. A. (2016). Deep learning the city: Quantifying urban perception at a global scale. In European Conference on Computer Vision (ECCV), pages 196–212.
[2] Vivanco, V., Nayak, G. K., and Shah, M. (2023). Geoclip: Clip-inspired alignment between locations and images for effective worldwide geo-localization. In Advances in Neural Information Processing Systems.<br>
[3] Klemmer, K., Rolf, E., Robinson, C., Mackey, L., and Rußwurm, M. (2025). Satclip: Global, general-purpose location embeddings with satellite imagery. Proceedings of the AAAI Conference on Artificial Intelligence, 39(4):4347–4355.<br>
[4] Liu, Z., Zhang, F., Jiao, J., Lao, N., and Mai, G. (2025). Gair: Improving multimodal geo-foundation model with geo-aligned implicit representations. arXiv preprint arXiv:2503.16683.<br>
[5] Mai, G., Lao, N., He, Y., Song, J., and Ermon, S. (2023). Csp: Self-supervised contrastive spatial pre-training for geospatial-visual representations. In International Conference on Machine Learning.<br>
[6] Yin, Y., Liu, Z., Zhang, Y., Wang, S., Shah, R. R., and Zimmermann, R. (2019). Gps2vec: Towards generating worldwide gps embeddings. In Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, pages 416-419.<br>
[7] Yin, Y., Zhang, Y., Liu, Z., Liang, Y., Wang, S., Shah, R. R., and Zimmermann, R. (2021). Learning multi-context aware location representations from large-scale geotagged images. In Proceedings of the 29th ACM International Conference on Multimedia, pages 899-907.<br>
[8] Agarwal, M., Sun, M., Kamath, C., Muslim, A., Sarker, P., Paul, J., Yee, H., Sieniek, M., Jablonski, K., Mayer, Y., Fork, D., de Guia, S., McPike, J., Boulanger, A., Shekel, T., Schottlander, D., Xiao, Y., Manukonda, M. C., Liu, Y., Bulut, N., el haija, S. A., Eigenwillig, A., Kothari, P., Perozzi, B., Bharel, M., Nguyen, V., Barrington, L., Efron, N., Matias, Y., Corrado, G., Eswaran, K., Prab- hakara, S., Shetty, S., and Prasad, G. (2024). General geospatial inference with a population dynamics foundation model. arXiv preprint arXiv:2411.07207.<br>
