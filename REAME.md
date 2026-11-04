# DeepSea-Net: A Mediterranean Sea Image Dataset for Underwater Marine Organism Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Framework-Ultralytics%20YOLO-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Task](https://img.shields.io/badge/Task-Underwater%20Object%20Detection-green.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Marine%20Biodiversity-blue.svg)]()

## Overview

**DeepSea-Net** is a Mediterranean Sea underwater image dataset developed for fine-grained detection of marine organisms using deep learning and computer vision.

The dataset was constructed from remotely operated vehicle (ROV) video surveys conducted across four Mediterranean regions: Calabria, Puglia, Sardinia, and Sicily. It provides manually annotated underwater imagery representing diverse benthic and pelagic organisms under realistic deep-sea environmental conditions.

This repository provides the Python implementation and experimental workflows for dataset preparation, taxonomic remapping, model training, evaluation, annotation visualization, and automated detection in underwater images and videos.

The project investigates the application of YOLO-based object detection models to marine organism recognition.

## Dataset Overview

DeepSea-Net:

| Dataset characteristic | Description |
|---|---|
| Geographic coverage | Mediterranean Sea |
| Study regions | Calabria, Puglia, Sardinia, Sicily |
| Original extracted images | 4,829 |
| Original annotated instances | 16,579 |
| Additional category | Plastic debris |
| Annotation format | YOLO bounding boxes |
| Annotation tool | CVAT |
| Data acquisition | ROV underwater video |


### Data Acquisition

The underwater imagery was collected using a remotely operated vehicle. The source material includes 114 ROV video transects, representing approximately 649 hours of underwater footage.


## Dataset Annotation

All images were manually annotated by domain experts using CVAT. Each visible organism was assigned a taxonomic label and annotated using a bounding box. The annotations were exported in the Ultralytics YOLO detection format:

```text
class_id x_center y_center width height
```

## Dataset Structure

```text
DeepSea-Net/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels_class/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels_phylum/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels_taxa/
│   ├── train/
│   ├── val/
│   └── test/
│
├── data_class.yaml
├── data_phylum.yaml
├── data_taxa.yaml
│
├── taxonomy_classification_CVAT.xlsx
```


### Taxonomic Detection 

The file `taxonomy_classification_CVAT.xlsx` contains the taxonomic reference information used to organize and map the annotated marine organisms across different hierarchical levels. It supports the mapping of original taxon labels to their corresponding taxonomic classes, phyla, etc..

### Dataset Configurations

Three YAML files are provided:

- **`data_taxa.yaml`** – Configuration for fine-grained detection across at taxa level.
- **`data_class.yaml`** – Configuration for detection at the biological class level.
- **`data_phylum.yaml`** – Configuration for detection at the phylum level.

These configurations allow researchers to train and evaluate object-detection models at different taxonomic levels without changing the underlying image collection.


### Taxonomic Classification and Remapping

The repository includes a taxonomy-based remapping script:

`dataset_preparation/Taxonomy_Class_Remap.py`

This script uses the Excel-based taxonomic classification table to map the original taxa-level annotation categories to higher taxonomic levels. The resulting dataset supports experiments at different taxonomic levels.


## Dataset Split

The final dataset contains 4,693 images and 15,652 annotations.

The dataset was randomly divided into training, validation, and testing subsets.

| Subset | Images | Percentage (approx.) |
|---|---:|---:|
| Training | 3,998 | 85% |
| Validation | 461 | 10% |
| Testing | 234 | 5% |
| **Total** | **4,693** | **100%** |

Data augmentation was applied exclusively to the training set.


## Model Training

The project uses Ultralytics YOLO object-detection architectures. The benchmark investigates two model families:

- YOLOv8: Nano, Small, Medium, Large, and Extra-Large.
- YOLOv11: Nano, Small, Medium, Large, and Extra-Large.

All models were initialized using COCO-pretrained weights and fine-tuned on DeepSea-Net.

### Training Configuration

The experimental configuration described in the paper includes:

| Parameter | Configuration |
|---|---|
| Framework | Ultralytics YOLO |
| Model families | YOLOv8 and YOLOv11 |
| Model variants | n, s, m, l, x |
| Pretrained weights | COCO |
| Input resolution | 1024 × 1024 |
| Annotation format | YOLO |
| Training approach | Transfer learning |
| Model selection | Best validtion checkpoint |

### Training Scripts

**train.py**

Provides a YOLO11s training workflow using pretrained weights.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=1024,
    batch=16,
    device=0,
    patience=20,
    pretrained=True,
    save=True,
    plots=True
)
```

Adjust the training parameters according to your hardware and experimental requirements.

## Model Evaluation

The trained models are evaluated using standard object-detection metrics:

- Precision
- Recall
- Mean Average Precision at IoU 0.5 (mAP@0.5)
- Mean Average Precision at IoU 0.5–0.95 (mAP@0.5:0.95)

Evaluation script:

`validation.py`


## Installation

Clone the repository and install the required packages.


## Data Availability
The DeepSea-Net dataset is available from the corresponding author upon request. For dataset access and related inquiries, please contact the corresponding author



## Code Availability
This repository provides the implementation and scripts associated with the DeepSea-Net research workflow. The source code, implementation details, and evaluation scripts are intended to support reproducibility and further research in underwater marine organism detection.


## Applications
Potential applications of DeepSea-Net include:

- Automated Mediterranean marine biodiversity monitoring.
- Deep-sea organism detection and recognition.
- ROV-based underwater image analysis.
- Marine ecological surveys.
- Taxonomic detection at different hierarchical levels.
- AI-assisted underwater image annotation.


## Citation
If you use DeepSea-Net or the associated implementation in your research, please cite the following work:
*DeepSea-Net: A Mediterranean Sea Image Dataset for Underwater Marine Organism Detection.*


## Acknowledgements
The authors acknowledge the Stazione Zoologica Anton Dohrn (SZN), Department of Research Infrastructure for Marine Biological Resources (RIMAR), and the scientific teams involved in the Mediterranean ROV expeditions, image acquisition, and marine organism annotation.
