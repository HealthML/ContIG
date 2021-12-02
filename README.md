# ContIG: Self-supervised Multimodal Contrastive Learning for Medical Imaging with Genetics

This is the code implementation of the paper "ContIG: Self-supervised Multimodal Contrastive Learning for Medical Imaging with Genetics". 

If you find this repository useful, please consider citing our paper in your work:
```
@misc{contig2021,
      title={ContIG: Self-supervised Multimodal Contrastive Learning for Medical Imaging with Genetics}, 
      author={Aiham Taleb and Matthias Kirchler and Remo Monti and Christoph Lippert},
      year={2021},
      eprint={2111.13424},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

To run the experiments, you will have to have access to UK Biobank data (requires application) and will need to set up the data modalities properly.

We handle the paths to different external files with the [paths.toml](paths.toml).
Model checkpoints are stored in `CHECKPOINTS_BASE_PATH` (`='checkpoints'` by default).
For some parts, we use `plink` and `plink2` software, which you can download from [here](https://www.cog-genomics.org/plink/1.9/) and [here](https://www.cog-genomics.org/plink/2.0/). Unzip and set the corresponding paths in the [paths.toml](paths.toml) file.

## Python

Install the dependencies via
```bash
conda env create --file environment.yml
```

## Setting up image data

See [image_preprocessing](image_preprocessing) for the code. We first use `resize.py` to find the retinal fundus circle, crop to that part of the image, and then filter out the darkest and brightest images with `filtering_images.py`.

After preprocessing the images, make sure to set `BASE_IMG` in [paths.toml](paths.toml) to the directory that contains the directories `{left|right}/512_{left|right}/processed/`.

## Ancestry prediction

We only included individuals that were genetically most likely to be of european ancestry. We used the genotype-based prediction pipeline [GenoPred](https://github.com/opain/GenoPred); see documentation on the site, and put the path to the output (a `.model_pred` file in `tsv` format) into the `ANCESTRY` variable in [paths.toml](paths.toml).

This ancestry prediction can also be replaced by the UKB variable [22006](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=22006). In this case, create a `tsv` file with two columns, `IID` and `EUR`; set `EUR = 1` for caucasians and `EUR = 0` for others, and point the `ANCESTRY` variable in [paths.toml](paths.toml) to this file.
Explicit ancestry prediction and the caucasian variable are mostly identical, but our ancestry prediction is a little more lenient and includes a few more individuals.

## Setting up genetic data

We use three different genetic modalities in the paper.


### Setting up Raw SNPs

Raw SNPs work mostly without preprocessing and use the basic microarray data from UKB. Make sure to set the `BASE_GEN` path in [paths.toml](paths.toml) to the directory that contains all the `bed/bim/fam` files from the UKB.

### Setting up Polygenic Scores

PGS requires the imputed data. See [the pgs directory](pgs) for a reference to set everything up. Make sure to update the `BASE_PGS` to point to the output directory from that. 
We also include [a list of scores used in the main paper](pgs_traits.csv).

### Setting up Burden Scores

Burden scores are computed using the whole exome sequencing release from the UKB. We used [faatpipe](https://github.com/HealthML/faatpipe) to preprocess this data; see there for details. Update the `BASE_BURDEN` variable in [paths.toml](paths.toml) to include the results (should point to a directory with `combined_burdens_colnames.txt`, `combined_burdens_iid.txt` and `combined_burdens.h5`).

## Setting up phenotypic UKB data

Point the `UKB_PHENO_FILE` variable in [paths.toml](paths.toml) to the full phenotype `csv` file from the UKB data release and run `export_card()` from `data.data_ukb.py` to preprocess the data (only needs to be run once; there may be a bug with `pandas >= 1.3` on some systems, so consider using `pandas = 1.2.5` for this step).

You can ignore the `BLOOD_BIOMARKERS` variable, since it's not used in any of the experiments.

## Setting up downstream tasks

Download and unzip the downstream tasks from [PALM](http://ai.baidu.com/broad/download), [RFMiD](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid) and [APTOS](https://www.kaggle.com/c/aptos2019-blindness-detection/data) and point the `{PALM|RFMID|APTOS}_PATH` variables in [paths.toml](paths.toml) correspondingly.

UKB downstream tasks are set up with the main UKB set above.

## Training self-supervised models
### ContIG
In order to train models with our method **ContIG**, use the script `train_contig.py`. In this script, it is possible to set many of the constants used in training, such as `IMG_SIZE`, `BATCH_SIZE`, `LR`, `CM_EMBEDDING_SIZE`, `GENETICS_MODALITY` and many others. We provide default values at the beginning of this script, which we use in our reported values. Please make sure to set the paths to datasets in [paths.toml](paths.toml) beforehand. 

### Baseline models
In order to train the baseline models, each script is named after the algorithm: **SimCLR** `simclr.py`, **NNCLR** `nnclr.py`, **Simsiam** `simsiam.py`, **Barlow Twins** `barlow_twins.py`, and **BYOL** `byol.py`

Each of these scripts allow for setting all the relevant hyper-parameters for training these baselines, such as `max_epochs`, `PROJECTION_DIM`, `TEMPERATURE`, and others.  Please make sure to set the paths to datasets in [paths.toml](paths.toml) beforehand. 

## Evaluating Models
To fine-tune (=train) the models on downstream tasks, the following scripts are the starting points:
* For [APTOS](https://www.kaggle.com/c/aptos2019-blindness-detection/data) Retinopathy detection: use `aptos_diabetic_retinopathy.py` 
* For [RFMiD](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid) Multi-Disease classification: use `rfmid_retinal_disease_classification.py`
* For [PALM](https://palm.grand-challenge.org/) Myopia Segmentation: use `palm_myopia_segmentation.py`
* For UK Biobank Cardiovascular discrete risk factors classification: use `ukb_covariate_classification.py`
* For UK Biobank Cardiovascular continuous risk factors prediction (regression): use `ukb_covariate_prediction.py`

Each of the above scripts defines its hyper-parameters at the beginning of the respective files. A common variable however is `CHECKPOINT_PATH`, whose default value is `None`. If set to `None`, this means to train the model from scratch without loading any pretrained checkpoint. Otherwise, it loads the encoder weights from pretrained models.

## Running explanations

### Global explanations
Global explanations are implemented in [feature_explanations.py](feature_explanations.py). See the `final_plots` function for an example to create explanations with specific models.

### Local explanations
Local explanations are implemented in [local_explanations.py](local_explanations.py). Individuals for which to create explanations can be set with the `INDIVIDUALS` variable. See the `final_plots` function for an example to create explanations with specific models.


## Running the GWAS
The GWAS is  implemented in [downstream_gwas.py](downstream_gwas.py). You can specify models for which to run the GWAS in the `WEIGHT_PATHS` dict and then run the `run_all_gwas` function to iterate over this dict.

