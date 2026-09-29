# LE-BTS
A Lightweight Fusion Network for Brain Tumor Segmentation

If you have any questions or require further information, please do not hesitate to contact me at the following email address: zhangcaiyin1021@gmail.com.

# LE-BTS Installation and Usage Guide

## 1. Creating the Virtual Environment

First, create a virtual environment:

```bash
conda create -n nnunet python=3.10
```

Activate the environment:

```bash
conda activate nnunet
```

Install PyTorch and the corresponding CUDA version:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 2. Installing nnUNetv2

First, download the source code and install it. If nnUNet has already been downloaded, make sure that there are no custom-created folders inside the `nnUNet` directory.

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

## 3. Setting Environment Variables

First, create the following three folders:

```text
nnUNet_raw/
nnUNet_preprocessed/
nnUNet_results/
```

Then, configure the corresponding environment variables. The configuration method is different for Windows and Linux.

You can directly create the three folders `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` under the `nnUNet` directory.

### 3.1 Windows

Use the following commands and replace the paths with your own paths:

```cmd
set nnUNet_raw=D:/DeepLearning/nnUNet/nnUNet_raw
set nnUNet_preprocessed=D:/DeepLearning/nnUNet/nnUNet_preprocessed
set nnUNet_results=D:/DeepLearning/nnUNet/nnUNet_results
```

### 3.2 Linux

Use the following commands and replace the paths with your own paths:

```bash
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```

### 3.3 Directly Modifying nnUNetv2

Alternatively, you can directly modify the paths in:

```text
nnunetv2/paths.py
```

Set the paths of the three folders directly in this file.

## 4. Training and Testing

Download the BraTS2021 dataset and place it under the `nnUNet_raw` directory. Then extract the dataset.

### 4.1 Dataset Format Conversion

Locate `Dataset137_BraTS21.py` under the `dataset_conversion` directory and modify the corresponding dataset paths. Then run the script.

After conversion, the following dataset folder will be generated under `nnUNet_raw`:

```text
Dataset137_BraTS2021
```

### 4.2 Preprocessing

Run the following command:

```bash
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity
```

Here, `137` is the dataset ID assigned after conversion.

After preprocessing is completed, the processed data will be generated under the `nnUNet_preprocessed` directory.

**Issue 1:** After executing this command, some required packages may not be installed automatically. Install them manually using `pip install`.

**Issue 2:** If the following error occurs:

```text
ImportError: cannot import name 'crop_to_bbox' from 'acvl_utils.cropping_and_padding.bounding_boxes'
```

Install the specified version of `acvl-utils`:

```bash
pip install acvl-utils==0.2
```

### 4.3 Training

By default, nnUNet uses five-fold cross-validation. You can also train using only one fold.

For the `3d_fullres` configuration, the last number represents the fold index (`0`, `1`, `2`, `3`, or `4`). For example:

```bash
nnUNetv2_train 137 3d_fullres 0
```

The above command uses deep supervision.

For training **without deep supervision**, first configure the model in:

```text
training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerNoDeepSupervision.py
```

Then run:

```bash
nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainerNoDeepSupervision
```

### 4.4 Validation and Inference

#### Step 1: Find the Best Configuration

This step is optional and can be skipped depending on your requirements.

For the standard trainer:

```bash
nnUNetv2_find_best_configuration 137 -c 3d_fullres -f 0 1 2 3 4
```

For the trainer without deep supervision:

```bash
nnUNetv2_find_best_configuration 137 -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerNoDeepSupervision
```

#### Step 2: Convert the Test Dataset

Convert the data in the `Test` folder into the `imagesTs` and `labelsTs` formats using the previously mentioned dataset conversion script.

#### Step 3: Perform Inference

Create an `inferTs` folder under `nnUNet_raw` to store the inference results.

Modify the network configuration in:

```text
predict_from_raw_data.py
```

Then run:

```bash
nnUNetv2_predict -i nnUNet_raw/Dataset137_BraTS2021/imagesTs -o nnUNet_raw/Dataset137_BraTS2021/inferTs -d 137 -c 3d_fullres -f 0
```

Here, `-f 0` indicates that inference is performed using the first fold.

#### Step 4: Evaluate the Results

Use your own evaluation script to calculate the final segmentation metrics.

Alternatively, you can use the official nnUNet evaluation command:

```bash
nnUNetv2_evaluate_folder -ref nnUNet_raw/Dataset137_BraTS2021/labelsTs -pred nnUNet_raw/Dataset137_BraTS2021/inferTs -l 1 2 3
```

## 5. Other Settings

### 5.1 Changing the Batch Size and Patch Size

The `batch_size` and `patch_size` can be modified in:

```text
nnUNet_preprocessed/nnUNetPlans.json
```

### 5.2 Changing the Number of Epochs, Learning Rate, and Optimizer

The number of epochs, learning rate, and optimizer can be configured in:

```text
training/nnUNetTrainer/nnUNetTrainer.py
```
