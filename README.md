<div align="center">

# Large Scale Pest Recognition

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/
)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![Paper](https://img.shields.io/badge/Paper_Publisher-Elsevier-orange)](https://www.sciencedirect.com/science/article/abs/pii/S0168169922005191)
[![Journal](http://img.shields.io/badge/Journal-Computers_&_Electronics_In_Agriculture-blue)](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/vol/200/suppl/C)

</div>

<br>
<br>

This repository contains the official implementation of a large-scale pest recognition system based on our research paper titled "[Large scale pest classification using efficient Convolutional Neural Network with augmentation and regularizers](https://www.sciencedirect.com/science/article/abs/pii/S0168169922005191)". The system utilizes efficient convolutional neural networks (CNNs) along with data augmentation and regularizers to classify images of pests into different categories.


## Description

Insect pest classification plays a crucial role in various domains, including agriculture, pest control, and ecological research. Rapid and accurate identification of insect pests is essential for effective pest management strategies, early detection of invasive species, and preservation of crop yield and quality. However, manual classification of insects based on visual inspection can be time-consuming, error-prone, and challenging, particularly when dealing with large-scale datasets.

Automated insect pest classification systems leveraging deep learning techniques offer a promising solution to this problem. By utilizing Convolutional Neural Networks (CNNs) and advanced image processing algorithms, these systems can effectively distinguish and categorize insect pests based on their visual characteristics. Such systems enable researchers, farmers, and pest control professionals to quickly identify pests, understand their behavior, and implement targeted control measures.

The implementation presented in this repository aims to provide a practical and efficient solution for large-scale insect pest classification. By leveraging an efficient CNN architecture along with augmentation techniques and regularizers, this implementation serves as a valuable resource for academic and research purposes. It can assist researchers, practitioners, and enthusiasts in exploring and advancing the field of insect pest classification, contributing to improved pest management strategies, ecosystem monitoring, and sustainable agricultural practices.

Please note that this implementation is intended for academic and research purposes only. It serves as a foundation for further studies and experimentation in insect pest classification. It is important to consider the specific requirements and limitations of practical applications before directly implementing this code in operational systems. 

## Dataset
The pest classification system relies on the IP102 dataset, a large-scale benchmark dataset specifically designed for insect pest recognition. The IP102 dataset is introduced in the paper "[IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition](https://ieeexplore.ieee.org/document/8954351)" presented at CVPR 2019 by Wu et al. The IP102 dataset contains a diverse collection of insect pest images with detailed annotations. It covers 102 categories of insect pests commonly found in agricultural and natural environments. The dataset provides a comprehensive representation of different pest species, including various insects and pests that impact crops, forests, and ecosystems.

To obtain the IP102 dataset, please follow these steps:
1. Visit the IP102 dataset repository on GitHub: [IP102 Dataset Repository](https://github.com/xpwu95/IP102)
2. Follow the instructions provided in the repository to download the dataset. You may need to agree to the dataset license terms and conditions.
3. Choose classification dataset, once you have downloaded the IP102 dataset, move the .tar file and classes.txt to `data/` folder in this repository. Preprocessing the dataset will be done in training pipeline.

Using the IP102 dataset, the pest classification system presented in this repository can effectively learn to recognize and classify insect pests, enabling accurate pest identification and supporting various applications in agriculture and pest control.

## Project Structure

The directory structure of new project looks like this:

```
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks, if exist. 
│
├── pest_rec               <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── scripts                <- Shell scripts
│
├── tests                  <- Tests of any kind
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
 
## Installation

We provide training from both a script and a package. Training from a package makes it easier to implement on different datasets without the need to modify or struggle with the code within the training framework. However, it less customization options. On the other hand, training from a script is highly customizable but requires more effort to make changes based on your specific use case. In this approach, you would need to delve into the code and manually customize it to fit your needs.

### Training from Script
#### Pip (Recommended)

```bash
# clone project
git clone https://github.com/adhiiisetiawan/large-scale-pest-recognition
cd large-scale-pest-recognition

# create virtual environment
python3 -m venv [your-environment-name]

# activate environment
source [your-environment-name]/bin/activate

# install pytorch according to instructions, choose pytorch with GPU if you have a GPU in your machine
# https://pytorch.org/get-started/

# install requirements
pip3 install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/adhiiisetiawan/large-scale-pest-recognition
cd large-scale-pest-recognition

# create conda environment and install dependencies
conda env create -f environment.yaml -n [your-environment-name]

# activate conda environment
conda activate [your-environment-name]
```

### How to run training with script

The training pipeline implemented in the paper consists of two steps:

**Step 1**: Freezing All Convolutional Layers.<br>
In this step, all the convolutional layers of the model are frozen. Freezing the layers means that their weights are not updated during the training process. By freezing the convolutional layers, the model utilizes the pre-trained features and focuses on fine-tuning the fully connected layers. This step helps the model to learn high-level representations and extract relevant features specific to the pest classification task.

**Step 2**: Fine-tuning and Unfreezing All Layers.<br>
In the second step, all layers, including the convolutional layers, are unfrozen and made trainable. This allows the model to further fine-tune the learned representations by adjusting the weights of all layers based on the pest classification task. By unfreezing all layers, the model can adapt and learn more task-specific features, improving its performance on the given classification problem.

By following this two-step training pipeline, the model benefits from the transfer learning approach, leveraging the pre-trained weights from the convolutional layers and then fine-tuning the entire network to improve its performance on the insect pest classification task.

---
To run the repository and utilize the pipeline as implemented in the paper, please follow these instructions:
1. Open the [Makefile](https://github.com/adhiiisetiawan/large-scale-pest-classification/blob/main/Makefile) provided in the repository.
2. Locate the train target within the Makefile. You will find the following command:
   ```bash
   train: 
    python pest_rec/train.py trainer=gpu model.net.freeze=true logger=wandb
   ```
   - `trainer=gpu`: This parameter indicates the use of GPU for training. By utilizing GPU acceleration, the training process can be significantly faster compared to using the CPU.
   - `model.net.freeze=true`: This parameter freezes all convolutional layers of the model. The frozen layers will utilize the pre-trained features and focus on fine-tuning the fully connected layers.
   - `logger=wandb`: This parameter sets the logger to use WandB for logging the training process. WandB (Weights & Biases) is a platform that provides tools for visualizing and tracking experiments. It allows you to monitor various metrics, visualize training progress, and compare different runs.<br><br>
   **Note**: You can modify the command based on your specific requirements. For example, you can use trainer=cpu if you want to use the CPU trainer instead of the GPU trainer. If you don't plan to use WandB logger, you can omit the logger=wandb parameter.

4. To start training, simply just type `make train` in root project and training will start automatically from data preparation until training done.
5. After training step 1 is done, you can continue to step 2, which is fine tuning with unfreeze all parameters. Change the Makefile like this.
   ```bash
   train: 
    python pest_rec/train.py \
     ckpt_path='./logs/train/runs/2023-06-28_23-10-21/checkpoints/epoch_009.ckpt' \
     trainer=gpu \
     model.net.freeze=false \
     logger=wandb \
     logger.wandb.id=b49b9fpd
   ```
   - `ckpt_path='./logs/train/runs/2023-06-28_23-10-21/checkpoints/epoch_009.ckpt'`: This parameter specifies the path to the checkpoint file of the model. Checkpoints are saved weights and parameters of the model at a specific epoch during training. By loading a specific checkpoint, you can resume training from that point or use it for evaluation. This should **change** with specific location in your case.
   - `trainer=gpu`: This parameter indicates the use of GPU for training. By utilizing GPU acceleration, the training process can be significantly faster compared to using the CPU.
   - `model.net.freeze=false`: This parameter unfreezes all layers of the model. In step 1, the convolutional layers were frozen, but now all layers, including the convolutional layers, will be trainable. This allows the model to update the weights and adapt to the pest classification task by fine-tuning its parameters.
   - `logger=wandb`: This parameter sets the logger to use WandB for logging the training process. WandB (Weights & Biases) is a platform that provides tools for visualizing and tracking experiments. It allows you to monitor various metrics, visualize training progress, and compare different runs.
   - `logger.wandb.id=b49b9fpd`: This parameter specifies the unique identifier (ID) for the WandB run. It helps to associate the training run with a specific experiment or configuration in the WandB platform, also the logger can continue from previous training in step 1. Making it easier to track and analyze the results. The wandb id should change with your wandb run id like in step 1.

**Note:** It is possible to run the command without the Makefile if you prefer. However, it can become slightly cumbersome when dealing with multiple arguments, as in step 2. The Makefile simplifies the process and makes it easier to handle.

### Training from Package
#### Install this repository as package
```bash
# clone project
git clone https://github.com/adhiiisetiawan/large-scale-pest-recognition
cd large-scale-pest-recognition

# install as package
pip install -e .
```

### How to run training with package
If you have already installed the package, you can use the installed package for quick and easy execution. Here's how to run the pest recognition system using the installed package:
```python
import lightning.pytorch as pl

from pest_rec.data.ip102_datamodule import IP102DataModule
from pest_rec.models.insect_pest_module import InsectPestLitModule


"""
train_dir, val_dir, and test_dir must have structure like this

root
├── label 1
│   ├── xxx.jpg
│   ├── xxy.png
│   └── xxz.jpg
└── label 2
    ├── 123.png
    ├── nsdf3.png
    └── asd932_.jpg
"""
datamodule = IP102DataModule(
    train_dir='your-training-dir',
    val_dir='your-validation-dir',
    test_dir='your-testing-dir')
datamodule.setup()

model = InsectPestLitModule(num_classes=102, freeze=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=3)
trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
```


## Result
The results obtained from running the pest classification system can be found in the paper "[Large scale pest classification using efficient Convolutional Neural Network with augmentation and regularizers](https://www.sciencedirect.com/science/article/abs/pii/S0168169922005191)". The paper presents comprehensive analyses of the experimental results, including performance comparisons and discussions.

## License
This project is licensed under the [MIT License](https://github.com/adhiiisetiawan/large-scale-pest-classification/blob/main/LICENSE), please read carefully about this license on `LICENSE` file. Feel free to use and modify the code for your purposes. However, please note that this repository does not provide any licenses or permissions for the dataset used in the project. Ensure that you comply with the terms and conditions of the dataset you use.

## Acknowledgments
We would like to express our sincere gratitude to the following individuals and resources for their valuable contributions to this project:
- Wu, X., Zhan, C., Lai, Y.K., Cheng, M.M. and Yang, J. for providing the IP102 dataset used in our research. The dataset, introduced in their paper titled "IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition" [1], has been instrumental in training and evaluating our insect pest classification model.
- The contributors of the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/) repository, which served as the foundation for our project. The project template's well-organized structure and modularity significantly expedited our development process and allowed us to focus on the core aspects of our research.

[1] Wu, X., Zhan, C., Lai, Y.K., Cheng, M.M. and Yang, J., 2019. IP102: A Large-Scale Nenchmark Dataset for Insect Pest Recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8787-8796).

## Citation 
If you find this code implementation or the accompanying paper useful in your research, please consider citing our work:
```
@article{setiawan2022large,
  title={Large scale pest classification using efficient Convolutional Neural Network with augmentation and regularizers},
  author={Setiawan, Adhi and Yudistira, Novanto and Wihandika, Randy Cahya},
  journal={Computers and Electronics in Agriculture},
  volume={200},
  pages={107204},
  year={2022},
  publisher={Elsevier}
}
```
