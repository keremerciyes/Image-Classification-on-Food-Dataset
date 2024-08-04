# Image Classification on iFood Dataset

## Overview
This project aims to evaluate and compare the performance of traditional feature extraction methods with Convolutional Neural Networks (CNN) with one million parameter limit in image classification tasks. By systematically analyzing the strengths and limitations of each approach, the study provides insights into their effectiveness and practical applications, contributing to the advancement of robust image classification systems.

## Project Structure
- `data/`: Contains the iFood dataset images (downloaded from Kaggle).
- `scripts/`: Contains preprocessing scripts and model training scripts.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `models/`: Saved models and results.
- `README.md`: Project overview and setup instructions.
- `requirements.txt`: Required packages for the project.
- `presentation.pdf`: Project presentation.
- `project_report.pdf`: Detailed project report.
- `cnn.ipynb`: CNN implementation notebook.
- `sift-bow.ipynb`: SIFT-BOW implementation notebook.

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Installation
1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the iFood dataset from Kaggle:
    - Go to the [iFood 2019 FGVC6 competition page](https://www.kaggle.com/c/ifood-2019-fgvc6).
    - Download the dataset and extract it into the `data/` directory.

## Running the Project

### Preprocessing Data
1. Run the preprocessing scripts to prepare the dataset:
    ```bash
    python scripts/preprocess_data.py
    ```

### Training Models
1. Train the SIFT-BOW-SVM model:
    ```bash
    jupyter notebook notebooks/sift-bow.ipynb
    ```
   Follow the instructions within the notebook to execute each cell and train the model.

2. Train the CNN model:
    ```bash
    jupyter notebook notebooks/cnn.ipynb
    ```
   Follow the instructions within the notebook to execute each cell and train the model.

## Evaluation
- The results of the trained models, including accuracy, precision, recall, and F1-score, are saved in the `models/` directory.
- Comparative analysis of the models is included in the `project_report.pdf` and `presentation.pdf`.

## Project Details
### Data Preprocessing
- Images are resized, normalized, and augmented (rotation, flipping, scaling, translation, brightness, and contrast adjustments).

### SIFT-BOW-SVM
- SIFT is applied to detect and describe local features in images.
- Features are clustered using k-means to form a visual vocabulary.
- Images are represented as histograms of visual words and classified using an SVM with RBF kernel.

### CNN
- A Convolutional Neural Network is trained on the dataset using data augmentation.
- The network architecture includes convolutional layers, pooling layers, and fully connected layers.

## Results
- The CNN outperforms the traditional SIFT-BOW-SVM approach in terms of accuracy and robustness to image variations.
- Detailed results and analysis can be found in the `project_report.pdf` and `presentation.pdf`.

## Contributors
- Kerem Erciyes - [k.erciyes@campus.unimib.it](mailto:k.erciyes@campus.unimib.it)
- Mustafa Soydan - [m.soydan@campus.unimib.it](mailto:m.soydan@campus.unimib.it)


## Acknowledgments
- We would like to thank Prof. Simone Bianco and his assistant, Mirko Agarla, for their guidance and support throughout this project.

## License
This project is licensed under the MIT License.

