# Wine Quality Analysis

## Overview

This project focuses on analyzing wine quality based on various physicochemical properties using a machine learning approach. Specifically, it employs the k-Nearest Neighbors (k-NN) algorithm to classify wines into different quality categories.

## Dataset

The dataset used in this project is sourced from the UCI Machine Learning Repository. It comprises two datasets related to red and white wine samples from the north of Portugal. Each sample has several physicochemical properties and a quality rating.

Link to the dataset: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

## Features

- Physicochemical properties like fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, and alcohol percentage.
- Quality rating (target variable) based on sensory data, on a scale from 0 (very poor) to 10 (very excellent).

## Setup and Installation

1. Ensure you have Python installed on your machine.
2. Clone the repository: 
\```
git clone [https://github.com/JosephDeleon-1/wine-quality-analysis-mL.git]
\```
3. Navigate to the project directory and install the required libraries:
\```
cd [wine-quality-analysis-mL]
pip install numpy pandas scipy
\```
4. Run the script:
\```
python wine_quality_analysis.py
\```

## Results

The k-NN classifier was used to predict wine quality. Various metrics like accuracy, precision, recall, and F1 score were computed to evaluate the model's performance. A k-fold cross-validation approach was also used to determine the optimal value of k for the classifier.

## Future Enhancements

- Incorporate other machine learning models for comparison.
- Feature engineering to enhance model performance.
- Implement a web-based interface for interactive analysis.

## Contributing

Pull requests are welcome. Please ensure to update tests as appropriate.
