# Breast Cancer Image Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify breast cancer images as either benign or malignant. The model is built using TensorFlow and Keras, and it includes data preprocessing, training, validation, and evaluation.

## Project Structure

```
.
├── Dataset2
│   ├── FNA
│   │   ├── benign
│   │   └── malignant
│   └── test
├── output.txt
├── accuracy_plot.png
├── loss_plot.png
├── model_cnn.h5
└── cnn_model.py
```

- **Dataset2/**: Contains the training and test data.
  - **FNA/**: Directory with subdirectories for benign and malignant images.
  - **test/**: Directory containing unlabeled test images.
- **output.txt**: Logs the model training progress, evaluation metrics, and predictions on the test set.
- **accuracy_plot.png**: Plot showing training and validation accuracy.
- **loss_plot.png**: Plot showing training and validation loss.
- **model_cnn.h5**: Saved model file after training.
- **cnn_model.py**: The main script to run the CNN model.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install required libraries**:
   Make sure you have Python installed, and then run:
   ```bash
   pip install numpy pandas matplotlib tensorflow scikit-learn
   ```

## Usage

1. **Prepare your dataset**: 
   Organize your images in the `Dataset2/FNA` directory under `benign` and `malignant` subdirectories for training. Place unlabeled test images in the `Dataset2/test` directory.

2. **Run the CNN model**:
   Execute the Python script:
   ```bash
   python cnn_model.py
   ```

3. **Output**:
   After running the script, the following will be generated:
   - **output.txt**: Contains detailed logs of the training process, evaluation metrics, and predictions for test images.
   - **accuracy_plot.png**: A plot showing the training and validation accuracy over epochs.
   - **loss_plot.png**: A plot showing the training and validation loss over epochs.
   - **model_cnn.h5**: The trained model file that can be reused for future predictions.

## Model Architecture

The CNN model consists of the following layers:
- Convolutional layers for feature extraction.
- MaxPooling layers for downsampling.
- Dense layers for classification.

## Evaluation

The model is evaluated using the validation data, and the performance metrics, including accuracy and loss, are written to `output.txt`. A confusion matrix and classification report are also generated and saved in the same file.

## Notes

- Ensure that the images in the dataset are appropriately formatted and organized.
- The model is configured for binary classification and is trained for 10 epochs by default, but you can modify the training parameters in the script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
