import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Redirect print outputs to output.txt file
output_file = open('output.txt', 'w', encoding='utf-8')

# Directories for training and test data
train_dir = 'Dataset2/FNA'
test_dir = 'Dataset2/test'

# Image data generator for augmenting and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test generator for unlabeled test images
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare test images
test_images = [img for img in os.listdir(test_dir) if img.endswith('.jpg') or img.endswith('.png')]
if len(test_images) == 0:
    print("No test images found. Please check your test directory.")
else:
    test_images = [os.path.join(test_dir, img) for img in test_images]  # Full path for test images
    test_df = pd.DataFrame({'filename': test_images})
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col=None,
        target_size=(150, 150),
        class_mode=None,
        batch_size=32,
        shuffle=False)

    # Debugging: Check the shape of the first batch
    for batch in test_generator:
        print("Shape of the first batch:", batch.shape)  # Should be (batch_size, 150, 150, 3)
        break  # Remove this line if you want to see all batches

# CNN Model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the layer and add fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary(print_fn=lambda x: output_file.write(x + '\n'))

# Training the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the model
model.save('model_cnn.h5')

# Write training and validation accuracy/loss to output.txt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

output_file.write("\nTraining and Validation Metrics:\n")
for epoch in range(len(acc)):
    output_file.write(f"Epoch {epoch+1}: Training Accuracy: {acc[epoch]:.4f}, Validation Accuracy: {val_acc[epoch]:.4f}, "
                      f"Training Loss: {loss[epoch]:.4f}, Validation Loss: {val_loss[epoch]:.4f}\n")

# Plot training and validation accuracy
plt.figure()
plt.plot(range(1, len(acc) + 1), acc, 'b', label='Training accuracy')
plt.plot(range(1, len(acc) + 1), val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')  # Save the plot as an image
plt.close()

# Plot training and validation loss
plt.figure()
plt.plot(range(1, len(loss) + 1), loss, 'b', label='Training loss')
plt.plot(range(1, len(loss) + 1), val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')  # Save the plot as an image
plt.close()

# Predicting on the test set
if len(test_images) > 0:
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    predicted_labels = ['benign' if p < 0.5 else 'malignant' for p in predictions]

    output_file.write("\nPredicted labels for test images:\n")
    for i, label in enumerate(predicted_labels):
        output_file.write(f"Image {i+1}: {label}\n")

# Evaluate the model on validation data
validation_loss, validation_acc = model.evaluate(validation_generator)
output_file.write(f"\nValidation Accuracy: {validation_acc:.4f}\n")
output_file.write(f"Validation Loss: {validation_loss:.4f}\n")

# Confusion matrix and classification report
Y_pred = model.predict(validation_generator)
y_pred = np.where(Y_pred > 0.5, 1, 0)

conf_matrix = confusion_matrix(validation_generator.classes, y_pred)
class_report = classification_report(validation_generator.classes, y_pred, target_names=['benign', 'malignant'])

# Write confusion matrix and classification report to output.txt
output_file.write("\nConfusion Matrix:\n")
output_file.write(str(conf_matrix) + '\n')
output_file.write("\nClassification Report:\n")
output_file.write(class_report + '\n')

# Close the output file
output_file.close()
