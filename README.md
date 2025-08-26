# Plant-Disease-Classification
This project focuses on multi-class plant disease classification using transfer learning. A standardized pipeline was built with TensorFlow/Keras, using ImageDataGenerator for preprocessing and stratified train/validation/test splits (39,333 / 9,846 / 12,307 samples). Five pretrained CNNs (InceptionV3, InceptionResNetV2, ResNet101V2, VGG16, MobileNetV2) were fine-tuned with custom dense layers (GlobalAveragePooling - Dense - Dropout - Softmax). Training was optimized with Adam (lr=1e-4), categorical cross-entropy, and EarlyStopping.

Final evaluation on the test set showed ResNet101V2 achieved 95.24% accuracy, outperforming the other architectures. The project demonstrates skills in computer vision, model benchmarking, and transfer learning
