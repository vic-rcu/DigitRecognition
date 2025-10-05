# Handwritten Digit Recognition - Paper Implementation

## Introduction  

This project implements a **Convolutional Neural Network (CNN)** for **handwritten digit recognition** on the MNIST dataset. Inspired by the classic work of *LeCun (1989)* and extended with modern deep learning practices, the goal is to design, train, and analyze a network that learns to classify digits (0–9) from 28×28 grayscale images.  

The project is structured as both a **learning exercise** and a **research-style exploration**. It goes beyond a baseline implementation by:  

-  Building a CNN architecture in **PyTorch** and training it with **Adam** and **SGD + Momentum** optimizers.  
-  Exploring the effects of **activation functions, pooling layers, and regularization (dropout)** on performance.  
-  Tracking key statistics (loss, accuracy, generalization gap, epoch time) and visualizing them with professional plots.  
-  Visualizing **convolutional kernels** and **feature maps** to gain insight into what the model learns at each layer.  


By the end of this project, the model achieves strong generalization on MNIST, reaching **99+%% validation accuracy**, with potential improvements via fine-tuning and hyperparameter optimization.  

## Repository Structure


```bash
├── data/                    # Dataset loading or preprocessing scripts
│   ├── utils.py
├── main/                    # Script to run demo of the project
│   ├── main.py
├── models/                  # Source code for both implemented models
│   ├── LeNet1989.py
│   ├── ModCNN.py
├── notebooks/               # Jupyter notebooks with analysis and experiments
│   ├── reprod.ipynb
│   ├── mod_CNN.ipynb
├── tests/                   # Tests for model outputs
│   ├── test_network.ipynb
├── training/                # Training and Finetuning Logic
│   ├── train.py
│   ├── tune.py
├── README.md     
└── requirements.txt         # Dependencies
```

## Part 1: Paper Implementation

The first part features the reimplementation of the paper **Handwritten Digit Recognition with a Back-Propagation Network** *(LeCun, 1989)*. The main goal here was to implement the [model](/models/LeNet1989.py) as presented in the original paper. The exact detail of this exploration can be found [here](/notebooks/reprod.ipynb). The overall reimplementation was a success, reaching  **96% accuracy**, while **generalizing the test well**. The model was very lighweight and its training setup is simple and quick. 

## Part 2: Modern CNN

The second part features the implementation of a modern **Convolutional Neural Network** to classify the MNIST dataset with **99+% accuracy**. The implemented [model](/models/mod_CNN.py) is more sophisticated, having significantly more filters and making use of more modern components **(MaxPooling, Dropuout, ReLU)**. The in-depth analysis of the model can be found [here](/notebooks/mod_CNN.ipynb). In summary, the desired accuracy was reached. 


## Running the Project

Using the following commands, a demo of training the reimplemented model from *(LeCun, 1989)* can be run. The gathered data is the printed to the terminal and the model is saved. 

```bash
# cloning the project
git clone https://github.com/<vic-rcu>/DigitRecognition
cd DigitRecognition

# running the project
pip install -r requirements.txt
python main/main.py

```


## Conclusion

Overall, the project was a success, implementing 2 different **CNNs**, training and tuning them to perform classification on the MNIST dataset. Through implementing this project, I was able to gain a deeper understanding of **PyTorch**, one of the most fundamental machine learning libraries. In particular, I was able to strengthen my understanding of **CNNs** and their modules, **training and finetuning infrastructure**, and **model evalutation techniques**. In the future, I might revisit this project adding more advanced concepts such as **hyperparameter tuning** or **data augmentation**. 

## References

- **LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989).**  
  *Handwritten Digit Recognition with a Back-Propagation Network.*  
  In **Advances in Neural Information Processing Systems (NIPS 1989)**.  
  [📄 Original Paper (PDF)](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)

- **MNIST Dataset**  
  *The MNIST Database of Handwritten Digits* — a benchmark dataset of handwritten digits.  
  [🌐 Official Page](http://yann.lecun.com/exdb/mnist/)  
  [📦 PyTorch Dataset Reference](https://pytorch.org/vision/stable/datasets.html#mnist)

- **PyTorch Documentation**  
  *Official PyTorch Library and API Reference.*  
  [📘 https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
