# Vietnamese-Spam-Detection <a name="vietnamese-spam-detection"></a>

## Table of Contents
* [Vietnamese-Spam-Detection](#vietnamese-spam-detection)
	* [About](#about)
	* [Installation](#installation)
	* [Usage](#usage)
   	* [Contributors](#contributors)


## About <a name="about"></a>

This package contains code for spam detection in the Vietnamese language. It includes machine learning models trained on Vietnamese text data to classify messages as spam or non-spam. Additionally, it provides a user-friendly UI application built using the `PyQt6` library for demonstration and testing purposes.


## Installation <a name="installation"></a>

To use this package, follow these steps:

1. Clone the repository: `git clone https://github.com/ankhanhtran02/Vietnamese-Spam-Detection.git`
2. Navigate to the project directory: `cd Vietnamese-Spam-Detection`
3. Install dependencies: `pip install -r requirements.txt`

## Usage <a name="usage"></a>

### Running the UI App

To run the UI app for demonstration:

1. Ensure you have completed the installation steps.
2. Run the following command: `python main.py`
3. On the left side of the newly created window, you will see a text box which will contain the messages you want to classify. In order to classify new text messages, you can use the **Add text** and **Read text from file** buttons, after that, please press the **PREDICT** button in the middle. The output will appear on the right side of the app window. Messages classified as non-spams will be appended with the string `"[0]"` and have blue color, while spam messages will be appended with `"[1]"` and have red color. You can use our prepared text files contained in the demos folder. The algorithm used in making predictions can also be switched using the **Choose an algorithm** combobox.
	* You should see a window like this pop up when first running the application:
![App_User_Interface](pictures/App_User_Interface.png)

	* You can choose one from 5 algorithms listed in the combobox for making predictions:
![App_User_Interface_Algorithms](pictures/App_User_Interface_Algorithms.png)

	* After you have chosen a file or added some messages yourself and pushed the PREDICT button, a result like this will be shown:
![App_User_Interface2](pictures/App_User_Interface2.png)

### Running experiments

To run the experiments described in our report again and check for validity:
1. Ensure you have completed the installation steps.
2. Run the following commands: `python KNN.py`, `python SVM.py`, `python ANN.py`, `python logistic_regression.py`, `python naive_bayes.py` to see the evaluation on the test set of each of the 5 algorithms when using different vectorizers.
3. Run the command `python vectorizers_comparison.py` to see the comparison between different vectorizers when using the majority vote model.
4. Run the command `python baseline_system.py` to see the evaluation of the baseline model.

## Contributors <a name="contributors"></a>
We want to thank the following contributors for their valuable contributions to this project:
- [ankhanhtran02](https://github.com/ankhanhtran02): Preprocessing and logistic regression implementation, experimenting
- [Decent-Cypher](https://github.com/Decent-Cypher): KNN implementation, UI designing
- [KingNoob2022](https://github.com/KingNoob2022): ANN implementation
- [AndrewNguyen4](https://github.com/AndrewNguyen4): Naive Bayes implementation
- [Vinh.TT](https://www.facebook.com/vinh.truongtuan): SVM implementation

We also appreciate the help from our classmates, friends and families who contributed by adding more spam message samples to our dataset, which are crucial to the overall performance of our algorithms and validity of our experiments.

