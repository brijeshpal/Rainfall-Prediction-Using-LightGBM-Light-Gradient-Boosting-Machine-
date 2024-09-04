# Rainfall-Prediction-Using-LightGBM-Light-Gradient-Boosting-Machine-
Tech Stack: Machine Learning Boosting Technique (Gradient Boosting, XGBoost, Adaboost etc.)

The aim is to develop accurate and reliable prediction models that can support decision-making in 
agriculture, water resource management, disaster management, and urban planning. The 
methodology involves using LGBM for its efficiency in handling large datasets and its 
ability to reduce training time by optimizing split operations in decision trees. Additionally, 
an ensemble classifier combining Random Forest, XG Boost, Gaussian Naive Bayes, and 
LGBM through soft and hard voting is employed to improve prediction accuracy.

# Project structure
The project is structured as follows:

Data Collection Methodology:
The work done so far includes importing and preprocessing a dataset from Kaggle, which 
comprises 10 years of daily weather observations across Australia.

Data Cleaning and Preprocessing: The preprocessing steps involve handling missing data, encoding categorical and dependent variables, splitting the 
dataset into training and test sets, and feature scaling. Following this, an artificial neural 
network model is built, trained, and tested for its ability to predict rainfall.

Model Building Using Machine Learning: Gradient Boosting, XGBoost, Adaboost etc.

# Results

Ensemble of [RFC, KNC, GNB, LGBM, XGBoost]: 0.92
LGBMClassifier: 0.8652
XGBClassifier: 0.8625
RandomForestClassifier: 0.8619
KNeighborsClassifier: 0.8484
GaussianNB: 0.8249
Reccurent Neural Network: 0.7979
Convolutional Neural Network: 0.6105



# Conclusion

The conclusion drawn from the study highlights the significant potential of using 
advanced machine learning techniques, specifically Light Gradient Boosting Machine 
(LGBM) and ensemble classifiers, for the accurate prediction of rainfall. This research 
is pivotal for agricultural countries like India, where rainfall plays a crucial role in the 
economy and livelihoods of millions. By comparing various existing methodologies, 
including artificial neural networks, support vector machines, fuzzy logic, and 
regression models, the study identifies the unique advantages of LGBM, such as its 
efficiency in processing large datasets and its effectiveness in reducing training time 
through optimized split operations.

# Future Work

• Incorporation of more diverse data sources (Indian Weather Dataset)
• Integration with climate change models
• Cross-Regional validation and adaptation
• Real-Time prediction and deployment
• Flood prediction



