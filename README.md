# Heart Disease Patient Identification
 ## To predict heart disease based on various health metrics.
 ***The project demonstrates various aspects of building a machine learning model with TensorFlow and Keras, including data preprocessing, model building, and training. It also showcases the use of the functional API for building complex models and the use of tf.data.Dataset for efficient data batching. The model’s predictions are based on the processed input data. The project ends by printing the prediction of the first three rows.***

 ***- Originating from 1988, this Kaggle dataset encompasses four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It comprises 76 attributes, including the predicted one, but published studies typically utilize a subset of 14. The “target” field signifies the existence of heart disease in the patient, with an integer value of 0 indicating no disease and 1 indicating disease.***

 ![Visualization-of-features-of-heart-disease-dataset](https://github.com/rbhardwaj2186/Heart-Disease-Patient-Identification/assets/143745073/a891d6d5-3fe1-4027-be22-a3a60e8cf48f)

 

 ### The Heart dataset contains the following features
 *age*
*sex*
*chest pain type (4 values)*
*resting blood pressure*
*serum cholestoral in mg/dl*
*fasting blood sugar > 120 mg/dl*
*resting electrocardiographic results (values 0,1,2)*
*maximum heart rate achieved*
*exercise induced angina*
*oldpeak = ST depression induced by exercise relative to rest*
*the slope of the peak exercise ST segment*
*number of major vessels (0-3) colored by flourosopy*
*thal: 0 = normal; 1 = fixed defect; 2 = reversable defect*

This project is a machine learning model built using TensorFlow and Keras. It’s designed to predict heart disease based on various health metrics. Here’s a breakdown of the project:

### Data Preparation: The project starts by downloading a CSV file containing heart disease data from a URL. The data is then read into a pandas DataFrame. The ‘target’ column, which indicates the presence of heart disease, is separated from the rest of the data.

### Data Exploration: The unique values in each column are printed out. Binary columns (those with only two unique values) are identified.

### Data Preprocessing: The numeric features (‘age’, ‘thalach’, ‘trestbps’, ‘chol’, ‘oldpeak’) are converted to tensors and normalized using a Normalization layer.

### Model Building: A basic model is defined using the Keras Sequential API. This model consists of the normalization layer followed by two dense layers with ReLU activation and a final dense layer.

### Model Training: The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as the metric. It’s then trained on the numeric features and target data. Various callbacks are defined for model checkpointing, early stopping, learning rate scheduling, TensorBoard logging, and learning rate reduction on plateau.

### Data Batching: The numeric features and target data are converted to a tf.data.Dataset, shuffled, and batched.

### Model Evaluation: The model is retrained on the batched data.

### Data Conversion: The numeric features are also converted to a dictionary dataset and used to train the model.

### Full Feature Model: A more complex model is built that includes binary and categorical features in addition to the numeric features. These features are preprocessed appropriately (binary features are cast to float32, numeric features are normalized, and categorical features are one-hot encoded) before being fed into the model.

### Model Training and Evaluation: The full feature model is compiled and trained on the entire dataset (including binary and categorical features). The training data is batched using tf.data.Dataset.
