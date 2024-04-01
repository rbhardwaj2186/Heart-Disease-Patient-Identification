import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

# DOWNLOAD CSV FILE
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

# READ CSV FILE
df = pd.read_csv(csv_file)
print(df)

print(df.dtypes)

target = df.pop('target')


for column in df.columns:
    print(f"Column: {column}")
    unique_values = df[column].unique()
    print(unique_values)

# FINDING BINARY COLUMNS
binary_list = []
for column in df.columns:
    if column != 'target' and df[column].nunique() == 2:
        binary_list.append(column)
        print(f"Binary Column: {column}")

# FINDING CATEGORICAL COLIMNS
#category_list = []
#for column in df.columns:
 #   if column != 'target' and df[column].nunique() > 2 and df[column].nunique < 7:
  #      category_list.append(column)
   #     print(f"Categorical Column: {column}")

# DATAFRAME AS ARRAY
numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]
numeric_features.head()


tf.convert_to_tensor(numeric_features)
# In general, if an object can be converted to a tensor with tf.convert_to_tensor it can be passed anywhere you can pass a tf.Tensor.

# NORMALIZING
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)

# VISUALIZE DATA
print(normalizer(numeric_features.iloc[:3]))

# Use the normalization layer as the first layer of a simple model:
def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

# CALLBACKS
MC = keras.callbacks.ModelCheckpoint(
    './Models/mnist_tfds/mnist_h5.h5',
    monitor='val_loss',
    save_best_only='True',
    verbose=1
)

ES = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights='True'
)

LR = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch/2), verbose=1)
TB = keras.callbacks.TensorBoard('./Models/tb_logs/mnist_tfds')

ROP = keras.callbacks.ReduceLROnPlateau(
    patience=2,
    verbose=1,
    factor=0.01,
    min_delta=0.12,
    min_lr=0.01
)


model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE, callbacks=[ES, MC, ])

'''WITH TF.DATA HAVING SIMILAR DTYPES '''
numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
  print(row)

numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)

model = get_basic_model()
model.fit(numeric_batches, epochs=15)

# DATAFRAME AS DICTIONARY
numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))
for row in numeric_dict_ds.take(3):
  print(row)
# DICTIONARIES WITH KERAS TWO WAYS
# WAY 1: MODEL SUBCLASS STYLE

def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)
#model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

# KERAS FUNCTIONAL STYLE
inputs = {}
for name, column in numeric_features.items():
  inputs[name] = tf.keras.Input(
      shape=(1,), name=name, dtype=tf.float32)

print(inputs)

x = stack_dict(inputs, fun=tf.concat)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, x)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)
model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)
print(f' The prediction of first three rows : {model.predict(dict(numeric_features.iloc[:3]))}')


# FULL FEATURE MODEL

# PREPROCESSING HEAD
binary_feature_names = ['sex', 'fbs', 'exang']
# Binary features on the other hand do not generally need to be encoded or normalized.
categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']

# The next step is to build a preprocessing model that will apply appropriate preprocessing to each input and concatenate the results.
inputs = {}
for name, column in df.items():
  if type(column[0]) == str:
    dtype = tf.string
  elif (name in categorical_feature_names or
        name in binary_feature_names):
    dtype = tf.int64
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

print(inputs)

# BINARY INPUTS
# Since the binary inputs don't need any preprocessing, just add the vector axis, cast them to float32 and add them to the list of preprocessed inputs:

preprocessed = []

for name in binary_feature_names:
  inp = inputs[name]
  inp = inp[:, tf.newaxis]
  float_value = tf.cast(inp, tf.float32)
  preprocessed.append(float_value)

print(preprocessed)
# NUMERIC INPUTS
# Like in the earlier section you'll want to run these numeric inputs through a tf.keras.layers.Normalization layer before using them.
# The difference is that this time they're input as a dict. The code below collects the numeric features from the DataFrame,
# stacks them together and passes those to the Normalization.adapt method.

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

# Calling adapt() on a Normalization layer is an alternative to passing in mean and variance arguments during layer construction.
# A Normalization layer should always either be adapted over a dataset or passed mean and variance.
# During adapt(), the layer will compute a mean and variance separately for each position in each axis specified by the axis argument
#  To calculate a single mean and variance over the input data, simply pass axis=None to the layer.

numeric_inputs = {}
for name in numeric_feature_names:
  numeric_inputs[name]=inputs[name]

numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

preprocessed.append(numeric_normalized)

print(preprocessed)

# CATEGORICAL FEATURES
# To use categorical features you'll first need to encode them into either binary vectors or embeddings
# Since these features only contain a small number of categories, convert the inputs directly to one-hot vectors using the output_mode='one_hot' option,
# supported by both the tf.keras.layers.StringLookup and tf.keras.layers.IntegerLookup layers.

# EXAMPLE
vocab = ['a','b','c']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
print(lookup(['c','a','a','b','zzz']))

vocab = [1,4,7,99]
lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
print(lookup([-1,4,1]))

# To determine the vocabulary for each input, create a layer to convert that vocabulary to a one-hot vector:
for name in categorical_feature_names:
  vocab = sorted(set(df[name]))
  print(f'name: {name}')
  print(f'vocab: {vocab}\n')

  if type(vocab[0]) is str:
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
  else:
    lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

  x = inputs[name][:, tf.newaxis]
  x = lookup(x)
  preprocessed.append(x)

print(f'Final preprocessed head: {preprocessed}')

# ASSEMBLE THE PREPROCESS HEAD
# At this point preprocessed is just a Python list of all the preprocessing results, each result has a shape of (batch_size, depth):
# Concatenate all the preprocessed features along the depth axis, so each dictionary-example is converted into a single vector.
# The vector contains categorical features, numeric features, and categorical one-hot features:

preprocessed_result = tf.concat(preprocessed, axis=-1)
preprocessed_result

# Now create a model out of that calculation so it can be reused:
preprocessor = tf.keras.Model(inputs, preprocessed_result)

preprocessor(dict(df.iloc[:1]))

# CREATE AND TRAIN A MODEL
body = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

# Now put the two pieces together using the Keras functional API.
print(inputs)
x = preprocessor(inputs)

result = body(x)
print(result)

model = tf.keras.Model(inputs, result)

# Define the split index for your data
SPLIT = int(len(df) * 0.8)  # 80% for training, 20% for validation

# Split your features and target into training and validation sets
train_features = dict(df[:SPLIT])
train_target = target[:SPLIT]

val_features = dict(df[SPLIT:])
val_target = target[SPLIT:]

model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
model.fit(train_features, train_target, epochs=5, batch_size=BATCH_SIZE, validation_data=(val_features, val_target), callbacks=[ES, MC, ROP,])

# EVALUATING THE MODEL ON 3 PATIENTS FROM THE SAMPLE TO SEE THE RESULT ALONG WITH THEIR FEATURES

# Take a sample of features
sample_features1 = dict(df.sample(1))

# Predict using the model
prediction1 = model.predict(sample_features1)

# Convert logits to probabilities
probabilities1 = tf.nn.sigmoid(prediction1)

# Take a sample of features
sample_features2 = dict(df.sample(2))

# Predict using the model
prediction2 = model.predict(sample_features2)

# Convert logits to probabilities
probabilities2 = tf.nn.sigmoid(prediction2)

# Take a sample of features
sample_features3 = dict(df.sample(3))

# Predict using the model
prediction3 = model.predict(sample_features3)

# Convert logits to probabilities
probabilities3 = tf.nn.sigmoid(prediction3)

print(f'Patient 1 has chances of heart attack' if max(probabilities1.numpy()) > 0.5 else 'Patient 1 is safe')
print(f'The features for the Patient 1 is is: {sample_features1}')
#print(f'The probability for the Patient 1 to have a heart disease is (1:Yes, 0:No): {max(probabilities2.numpy())}')
print(f'Patient 1 has chances of heart attack' if max(probabilities2.numpy()) > 0.5 else 'Patient 1 is safe')
print(f'The features for the Patient 2 is is: {sample_features2}')
#print(f'The probability for the Patient 1 to have a heart disease is (1:Yes, 0:No): {max(probabilities3.numpy())}')
print(f'Patient 1 has chances of heart attack' if max(probabilities3.numpy()) > 0.5 else 'Patient 1 is safe')
print(f'The features for the Patient 3 is is: {sample_features3}')

'''ds = tf.data.Dataset.from_tensor_slices((
    dict(df),
    target
))

ds = ds.batch(BATCH_SIZE)

import pprint

for x, y in ds.take(1):
  pprint.pprint(x)
  print()
  print(y)

history = model.fit(ds, epochs=5)'''


