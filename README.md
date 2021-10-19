# ML in Practice


Source code for the Seminar "Machine Learning in Practice", taught at Osnabrück University in the winter term 2021/2022.

As data source, we use the "Data Science Tweets 2010-2021" data set (version 3) by Ruchi Bhatia from [Kaggle](https://www.kaggle.com/ruchi798/data-science-tweets). The goal of our example project is to predict which tweets will go viral, i.e., receive many likes and retweets.

---
<br>
<br>

## Virtual Environment

In order to install all dependencies, please make sure that you have a local [Conda](https://docs.conda.io/en/latest/) distribution (e.g., Anaconda or miniconda) installed. Begin by creating a new environment called "MLinPractice" that has Python 3.6 installed:

```conda create -y -q --name MLinPractice python=3.6```

You can enter this environment with `conda activate MLinPractice` (or `source activate MLinPractice`, if the former does not work). You can leave it with `conda deactivate` (or `source deactivate`, if the former does not work). Enter the environment and execute the following commands in order to install the necessary dependencies (this may take a while):

```
conda install -y -q -c conda-forge scikit-learn=0.24.2
conda install -y -q -c conda-forge matplotlib=3.3.4
conda install -y -q -c conda-forge nltk=3.6.3
conda install -y -q -c conda-forge gensim=4.1.2
conda install -y -q -c conda-forge spyder=5.1.5
conda install -y -q -c conda-forge pandas=1.1.5
conda install -y -q -c conda-forge mlflow=1.20.2
conda install -y -q -c conda-forge spacy
python -m spacy download en_core_web_sm
```

You can double-check that all of these packages have been installed by running `conda list` inside of your virtual environment. The Spyder IDE can be started by typing `~/miniconda/envs/MLinPractice/bin/spyder` in your terminal window (assuming you use miniconda, which is installed right in your home directory).

In order to save some space on your local machine, you can run `conda clean -y -q --all` afterwards to remove any temporary files.

The installed libraries are used for machine learning (`scikit-learn`), visualizations (`matplotlib`), NLP (`nltk`), word embeddings (`gensim`), and IDE (`spyder`), and data handling (`pandas`)

---
<br>
<br>

## Overall Pipeline

The overall pipeline can be executed with the script [code/pipeline.sh](https://github.com/team-one-ML/MLinPractice/blob/main/code/pipeline.sh), which executes all of the following shell scripts:
- The script [code/load_data.sh](https://github.com/team-one-ML/MLinPractice/blob/main/code/load_data.sh) downloads the raw csv files containing the tweets and their metadata. They are stored in the folder `data/raw/` (which will be created if it does not yet exist).
- The script [code/preprocessing.sh](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing.sh) executes all necessary preprocessing steps, including a creation of labels and splitting the data set.
- The script [code/feature_extraction.sh](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction.sh) takes care of feature extraction.
- The script [code/dimensionality_reduction.sh](https://github.com/team-one-ML/MLinPractice/blob/main/code/dimensionality_reduction.sh) takes care of dimensionality reduction.
- The script [code/classification.sh](https://github.com/team-one-ML/MLinPractice/blob/main/code/classification.sh) takes care of training and evaluating a classifier.
- The script [code/application.sh](https://github.com/team-one-ML/MLinPractice/blob/main/code/application.sh) launches the application example.

---
<br>
<br>

## Preprocessing

All python scripts and classes for the preprocessing of the input data can be found in [code/preprocessing/](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/).
<br>
<br>

### Creating Labels

The script [create_labels.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/create_labels.py) assigns labels to the raw data points based on a threshold on a linear combination of the number of likes and retweets. It is executed as follows:

```python -m code.preprocessing.create_labels path/to/input_dir path/to/output.csv```

Here, `input_dir` is the directory containing the original raw csv files, while `output.csv` is the single csv file where the output will be written.

The script takes the following optional parameters:
- `-l` or `--likes_weight` determines the relative weight of the number of likes a tweet has received. Defaults to 1.
- `-r` or `--retweet_weight` determines the relative weight of the number of retweets a tweet has received. Defaults to 1.
- `-t` or `--threshold` determines the threshold a data point needs to surpass in order to count as a "viral" tweet. Defaults to 50.
<br>
<br>

### Classical Preprocessing

The script [run_preprocessing.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/run_preprocessing.py) is used to run various preprocessing steps on the raw data, producing additional columns in the csv file. It is executed as follows:

```python -m code.preprocessing.run_preprocessing path/to/input.csv path/to/output.csv```

Here, `input.csv` is a csv file (ideally the output of [create_labels.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/create_labels.py)), while `output.csv` is the csv file where the output will be written.

If you wish to remove tweets that are not english:
- `--prune-lang`: Drop rows of a language other than english 

The preprocessing steps to take can be configured with the `--pipeline` flag:

```
--pipeline <column> <preprocessor1> <preprocessor2> ...
```

Available preprocessors are:
- `remove_urls`: Remove all URLs from the tweet using [regex_replacer.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/regex_replacer.py) and return it as new column with suffix "_urls_removed" (has to be specified before `--punctuation` and `--tokenize`).
- `lowercase`: Lowercase the current column and return it as new column with suffix "_lowercased" containing the lowercased text (See [lowercase.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/lowercase.py) for more details).
- `expand`: Expand contractions in the current column to their long form and return it as new column with suffix "_expanded" (See [expand.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/expand.py) for more details).
- `punctuation`: Remove all punctuation from the current column and return it as new column with suffix "_no_punctuation" (See [punctuation_remover.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/punctuation_remover.py) for more details).
- `standardize`: Standardize UK and US spelling variations to US spelling and return it as new column with suffix "_standardized" (See [standardize.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/standardize.py) for more details).
- `tokenizer`: Tokenize the current column and return the tokenized tweet as new column with suffix "_tokenized" (See [tokenizer.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/standardize.py) for more details).
- `numbers`: Replace numbers with a generic number token and return it as new column with suffix "_numbers_replaced" (See [regex_replacer.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/regex_replacer.py) for more details).
- `lemmatize`: Replace words in the current column with their lemmas and return them as new column with suffix "_lemmatized" (See [lemmatizer.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/lemmatizer.py) for more details).
- `remove_stopwords`: Remove stopwords from the current column and return it as new column with suffix "_removed_stopwords" (See [stopword_remover.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/stopword_remover.py) for more details).

Moreover, the script accepts the following optional parameters:
- `-e` or `--export` gives the path to a pickle file where an sklearn pipeline of the different preprocessing steps will be stored for later usage.
- `--fast` only runs preprocessors on a small subset of the dataset. Specify the subset size as an integer argument.
<br>
<br>

### Splitting the Data Set

The script [split_data.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/split_data.py) splits the overall preprocessed data into training, validation, and test set. It can be invoked as follows:

```python -m code.preprocessing.split_data path/to/input.csv path/to/output_dir```

Here, `input.csv` is the input csv file to split (containing a column "label" with the label information, i.e., [create_labels.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/create_labels.py) needs to be run beforehand) and `output_dir` is the directory where three individual csv files `training.csv`, `validation.csv`, and `test.csv` will be stored.
The script takes the following optional parameters:

- `-t` or `--test_size` determines the relative size of the test set and defaults to 0.2 (i.e., 20 % of the data).
- `-v` or `--validation_size` determines the relative size of the validation set and defaults to 0.2 (i.e., 20 % of the data).
- `-s` or `--seed` determines the seed for intializing the random number generator used for creating the randomized split. Using the same seed across multiple runs ensures that the same split is generated. If no seed is set, the current system time will be used.

---
<br>
<br>

## Feature Extraction

All python scripts and classes for feature extraction can be found in [feature_extraction.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/).

The script [extract_features.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/extract_features.py) takes care of the overall feature extraction process and can be invoked as follows:

```python -m code.feature_extraction.extract_features path/to/input.csv path/to/output.pickle```

Here, `input.csv` is the respective training, validation, or test set file created by [split_data.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/split_data.py). The file `output.pickle` will be used to store the results of the feature extraction process, namely a dictionary with the following entries:
- `"features"`: a numpy array with the raw feature values (rows are training examples, colums are features)
- `"feature_names"`: a list of feature names for the columns of the numpy array
- `"labels"`: a numpy array containing the target labels for the feature vectors (rows are training examples, only column is the label)

The features to be extracted can be configured with the following optional parameters:

- `-c` or `--char_length`: Count the number of characters in the "tweet" column of the data frame (See [character_length.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/character_length.py) for more details).
- `-n` or `--ner`: Recognize all named entities in the tweet (See [ner.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/ner.py) for more details).
- `-w`or `--weekday`: Extract the day of the week (0-6) that the tweet was posted using [onehot_time_extraction.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/onehot_time_extraction.py).
- `-b`or `--month`: Extract the month (1-12) that the tweet was posted using [onehot_time_extraction.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/onehot_time_extraction.py).
- `--season`: Extract the season (winter, spring, summer, fall) that the tweet was posted using [onehot_time_extraction.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/onehot_time_extraction.py).
- `-d`or `--daytime`: Extract the time of day (night, morning, afternoon, evening) that the tweet was posted using [onehot_time_extraction.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/onehot_time_extraction.py).
- `-t` or `--tfidf`: Calculate tf-idf for the top words in the dataset (See [tf_idf.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/tf_idf.py) for more details).
- `--sentiment`: Analyse the sentiment of the tweet in terms of negativity, positivity, neutrality and overall sentiment (See [sentiment.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/sentiment.py) for more details).
- `--threads`: Detect tweets that are part of a thread (See [threads.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/threads.py) for more details).

The following features retrieve a boolean using [count_boolean.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/count_boolean.py), if tweet has one of the following attributes:

- `--hashtag_count`: Count the number of hashtags used in the tweet.
- `--mentions_count`: Count the number of mentions in the tweet.
- `--reply_to_count`: Count the number of accounts the tweet is in reply to.
- `--photos_count`: Count the number of photos used in the tweet.
- `--url_count`: Count the number of urls used in the tweet.
- `--video_binary`: Convert if the tweet contains a video into binary boolean values.
- `--retweet_binary`: Convert if the tweet is a retweet into binary boolean values.

If you wish to retrieve the absolute counts of the above attributes, add the optional flag:

- `--item_count`: Specify, to retrieve absolute counts of the above attributes except for `--video_binary` and `--retweet_binary`.

Moreover, the script support importing and exporting fitted feature extractors with the following optional arguments:
- `-i` or `--import_file`: Load a configured and fitted feature extraction from the given pickle file. Ignore all parameters that configure the features to extract.
- `-e` or `--export_file`: Export the configured and fitted feature extraction into the given pickle file.

---
<br>
<br>

## Dimensionality Reduction

All python scripts and classes for dimensionality reduction can be found in [code/dimensionality_reduction/](https://github.com/team-one-ML/MLinPractice/blob/main/code/dimensionality_reduction/).

The script [reduce_dimensionality.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/dimensionality_reduction/reduce_dimensionality.py) takes care of the overall dimensionality reduction procedure and can be invoked as follows:

```python -m code.dimensionality_reduction.reduce_dimensionality path/to/input.pickle path/to/output.pickle```

Here, `input.pickle` is the respective training, validation, or test set file created by [extract_features.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/extract_features.py). 
The file `output.pickle` will be used to store the results of the dimensionality reduction process, containing `"features"` (which are the selected/projected ones) and `"labels"` (same as in the input file).

The dimensionality reduction method to be applied can be configured with the following optional parameters:
- `-m` or `--mutual_information`: Select the `k` best features (where `k` is given as argument) with the Mutual Information criterion.
- `--tsvd`: Reduce dimensionality using truncated SVD (a fast variant of PCA).
- `-p` or `--pca`: Projects features into the `k` main dimensions of variation (where `k` is given as argument) with the Principle Component Analysis criterion.

Moreover, the script support importing and exporting fitted dimensionality reduction techniques with the following optional arguments:

- `-i` or `--import_file`: Load a configured and fitted dimensionality reduction technique from the given pickle file. Ignore all parameters that configure the dimensionality reduction technique.
- `-e` or `--export_file`: Export the configured and fitted dimensionality reduction technique into the given pickle file.

Finally, if the flag `--verbose` is set, the script outputs some additional information about the dimensionality reduction process.

---
<br>
<br>

## Classification

All python scripts and classes for classification can be found in [code/classification/](https://github.com/team-one-ML/MLinPractice/blob/main/code/classification/).

The script [run_classifier.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/classification/run_classifier.py) can be used to train and/or evaluate a given classifier. It can be executed as follows:

```python -m code.classification.run_classifier path/to/input.pickle```

Here, `input.pickle` is a pickle file of the respective data subset, produced by either `extract_features.py` or `reduce_dimensionality.py`. 

By default, this data is used to train a classifier, which is specified by one of the following optional arguments:

- `-m` or `--majority`: Majority vote baseline classifier that always predicts the majority class.
- `-q` or `--frequency`: Label-Frequency baseline classifier that predicts the class according to the ratio of true:false labels in the training set.
- `--svm`: Support Vector Machine classifier.
- `--knn`: K Nearest Neighbors classifier that predicts the class exhibited my the majority of an instance's k nearest neighbors in the feature space.
- `--mlp`: Multi Layered Perceptron classifier that takes hyperparameters as arguments (If none are entered, the values which performed best in our tests are chosen)<br>
Arguments: hidden_layer_sizes (int), activation (str), solver (str), max_fun (int)

The classifier is then evaluated, using the evaluation metrics as specified through the following optional arguments:
- `-a`or `--accuracy`: Classification accurracy (i.e., percentage of correctly classified examples).
- `--mcc`: Mathew's Correlation Coefficient (a coefficient of +1 represents a perfect prediction, 0 no better than random prediction and −1 indicates total disagreement between prediction and observation).
- `-n` or `--informedness`: Youden’s J statistic (0 means the amount of correct classifications is equal to the amount of incorrect classifications, 1 means there are no incorrect classifications).
- `-k`or `--kappa`: Cohen's Kappa (Includes the probability of correct classification by chance in the evaluation).
- `-b` or `--balanced_accuracy`: Balanced accuracy (arithmetic mean between true positive rate and true negative rate).
- `-f` or `--f1_score`: F1 Score (weighted average of precision and recall).

Moreover, the script support importing and exporting trained classifiers with the following optional arguments:

- `-i` or `--import_file`: Load a trained classifier from the given pickle file. Ignore all parameters that configure the classifier to use and don't retrain the classifier.
- `-e` or `--export_file`: Export the trained classifier into the given pickle file.

Finally, the optional argument `-s` or `--seed` determines the seed for intializing the random number generator (which may be important for some classifiers). 
Using the same seed across multiple runs ensures reproducibility of the results. If no seed is set, the current system time will be used.

---
<br>
<br>

## Application

All python code for the application demo can be found in [code/application/](https://github.com/team-one-ML/MLinPractice/blob/main/code/application/).

The script [application.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/application/application.py) provides a simple command line interface, where the user is asked to type in their prospective tweet, which is then analyzed using the trained ML pipeline.
The script can be invoked as follows:

```python -m code.application.application path/to/preprocessing.pickle path/to/feature_extraction.pickle path/to/dimensionality_reduction.pickle path/to/classifier.pickle```

The four pickle files correspond to the exported versions for the different pipeline steps as created by [run_preprocessing.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/run_preprocessing.py), [extract_features.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/extract_features.py), [reduce_dimensionality.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/dimensionality_reduction/reduce_dimensionality.py), and [run_classifier.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/classification/run_classifier.py), respectively, with the `-e` option.