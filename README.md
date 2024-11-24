# Food-Hazard-Detection-Challenge

This Repository conatins coode related to the competition hosted on [Competition_Link](https://food-hazard-detection-semeval-2025.github.io/)

## Firstly it will mentioned teh data exploration and preprocess part.

### Data
- `data/incidents_train.csv` : Contains the initial dataframe for training. 
- `data/incidents.csv` : Containes unlabeled data of teh competition 
- `data_augmented__nlp_incidents_train.csv`: transformed ddataframe with data augmented and basic nlp preprocess. This will be used for the training process as input in the jupyter  `training_process.ipynb`

### Data Exporation / Preprocess 
- `data_exploration.ipynb`: Explore columns , distributions of Data. Basic Data Exploration. 
- `translate.ipynb`: Apply Google Translate to not English sentences (finally useless due to extremely low valuw symantic)
- `data-preprocess.ipynb`: Apply data augmentic strategy to craete synthetic data with usage of synonyms and randomeness. Furtherre. applied basic NLP preproces (tokenization, stemmig, removeal numbers, punctuations etc.)

| Input       | Jupyter Notebook      | Output     |
|----------------|----------------|----------------|
| data/incidents_train.csv | data_exploration.ipynb | - |
| data/incidents_train.csv |translate.impynb| - |
| data/incidents_train.csv | data-preprocess.ipynb| data/data_augmented__nlp_incidents_train.csv |

## Secondly it will mentioned the Benchmark Analysis and training process with usage of traditional and advanced ML algorythms

| Input       | Jupyter Notebook      | Output     |
|----------------|----------------|----------------|
| data/data_augmented__nlp_incidents_train.csv  | data_exploration.ipynb | reports/ |

Note: The folder `reports` cotanins the analytical classifcagion report per label and the overall valculated maro avarage f1 score per label (hazard, product, hazard-cateory, product-category).
