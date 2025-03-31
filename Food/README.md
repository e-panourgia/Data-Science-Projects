# Food-Hazard-Detection-Challenge

This Repository conatins coode related to the competition hosted on [Competition_Link](https://food-hazard-detection-semeval-2025.github.io/)

### Python Packages Requirements && Project Instalation 
- Clone the Repository via running to the terminal and go to the main folder of the Project: 
```bash 
    git clone git@github.com:e-panourgia/Data-Science-Projects.git
    cd Food-Hazard-Detection-Challenge
```
- This project depends on several Python libraries for data manipulation, machine learning, natural language processing, visualization, and more. The required libraries are listed in the `requirements.txt` file.
- Create virtual environemt and install to it the libaries of `requirements.txt`
    - Adapt Jupyter Notepook to "see" the virtual enviroment
- Commands to Run:
```bash
# Step 1: Create virtual environment
python3 -m venv venv

# Step 2: Activate virtual environment
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Set up Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
``` 
- Alternatively, dowload them with the comannd !pip install NAMES_OF_NEEDED_LIBRARIES 
    ```bash 
    !pip install numpy pandas scikit-learn xgboost joblib nltk imblearn deep-translator langid matplotlib seaborn
    ```
- Usage of `Python 3.10.13` 

## Data Exploartion / Preprocess 

### Data 
- `data/incidents_train.csv`: The initial dataset used for training purposes.  
- `data/incidents.csv`: Contains unlabeled competition data with basic NLP preprocessing applied to the `title` and `text` columns (e.g., lowercasing, stemming, etc.).  
- `data/data_augmented__nlp_incidents_train.csv`: A transformed dataset with augmented data and basic NLP preprocessing, used as input for the training process in `augmented_training_process.ipynb`.  
- `data/data_nlp_incidents_train.csv`: A transformed dataset with duplicate rows removed and basic NLP preprocessing applied. This dataset **does not** include synthetic data generated through techniques like synonym replacement or random word removal.  
- `data/incidents_unlabeled_without_nlp_preprocess.csv`: The unlabeled competition data without any NLP preprocessing applied to the `title` and `text` columns.  

### Data Exporation / Preprocess 
- `data_exploration.ipynb`: Performs basic data exploration.  
- `translate.ipynb`: Utilizes Google Translate to process non-English sentences (ultimately deemed ineffective due to limited semantic value).  
- `data-preprocess-augmented.ipynb`: Implements data augmentation strategies to generate synthetic data using synonyms and randomness. Additionally, applies basic NLP preprocessing (e.g., tokenization, stemming, removal of numbers, punctuation, etc.).  
- `data-preprocess-initial.ipynb`: Applies basic NLP preprocessing (e.g., tokenization, stemming, removal of numbers, punctuation, etc.) and eliminates duplicate rows from the dataset.  In addition, applied basic NLP preprocess to the unlabelled columns "title" and "text" of the competition.


| Input       | Jupyter Notebook      | Output     |
|----------------|----------------|----------------|
| data/incidents_train.csv | data_exploration.ipynb | - |
| data/incidents_train.csv |translate.impynb| - |
| data/incidents_train.csv | data-preprocess-augmented.ipynb | data_augmented__nlp_incidents_train.csv |
| data/incidents_train.csv, incidents_unlabeled_without_nlp_preprocess.csv | data-preprocess-initial-data.ipynb| data/data_nlp_incidents_train.csv,  incidents.csv|

## Benchmark Analysis 
- `augmented_training_process.ipynb`: Contains the training process based on the augmented dataset, which includes synthetic data.  
- `initial_training_process.ipynb`: Contains the training process based on the initial dataset with basic NLP preprocessing.  
- `augmented_submission/`: Directory containing predictions for the competition's unlabeled data, generated using the best models trained on the augmented dataset.  
- `initial_submission/`: Directory containing predictions for the competition's unlabeled data, generated using the best models trained on the initial dataset with basic NLP preprocessing.  
- `reports_augmented/`: Directory hosting classification evaluation reports generated from the training process in `augmented_training_process.ipynb`.  
- `reports_initial/`: Directory hosting classification evaluation reports generated from the training process in `initial_training_process.ipynb`.  


| Input       | Jupyter Notebook      | Output     |
|----------------|----------------|----------------|
| data/data_augmented__nlp_incidents_train.csv  | augmented_training_process.ipynb | reports_augmented/ , augmented_sumission/|
| data/data_nlp_incidents_train.csv  | initila_training_process.ipynb | reports/_inital initial_submission/|

### Table based on `Augmented Data` for SubTask 1 , SubTask 2
- Guide to Reading the Table Below (Rows):
    - To help interpret the table, the following information are provided: 
        - First two rows represent the baseline models.
        - The remaning ones the tarditional and advances models for both "Title" and "Text" as input.
- Guide to Reading the Table Below (Columns):
    - To help interpret the table, the following information are provided:  
        - First column has the custom competition score for the sub task (`ST1`) that is Text classification for food hazard prediction, predicting the type of hazard and product.
        - Second column has the custom competition score for the sub task (`ST2`) that is Food hazard and product “vector” detection, predicting the exact hazard and product.

- **Note:** `Random` and `Majority` classifiers are used as baselines. These are not followed by `title` or `text` as they are independent of input type.  


|     | `Sub Task 1`     | `Sub Task 2`    |
|--------------|--------------|--------------|
| `Random Baseline`| 0.057 | 0.003 |
| `Majority Baseline`| 0.031 | 0.001 |
| `LogisticRegression Title` | 0.690 | 0.425 |
| `Random Forest Title` | `0.760` | `0.721` |
| `X-Boost Title`| 0.741 | 0.647 |
| `LogisticRegression Text` | 0.695 | 0.427 |
| `Random Forest Tetx` | 0.784 | 0.758 |
| `X-Boost Text`| `0.814` | `0.759` |

- With yellow color we have the classifiers outperforms regarding the custom evalution scores (for st1, st2). 
- In both cases of input ("title" and "text") X-Bosst wins. Overall, the classifiers has better performance in the "text" input. Furtehrmore, all classifiers outperform the baselines meaning that they predict better than randomness and the frequenct value (mode).

- We also applied systematic hyperparameter tuning for XGBoost with the input "text," as it demonstrated better potential to perform well in the competition.  
    - Using `grid search` with 3-fold cross-validation (`K=3`), we focused on tuning the `max_depth` hyperparameter.  
        - Due to resource limitations, we were unable to explore additional hyperparameters.  
    - Based on the results, a `max_depth` of 8 consistently outperformed values of 6 and 7 across all iterations. This approach provided a more robust decision than manual tuning, as it relied on automated, cross-validated results.  
    - While the competition scores improved slightly after tuning, the table below highlights the best scores achieved by our model, comparing results from the benchmark analysis and after systematic tuning.

|     | `Sub Task 1`     | `Sub Task 2`    |
|--------------|--------------|--------------|
| `X-Boost (max_depth=6) Text`| 0.814 | 0.759 |
| `tuned X-Boost (max_depth=8) Text`| 0.815 | 0.762 |

### Table based on `Initial Data` for SubTask 1 and SubTask 2


|     | `Sub Task 1`     | `Sub Task 2`    |
|--------------|--------------|--------------|
| `Random Baseline`| 0.051 | 0.002 |
| `Majority Baseline`| 0.039 | 0.002 |
| `LogisticRegression Title` | 0.39 | 0.13 |
| `Random Forest Title` | `0.50` | `0.32` |
| `X-Boost Title`| `0.54` | `0.31` |
| `LogisticRegression Text` | 0.36 | 0.11 |
| `Random Forest Text` | 0.42 | 0.26 |
| `X-Boost Text`| `0.51`| `0.33` |

- With yellow color we hve the classifiers outperforms regarding the custom evalution scores (for st1, st2).
- based on the initial data we have three classifers with close custom compettion scores for both "titile" and "text" input. Again X-Boost has key dominance. Furtehrmore, all classifiers outperform the baselines meaning that they predict better than randomness and the frequenct value (mode).
- We attempted systematic hyperparameter tuning for XGBoost with the "text" input by randomly selecting one of the best-performing benchmark models (due to time constraints). Using Optuna, a modern MLOps tool for hyperparameter optimization, we aimed to tune both the TF-IDF parameter (`max_features` ranging from 1000 to 5000 in steps of 1000) and XGBoost parameters (`max_depth` ranging from 8 to 10 and `n_estimators` ranging from 50 to 100 in steps of 50). However, the process proved to be time-consuming, and due to limited time, we had to terminate the tuning operation before it could complete.


### Decision making Strategy during BenchMark Analysis / Methodology In short 
- As part of the scope for benchmark analysis, we have selected three algorithms to evaluate their performance on data classification tasks:

1. **Logistic Regression**  
   - Chosen for its reliance on linearity, making it suitable for identifying linear relationships in the data.

2. **Random Forest**  
   - Selected to capture non-linear relationships due to its ensemble-based, decision-tree methodology.

3. **XGBoost**  
   - A gradient-boosting algorithm known for its advanced and highly efficient operations.

These algorithms were chosen to explore different aspects of data classification, such as linearity versus non-linearity, while leveraging varying levels of algorithmic complexity from simpler models (e.g., Logistic Regression) to more sophisticated approaches (e.g., XGBoost).

- Firlty,we focus on "title" as input. 
    - We applied hyperparameter tuning manually, as using systematic methods such as grid search or modern MLOps tools like `Optuna` would have required significant time. This consideration was particularly relevant given that we were working with three models, running the tuning process twice for each input type ("title" and "text") and applying cross-validation.

    - To achieve effective tuning within these constraints, we focused on manually adjusting the algorithms based on the following dimensions:

        - a) We improved the custom evaluation metrics defined for the competition in both Subtask 1 and Subtask 2. Specifically, we applied a common evaluation metric across all algorithms to ensure a more consistent and fair basis for comparison.
        - b) We adapted the hyperparameters to reflect the nature of the `multiclass` problem: using `multinomial` for Logistic Regression, `multi:softmax` for XGBoost, and `mlogloss` as the evaluation metric. For Random forest ,we did not tuned a related parameter, as  Random Forest inherently supports multiclass classification without requiring specific parameter adjustments. Its tree-based structure can naturally handle multiple classes, making it unnecessary to adapt hyperparameters specifically for this purpose.
        - c) We adapted parameters within a specific scope due to memory and time limitations. For example, during the TF-IDF vectorization process, we restricted the `max_features` parameter to reduce dimensionality. Similarly, in Logistic Regression, we avoided using the `ovr` (One-Vs-All) parameter due to its extremely high computational complexity.
        - d) To ensure a more "fair" comparison, we used the same vectorization technique, TF-IDF, across all models. We manually adjusted the `max_features` parameter by experimenting with values (e.g., 2000 and 5000) and selected the one that achieved the best custom scores for each subtask.

- **Note 1:** Regarding the different inputs, "title" and "text," we manually experimented with the `max_features` parameter in TF-IDF during preprocessing (trying max_features = 2000 and max_features = 5000). However, we did not observe any significant improvements in the custom evaluation scores for the competition's subtasks. As a result, we proceeded by running the same algorithms while simply changing the input type.


- Note 1: Despite the fact that we our evalution based on the `custom competition metrics for sub task 1 and sub task 2`, we printed the classification evalution reports, too. These offer us the potential to have an overview of the average macro f score beingpart of the custom mentric evalution and in depth undestanding about the classes in which the algorythm has low f1 score (low precision and recall), this insight could be leveraged to group these categories to a new category e.g. "Other" in teh scope of trying to improve the performance of teh model. Gnerally, we want high f1 sores per all classes but this is unvoidable due to the classes having low number of instances. 

- **Note 2:** Systematic hyperparameter tuning techniques were applied only to the best-performing algorithm after the benchmark analysis. While this is a more professional and thorough approach, time and memory constraints prevented us from implementing it during the benchmark analysis phase.

- **Note 3:** In the current training Jupyter Notebooks, we retained only the parameters that achieved the best scores based on the competition's evaluation metrics.