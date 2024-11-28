# Food-Hazard-Detection-Challenge

This Repository conatins coode related to the competition hosted on [Competition_Link](https://food-hazard-detection-semeval-2025.github.io/)

### Python Packages Requirements && Project Instalation 
- Clone the Repository via running to the terminal and go to the main folder of the Project: 
```bash 
    git clone git@github.com:e-panourgia/Food-Hazard-Detection-Challenge.git
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
- `data_exploration.ipynb`: Performs basic data exploration, including column analysis and distribution visualization.  
- `translate.ipynb`: Utilizes Google Translate to process non-English sentences (ultimately deemed ineffective due to limited semantic value).  
- `data-preprocess-augmented.ipynb`: Implements data augmentation strategies to generate synthetic data using synonyms and randomness. Additionally, applies basic NLP preprocessing (e.g., tokenization, stemming, removal of numbers, punctuation, etc.).  
- `data-preprocess-initial.ipynb`: Applies basic NLP preprocessing (e.g., tokenization, stemming, removal of numbers, punctuation, etc.) and eliminates duplicate rows from the dataset.  


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
| data/data_augmented__nlp_incidents_train.csv  | augmented_training_process.ipynb | reports/ , augmented_sumission/|
| data/data_nlp_incidents_train.csv  | initila_training_process.ipynb | reports/ initial_submission/|

### Table based on `Augmented Data` for SubTask 1 , SubTask 2
- Guide to Reading the Table Below (Rows):
    - To help interpret the table, the following abbreviations are provided:  
        - `LR`: Refers to **Logistic Regression**.  
        - `RF`: Refers to **Random Forest**.  
        - `XB`: Refers to **XGBoost**.  
        - `ti`: Represents **title**.  
        - `te`: Represents **text**. 

- Guide to Reading the Table Below (Columns):
    - To help interpret the table, the following abbreviations are provided:  
        - First column has the custom competition score for the sub task (`ST1`) that is Text classification for food hazard prediction, predicting the type of hazard and product.
        - Second column has the custom competition score for the sub task (`ST2`) that is Food hazard and product “vector” detection, predicting the exact hazard and product.

- **Note:** `Random` and `Majority` classifiers are used as baselines. These are not followed by `ti` or `te` as they are independent of input type.  


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

- With yellow color we hve the classifiers outperforms regarding the custom evalution scores (for st1, st2). 
- In both cases of input ("title" and "text") X-Bosst wins. Overall, the classifiers has better performance in the "text" input. Furtehrmore, all classifiers outperform the baselines meaning that they predict better than randomness and the frequenct value (mode).

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

### Important Comment / Mistake Affected the Overall Performance of the  Assignment

- Unfortunately, we must acknowledge that during the process, a significant issue arose, leading to a lower-than-expected score. We sincerely apologize for this oversight, which we only realized after several days. As a result, we were unable to rerun the entire process in the correct sequence or properly adjust the parameters of the TF-IDF and models.  
- More specifically, instead of first conducting the benchmark analysis on the initial dataset and then on rerunnning the tuned models stemming from the initial data to the NLP-preprocessed dataset (which we intuitively assumed would yield slightly better performance), and then on the augmented datset, too, we mistakenly began with the augmented dataset. This dataset was later identified as flawed due to "fake" synthetic data. Unfortunately, we only realized this issue after several days, when we submitted the predictions of the unlabeleld dataset to the competition platform and we received a score quite close to 0. 
- The correct pipeline logic would be to run the raw data, then the nlp preprocessed data and then the augmented data all of them on the same tuned models stemming from the raw data and compare the results.
- Due to time constraints, we mistakenly assumed the initial dataset was the augmented one. The manual process of tuning the models based on the correct raw data was time-intensive, and we lacked the time to complete it.  
    - This explains why the individual evaluation reports show low scores for the simple nlp preprocess data.  
- We sincerely apologize for this critical oversight, which significantly impacted overall performance.  
- Nevertheless, we have made an effort to thoroughly explain the reasoning behind our decision-making process.  

### Decision making Strategy during BenchMark Analysis / Methodology In short 
- During teh scope of benchmark  analysis we decied to analyse `Logistic Regression` (based on linearity), `Random Forest` (non-linear relations) and `X-Boost` (gradiet-boosting algorythm). 
- Firlty,we focus on "title" as input. 
    - We applied hyperparameter tuning manually (as a systematic way using either greed search or a modern MLOp tool like `Optuna` would required much time bearing in mind tha we have 3 models , running it 2 times for each input "title" and "text" (and applying cross validation), so we tried to manually tune the algorythms based on teh following diamensions : 
        - a) improve the custom evalution metric of the competition for both sub task 1 and sub task 2. In other words, we used common evalution metric acroiss all algorythms in order to be more "fair" the comparison. 
        - b) adapt the hyperparamters based on teh nature of teh problem e.g. our problem is `multiclass` so we reflected it in logistic regression with the parameter `multinomial` and in x-boost with the parameter `multi:softmax` and for the evaluation with the parameter `mlogloss`.
        - c) adapt the parameetr in specific scope as we had memory time limitations. e.g.  for tf-idf vectorization process we had to limit max_features parameter, or for logistic regression with hparameter  `ovr` (One-Vs-All) would have extreemly high computational complexity.
        - d) For having more "fair" comparison we used the same vectorization technique tf-idf. Manually we tries to adapth the max_feature (trying 2000 and 5000 and holding the best one that offers better custome scores per sub task).

- Note 1: Despite the fact that we our evalution based on the `custom competition metrics for sub task 1 and sub task 2`, we printed the classification evalution reports, too. These offer us the potential to have an overview of the average macro f score beingpart of the custom mentric evalution and in depth undestanding about the classes in which the algorythm has low f1 score (low precision and recall), this insight could be leveraged to group these categories to a new category e.g. "Other" in teh scope of trying to improve the performance of teh model. Gnerally, we want high f1 sores per all classes but this is unvoidable due to the classes having low number of instances. 

- Note 2: We applied systematic hyperparametr tuning technoques only for the best algorythm after benchmark analysis. Of course, this is more prfessional approach but due to time and memory limitations we could not apply it in teh scope of benchmark analysis. 

- Note 3: Regardign the diffrent input "title" and "text" we only manually tried diffrent hyperparameter in tf-idf for the parameter `max_feature` but we did not observe better scores in the custom evaluation scores of teh competition for teh sub tasks. So, we simply re-run the same algorythms via changing teh input. 

### Future Work 
A) Re-run the whole proces with the correct order meaning : `nlp preprocess data` and tune the models (if I had time I would tuned oth models and tf-idf hyperparameters systematically using MLOp optuna or greed search)

B) For the hyperparamters defined in A) Re-run via changing only the input dataframe having the **raw** data 

C) Re-run via changing only the input dataframe having the **augmented** data 

D) Regarding achieving betetr scores, we could try tf-idf (more max features) in combination with PCA (reduce diamensionality of input space stemming from tf-idf), more hyperparameter tuning, regrouping classes with low f1 score in classificatio nreport, and maybe oversampling methos like SMOTE insteadd of our own augmented data.  

### Lifeboard Comments 
- Due to the fake nature of our augmented data having a score extrleme close to 0 (0.00..) we hold as best scores the scores stemmign from the nlp preprocessed data, and having as algorythm `X-boost` with input `text`.
    
    - Scores Competition : 
    - Sub Task 1 : `0.0710` (stemming from competition score data 27 November 2024) 
    - Sub Task 2 : `0.0057` (stemming from competition score data 27 November 2024)
