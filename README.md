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
    - E.G. 
    ```bash 
    !pip install numpy pandas scikit-learn xgboost joblib nltk imblearn deep-translator langid matplotlib seaborn
    ```
- Usage of `Python 3.10.13` 

## Firstly it will mentioned teh data exploration and preprocess part.

### Data
- `data/incidents_train.csv` : Contains the initial dataframe for training. 
- `data/incidents.csv` : Containes unlabeled data of teh competition 
- `data_augmented__nlp_incidents_train.csv`: transformed dataframe with data augmented and basic nlp preprocess. This will be used for the training process as input in the jupyter  `training_process.ipynb`
- `data_nlp_incidents_train.csv`: tarnsformed dataframe removel duplicated rows and applied basic nlp preprocess (this dataset **don't**) ontain synthetic data (based on synionyms and random removel of words).
- 

### Data Exporation / Preprocess 
- `data_exploration.ipynb`: Explore columns , distributions of Data. Basic Data Exploration. 
- `translate.ipynb`: Apply Google Translate to not English sentences (finally useless due to extremely low valuw symantic)
- `data-preprocess-augmented.ipynb`: Apply data augmentic strategy to craete synthetic data with usage of synonyms and randomeness. Furtherre. applied basic NLP preproces (tokenization, stemmig, removeal numbers, punctuations etc.)
- `data-preprocess-intiial.ipynb`: Apply basic NLP preproces (tokenization, stemmig, removeal numbers, punctuations etc.) and removal of duplicated rows.

| Input       | Jupyter Notebook      | Output     |
|----------------|----------------|----------------|
| data/incidents_train.csv | data-preprocess-augmented.ipynb | - |
| data/incidents_train.csv |translate.impynb| - |
| data/incidents_train.csv | data-preprocess.ipynb| data/data_augmented__nlp_incidents_train.csv |
| data/incidents_train.csv | data-preprocess-initial.ipynb | - |
| data/incidents_train.csv | data-preprocess.ipynb| data/data_nlp_incidents_train.csv |

## Benchmark Analysis and Training process with usage of traditional and advanced ML algorythms
- `augmented_training_process.ipynb` contains training based on the augmented dataset (with the syntetic data). 
- `initial_training_process.ipynb` contains training based on the initial dataset with basic nlp preprocess.
- the folders `reports` cotanins herarhically the classification reports per  text titles per ML Algorythm.
- the folders `augmented_sumission/` (based on augmented data) ans `initial_submission/` (based on initial data with basic NLP Preprocess) contains the predictions of our best models.


| Input       | Jupyter Notebook      | Output     |
|----------------|----------------|----------------|
| data/data_augmented__nlp_incidents_train.csv  | augmented_training_process.ipynb | reports/ , augmented_sumission/|
| data/data_nlp_incidents_train.csv  | initila_training_process.ipynb | reports/ initial_submission/|


Note: The folder `reports` cotanins the analytical classifcagion report per label and the overall valculated maro avarage f1 score per label (hazard, product, hazard-cateory, product-category).

**Key Notw** Because to impelement with desipline the benchmark analysis bearing in mind that I had to run two times all models for each dataset that is augmented dataset and initial dataset having the intial data with basic nlp preprocess only, the training jupyter `augmented_training_process` has more comments in comparison to `initila_training_process`, so read it firstly, as many things are similar in intiial data too. We made it to save time. 

### Table based on `Augmented Data` with sub tasks custom evaluation for all models for all input s 
- Notes for reading teh below tables 
    - Note 1: 
        - `LR` = (meaning) `Logistic Regresion`
        - `RF` = (meaning) `Random Forest`
        - `XB` = (menaing) `X-Boost`
        - ti = (meaning) title
        - te = (meaning) text 

    - For example, LD-ti = (meaning) Logistic Regression as model and title as input.  

    - Note 2: The SemEval-Task combines two sub-tasks:
        - (ST1) Text classification for food hazard prediction, predicting the type of hazard and product.
        - (ST2) Food hazard and product “vector” detection, predicting the exact hazard and product.

    - Note 3: 
        - During Benchmark Analysis we used TF-idf for all of the aforementioned models.

    - Note 4 : 
        - Random and Majority Classifier being our baselines, they dont have defined inpute (ti or te) as they are independent of X.


- `all sub taks` menaing sub task 1 and sub task 2 of the competition 
- `all models` meaning `Logistic Regresiion`, `Random Forest` amd `X-Boost`
- `all inputs` meaning `title` , and `text`


|       | **SubTask 1 Custom Score**    | **SubTask 2 Custom Score**     |
|---------------|---------------|---------------|
| **Random** | ~0.057| ~0.003 |
| **Majoriy** | ~0.0031 | ~0.001 |
| **LR-ti** | ~0.68 | ~0.42 |
| **RF-ti** | ~0.76 | ~0.72 |
| **XB-ti** | ~0.74 | ~0.64 |
| **LR-te** | ~0.69 | ~0.42 |
| **RF-te** | ~0.78 | ~0.75 |
| **XB-te** | ~`0.81`| ~`0.75` |

- Note: For `Augmented Data` the best  model is X-Boost with Text as input and with hyperparameter tuning (using Greed Search) only for the best models we defined taht for all sub classefiers per label the best `max_depth` is 8 instead of 6,7. During greed search , we used Cross validation k = 3. 

### Table based on `Initial Data only basic NLP Preprocess` with sub tasks custom evaluation for all models for all input s 

|       | **SubTask 1 Custom Score**    | **SubTask 2 Custom Score**     |
|---------------|---------------|---------------|
| **Random** | ~0.051| ~0.002 |
| **Majoriy** | ~ 0.039 | ~0.002 |
| **LR-ti** | ~0.439| ~0.121 |
| **RF-ti** | ~0.53 | ~0.28 |
| **XB-ti** | ~0.55 | ~0.27 |
| **LR-te** | ~0.41 | ~0.10 |
| **RF-te** | ~0.51 | ~0.24 |
| **XB-te** | ~`0.61`| ~`0.28` |

- Note for `Initial Data` the best model is again X-Boost for text input. 

- `From the table above, it is obcious that the "best" scores in both subtask 1 and sub task 2 is X-Boost with input the text`.
- Furthermore, it seems that across all scores per title and text, text (te) scores are equal or higher than title (ti).

### Limitations - Difiiculties 
- In the given time, we could not try Oversampling Methods (like `SMOTE`), as SMOTE in Combination with TF-Idf requires resources (time and memory). 
- We needed to restrict some paramaters empirically to run the algorythms in the limited time of the deadline. 
    - For example, for logistic regression at the beggining we tried to approach the training process of the paper [Paper](https://aclanthology.org/2024.findings-acl.459.pdf) where use one-vs-all classification, but in practice, locally, in combiantions with TF-idf where we have high number in `max_features` it requiremed much time to run locally, so we used multinomial logistic.
    - In other words, many parameters adapted for memory and time limitations after empirical trials. 

### Future Work 
- Run all models for the initial csv instead of the augmented one. 
- Apply Overasample methods like `SMOTE` 
- Try diamensionality reduction techniques like embeddings. 
- Run hyperparameter tuning for more parameters (but is it is a time consuming process).
- Cross Validation for more robostness in evaluation metrics.