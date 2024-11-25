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
| data/data_augmented__nlp_incidents_train.csv  | training_process.ipynb | reports/ |

Note: The folder `reports` cotanins the analytical classifcagion report per label and the overall valculated maro avarage f1 score per label (hazard, product, hazard-cateory, product-category).

### Table with sub tasks custom evaluation for all models for all input s 
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

- `From the table above, it is obcious that the "best" scores in both subtask 1 and sub task 2 is X-Boost with input the text`.
- Furthermore, it seems that across all scores per title and text, text (te) scores are equal or higher than title (ti).