# AI-Based-Wine-Recommendation-System

This projects recommends suitable wine recommendation according to user's request. The project leverages large language models (LLMs) to generate meaningful matches between different types of wines.

<h2>FEATURES</h2>

- Wine Reviews dataset from Kaggle was used and data analysing, processing and text cleaning operations were executed.
- Processes user sentences to provide wine recommendations
- Containts chroma vector database. Name of wine and its description join on page content part and other features such as price, country etc. join metadata.
- In order to obtain answers closer to human language google/flan-t5-large model was used.

<h2>STRUCTURE</h2>

WineRecommendation/
│
├── main.py                User enters a query and an answer is generated
├── MissingValues.py       Display number of missing values and their ratios for each columns
├── wine_model.py          Answer generation
├── text_cleaner.py        Cleaning text columns
├── wine_nlp.py            Exploratory data analysis and embedding data to vector database
├── requirements.txt     
└── README.md            

<h2>SET THE PROJECT</h2>
git clone https://github.com/ozgurumutpirinc/AI-Based-Wine-Recommendation-System.git
cd AI-Based-Wine-Recommendation-System

pip install -r requirements.txt
