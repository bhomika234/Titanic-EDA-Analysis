Titanic Dataset Exploratory Data Analysis & Cleaning
This project focuses on performing Exploratory Data Analysis (EDA) and data cleaning on the Titanic dataset using Python to uncover insights about passenger survival factors.

🛠️ Tools & Libraries
Python

Pandas

NumPy

Matplotlib

Seaborn

📋 Project Steps & Workflow
1. Data Loading & Inspection
Loaded the Titanic dataset into a Pandas DataFrame.

Checked the dataset dimensions (891 rows and 12 columns).

Inspected data types, non-null counts, and verified zero duplicate rows in the dataset.

2. Data Cleaning & Handling Missing Values
Missing Values Analysis: Identified columns with missing data, specifically Age (177 missing values), Cabin (687 missing values), and Embarked (2 missing values).

Imputation & Dropping:

Filled missing Age values using the median age.

Filled missing Embarked values using the mode.

Dropped the Cabin column due to an excessive number of missing records.

3. Exploratory Data Analysis (EDA) & Visualizations
Overall Survival Rate: Analyzed the proportion of survivors versus non-survivors (approx. 38.38% overall survival rate).

Gender-wise Analysis: Visualized survival rates grouped by gender, showing that female passengers had a significantly higher survival rate than males.

Passenger Class (Pclass) Analysis: Explored how socio-economic status (1st, 2nd, and 3rd class) impacted passenger survival chances.

Statistical Summary: Calculated descriptive statistics (mean, median, standard deviation) for numerical features like Age and Fare.

🚀 How to Run the Code
Clone the repository or download the Jupyter Notebook file.

Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.

Ensure you have the required libraries installed (pip install pandas numpy matplotlib seaborn).

Run the cells sequentially to view the data analysis and charts.