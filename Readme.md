### 6th Semester Project Demo

### Project Name:
#### Wine Quality Prediction using Machine Learning Algorithms
### Team Members
1. [Sayak Ghosh]() - 20101106010
2. [Rhitam Banerjee]() - 20101106003
3. [Avijit Sen]() - 20101106040


### Project Description

The project is about predicting the quality of wine using machine learning algorithms. The dataset used for this project is the Wine Quality Dataset from Kaggle. The dataset contains 12 columns and 6497 rows. The columns are:
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality


### Project Mentors
1. [Shrayasi Dutta]()


### Run Instructions

1. Clone the repository

2. Create a virtual environment 
### For Windows
```bash
python -m venv venv
```
### For Linux
```bash
python3 -m venv venv
```
3. Activate the virtual environment
### For Windows
```bash
venv\Scripts\activate
```
### For Linux
```bash
source venv/bin/activate
```
4. Install the requirements
```bash
pip install -r requirements.txt
```
5. Run the app
```bash
python app.py
```

### How to make a pull request

1. Fork the repository

2. Swwitch to the development branch
```bash
git checkout dev
```
3. Create a new branch
```bash
git checkout -b <branch-name>
```
4. Make changes and commit
```bash
git add .
git commit -m "commit message"
```
5. Push the changes
```bash
git push origin <branch-name>
```

### Code Workflow

1. Preprocessing the data
2. Splitting the data into train and test sets
3. Training the model
4. Testing the model
5. Saving the model
6. Loading the model
7. Predicting the quality of wine
8. Displaying the results


### Project Report
[Project Report]( ./projectreport.md)



