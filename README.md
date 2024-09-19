# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM :
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program :
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ROSHINI S
RegisterNumber:  212223240142
*/


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv("Placement_Data.csv")

# Print the entire DataFrame
print("Placement Data:")
print(data)

# Print only the salary column (if it exists)
if 'salary' in data.columns:
    print("\nSalary Data:")
    print(data['salary'])
else:
    print("\n'Salary' column not found in DataFrame")

# Remove unnecessary columns (if any)
data1 = data.drop(["salary"], axis=1, errors='ignore')

# Check for missing values
print("\nMissing Values Check:")
print(data1.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows Check:")
print(data1.duplicated().sum())

# Print the cleaned data
print("\nCleaned Data:")
print(data1)

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
categorical_columns = ['workex', 'status', 'hsc_s']  # List of categorical columns to encode
for column in categorical_columns:
    if column in data1.columns:
        data1[column] = le.fit_transform(data1[column])
    else:
        print(f"'{column}' column not found in DataFrame")

# Prepare features and target
x = data1.drop('status', axis=1, errors='ignore')  # Features
y = data1['status']  # Target

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the model
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_report1)

# Print the y_pred array
print("\nY Prediction Array:")
print(y_pred)

```

## Output :
### Placement Data :
![image](https://github.com/user-attachments/assets/cff67165-cd36-49c9-96ba-284276f45b9e)

### Salary Data :
![image](https://github.com/user-attachments/assets/83b94de5-71d2-4ad8-8dbf-16a304f4e230)

### Checking the null() function :
![image](https://github.com/user-attachments/assets/b540a324-7896-41a0-8c86-f468c75b2a41)

### Data Duplicate :
![image](https://github.com/user-attachments/assets/609408fb-e50a-420d-a7d1-1fd55f1c23fd)

### Clean Data :
![image](https://github.com/user-attachments/assets/22f20d89-9223-4f7f-bd41-d048528d081b)

### Y-Prediction Array :
![image](https://github.com/user-attachments/assets/76b1807a-e0b1-497a-82a9-b1cb2d26387a)

### Missing Values Check :
![image](https://github.com/user-attachments/assets/e41434bc-d45d-4def-95d4-fcf514245ee7)

### Accuracy value :
![image](https://github.com/user-attachments/assets/849c1dec-93e4-4f51-8235-7c9744eeb9dc)

### Confusion array :
![image](https://github.com/user-attachments/assets/bcf0b6f7-20be-4590-b818-3ad35ebb3c39)

### Classification Report :
![image](https://github.com/user-attachments/assets/e800038a-3a15-4b55-a2f8-7f16d04ba40a)


## Result :
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
