### predicting-absenteeism
###### [in progress] :briefcase: Learning preprocessing and machine learning (logistic regression model) on easy dataset. 


Dataset: https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work 
*here you can download dataset and description to it* 

Always we starting with data **preprocessing**: group of operations that will convert raw data into a format that is easier to understand and useful for further processing and analysis. Also helps organize information in suitable and practical way. 
It takes the most of the time and it is crucial part of every analytical ask. 
While preprocessing we make raw dataset usable for machine learning algorithm. 


So let's start with 
### **preprocessing part** 

Pandas allows us to work with panel data - Allows handling data in tabular format. It also lets us to even read the .csv files. 

```
import pandas as pd 
rawdata = pd.read_csv(‘Absenteeism_at_work.csv')
rawdata
```
After loading a data always I’m exploring it manually. It helps to have some first predictions. Sometimes it helps find some errors - even like importing wrong file :sweat_smile: and let us dive in into problem. 
Jupyter Notebook or JupyterLab dont let us see whole table so I can use: 
```
pd.options.display.max_rows = None
pd.options.display.max_columns = None
```
And write the number of columns and rows that we would like to see. Also I can use value “None” which is understanding like “no max value”. 
Usually to public datasets its additional text file with description of the valuables. It is also important to read this. 
It’s also good practice to make copy of our initial dataset. When we are start manipulating the data frame the changes we make are applied in original dataset. 
```
dataset = rawdata.copy() 
```
When we dealing with much larger dataset and still we need to get through it we can use:
```
dataset.info()  
```
As an output we see that there is no missing values (which barely happens in real world datasets (but let's practice on something easier :sweat_smile:), usually is full of N/A).

D - individual identification - indicates precisely who has been away during working hours. It is a label variable to distinguish the individuals from one another, not to carry any numeric information. 

We have to drop variable “ID” because it harm the estimation. 
```
dataset = dataset.drop([‘ID’], axis = 1) 
```
Next column, Reason from Absence - we have to keep in mind that they are represent categories that are equally meaningful  so they are categorical nominal variables. 
We use numbers and provide to them descriptions because using less characters will think the volume of our dataset, it's easier to digest, btw it is called “database theory”. 


Extracting distinct values only:
```
dataset[‘Reason for Absence’].unique() 
len(dataset['Reason for absence'].unique())
```
As an output we get 28. We can list them in order. 
```
sorted(dataset['Reason for absence’].unique())
```
There is no number ’20’ in the list. That means that nobody left the work because of “External causes of morbidity and mortality” (we know from the additional info UCI_ABS_TEXT) phew! 
We have to change this variables into dummy variables.
Dummy variable is an explanatory binary variable that equals
1 - if a certain categorical effect is present 
0 - if the same effect is absent

We our data we will do like this: 
1 - if person was absent because of reason 1
0 - if person was absent because any other reason 

next: 
1 - if person was absent because of reason 2
0 - if person was absent because any other reason


Fortunately I don’t have to do it manually it is possible thanks to panda by simply .get_dummies()
```
rcolumn = pd.get_dumies(dataset[‘Reason for Absence’]) 
```

rcolumn is new dataframe with 28 columns which contains information about which I wrote above. 

To this data frame we can add another column where it will be sum: 
```
rcolumn['check'] = rcolumn.sum(axis = 1)
```
In next stage I have to drop column ‘0’ from rcolumn dataframe. I'm doing this to avoid multicollinearity. For n categories we using n-1 dummies so I am dealing with 28 categories so I need only 27 dummies. (https://www.quora.com/How-and-why-having-the-same-number-of-dummy-variables-as-categories-is-problematic-in-linear-regression-Dummy-variable-trap-Im-looking-for-a-purely-mathematical-not-intuitive-explanation-Also-please-avoid-using-the) 

In original ‘dataset’ I still have column called ‘Reason for absence’, if we will leave it we will have duplication of information which lead to multicollinearity. So lets drop this column from ‘dataset’. 
If we will add our ‘rcolumn’ into ‘dataset’ that means that we will have additional 27 columns in dataframe. A bit too much. Lets group these variables, this action we call **classification**. 
We will group basing on features descriptions: 
Reson1 1-14 diseases 
Reason2 15-17 - pregnancy related 
Reason 3 18 - 21 - poisonings 
Reason 4 22-28 - light reasons 

We will create new data frame for each group. Thats why we needed to drop column with ID - because we need every individual have only one reason being out of work. 
So now we want to create a tables with only type of reason 
```
rcolumn.loc[:, 1:14].max(axis=1)
```
```
reasontype1 = rcolumn.loc[:, 1:14].max(axis=1)
```

Adding to data frame and rename it:
```
dataset = pd.concat([dataset, reasontype1, reasontype2, reasontype3, reasontype4], axis = 1)
dataset.columns.values
column_names = ['Month of absence', 'Day of the week', 'Seasons',
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours', 'Reason1', 'Reason2', 'Reason3', ‘Reason4']
dataset
```
Reordering columns because we want to see the reason first: 
```
reordered = ['Reason1', 'Reason2', 'Reason3', 'Reason4','Month of absence', 'Day of the week', 'Seasons',
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours’]
dataset = dataset[reordered]
```

Creating a checkpoints - an interim save of your work 
```
d_mod1 = dataset.copy() 
```

I called it d_mod1 which stands for modified data version 1. It is very good practice to creating checkpoints - it is help to organize, storing the current version of code so we reducing risk of losing our data at a later stages. 
We don’t have to do anything with date - month and the day of the week. 

Let’s move to next columns: 
Transportation Expense, Distance, Age, Daily Work Load, BMI - we are not going to manipulate them too. 

What we have next is: 
‘Education’ (high school (1), graduate (2), postgraduate (3), master and doctor (4))
'Son’ - Number of children 
‘Pet’ - Number of pets
Columns ‘Son’ and ‘Pet’ we will leave untouched. 

We have to change education into dummy variable. To not scroll down everything lets check what we have in ‘education’ variable 
```
d_mod1[‘Education’].unique()
d_mod1[‘Education’].value_counts() 
```
Now we can see that 611 is undergraduate and only 129 people holds higher degree (graduate, postgraduate, a master or a doctor) so it is not so relevant anymore. We can combine them in single category. 
We can assign undergraduate as 0 and at least graduate to 1: 
1 -> 0 
2 -> 1 
3 -> 1 
4 -> 1 
```
d_mod1['Education'] = d_mod1['Education'].map({1:0, 2:1, 3:1, 4:1})
d_mod1[‘Education'].unique()
d_mod1['Education'].value_counts() 
```
Everything is fine, we have only 0 and 1 and as we calculated previously: 0 - 611
1 - 129 

We are done with preprocessing! :partying_face: We should save it: 
```
d_pre = d_mod1.copy() 
```
Before we will start using ML we need to export this data as a *.csv file. Store this file in the same folder where you are currently working on. 

To machine learning part I will use the other notebook. 
Why? 
- Usually in teams different person is responsible for preprocessing and different one for actual ML 
- Because preprocessing file may start lagging 
- And because maybe we don’t know which method we should use and we can try several of them in different files 

Since is the new file we have to import libraries: 
```
import pandas as pd
import numpy as np
```
and load the preprocessed data: 
```
data = pd.read_csv('Absenteeism_preprocessed.csv')
```
We will use logistic regression. It is classifying technique so we are kind of classifying people into classes. 
The classes that we can make are: 
- people who are often absent 
- people who are not often absent 

We need cut-off line so we can use median. Everything below the medial will be considered as normal and above as often absent. 
```
data['Absenteeism time in hours'].median()
```
Result is 3.0 so we can make classes like this: 
<= 3.0 normal 
>= 4.0 often absent 

And this classes we have to change into 0s and 1s. It will be our targets. 
```
targets = np.where(data['Absenteeism time in hours'] > 3, 1, 0)
```
the numbers are respectively: np.where(condition,value if true, value if false) 

We can also write it like this to make it clearer: 
```
targets = np.where(data['Absenteeism time in hours'] > data['Absenteeism time in hours'].median(), 1, 0) 
```
And add it to our data: 
```
data['Excessive absenteeism'] = targets
```

I used median because it's also balancing the data 
```
targets.sum() / targets.shape[0]
```
and the result is around 0.46. And we can use split up to 40%-60% in logistic regression. When it comes to NN it need to be more balanced 55%-45%. 

Now we are creating inputs for regression: 
```
inputs = data.iloc[:, 0:22]
```
or even simplier: 
```
inputs = data.iloc[:,:-1]
```
Now we have to standarize the data: 
importing libraries, creating an object to scale data
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(inputs)
```
Scaler will be substracting the mean and devide it by the standard deviation, actual scalling: 
```
scaled_inputs = scaler.transform(inputs)
```
we can use scaler and transform function to new data too! 

Next we have to shuffle the data and split it into training dataset and test dataset. 
```
from sklearn.model_selection import train_test_split
train_test_split(scaled_inputs, targets)
```
sklearn also id doing shuffling, if you rerun this code you will get everytime different output 

Output have four arrays: 
training inputs 
testing inputs 
training outputs
testing outputs 

assigning arrays: 
```
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets)
```

We can check if it is splitted correctly: 
```
x_test.shape[0] / (x_train.shape[0] + x_test.shape[0])
```

This means that we have 25 / 75 split because it is set by default, usually we have 20 / 80 or 10 / 90 because we want to train on more data, so we can change it here: 
```
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, random_state = 20)
```
by random_state we make shuffle that it going to be shuffled in the same 'random' way 

Ok. Its end of the preprocessing, this time for real :smile: 

```
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
```

ML part: 
```
reg = LogisticRegression()
reg.fit(x_train, y_train)
```

as an output we got parameters that we can change 
```
reg.score(x_train, y_train)
```
Accuracy of my model is around 76% 
in other words : model learned to classify 76% of observation correclty

We can also check accuracy manually: 
```
model_outputs = reg.predict(x_train)
```
^ here are stored outputs from our model 

y_train contains true output so we can compare it 
```
y_train.shape
```
```
np.sum(model_outputs == y_train) 
```
and by deviding it: 
```
np.sum(model_outputs == y_train)/ y_train.shape
```

Intercept & coefficients: 

```
reg.intercept_
```
```
reg.coef_
```

Okay, it was easy :smile:. But its not says a lot, we want to know which variables this coefficients describes. We cannot get names of the columns simpy by columns.values using on scaled_inputs, because it is no longer pandas DataFrame, we used sklearn so the results are stored as "ndarray". 
BUT! 
We still have 'inputs' before scalling so we can use them! :smile:, so: 
```
features = inputs.columns.values
```
lets sum it up in table: 
```
table = pd.DataFrame(columns = ['Feature'], data = features)
table['Coefficients'] = np.transpose(reg.coef_) 
```



