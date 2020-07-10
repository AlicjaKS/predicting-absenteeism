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
