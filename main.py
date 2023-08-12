# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# %%
train_data=pd.read_csv("data/train.csv")

test_data=pd.read_csv("data/test.csv")



# %%
train_data.describe()

# %%


# %%
train_data.isna().values.any()


# %%
train_data.isna().sum()


# %%
train_data.Gender.fillna(train_data.Gender.mode()[0],inplace=True)
train_data.Dependents.fillna(0,inplace=True)
train_data.Self_Employed.fillna(train_data.Self_Employed.mode()[0],inplace=True)
train_data.LoanAmount.fillna(train_data.LoanAmount.mode()[0],inplace=True)
train_data.Loan_Amount_Term.fillna(train_data.Loan_Amount_Term.mode()[0],inplace=True)
train_data.Credit_History.fillna(train_data.Credit_History.mode()[0],inplace=True)
train_data.Married.fillna("No",inplace=True)
train_data.duplicated().values.any()

# %%
gen_dt=train_data.Gender.value_counts()
gen_dt

# %%
fig=px.histogram(gen_dt,x=gen_dt.index,y=gen_dt.values,color=gen_dt.index)
fig.update_xaxes(title_text="Gender")
fig.update_yaxes(title_text="Number of People")



# %%
Area=train_data.Property_Area.value_counts()
fig2=px.pie(labels=Area.index,values=Area.values,names=Area.index,hole=0.5,title="Property Area")
fig2.show()

# %%
train_data.head(5)

# %%
temp = train_data['Credit_History'].value_counts(ascending = True)
fig=px.bar(temp,x=temp.index,y=temp.values,color=temp.index)
fig.update_xaxes(title_text="Credit")
fig.update_yaxes(title_text="No. of Applicants")
fig.update_layout(coloraxis_showscale=False)
fig.show()

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

for i in var_mod:
    if train_data[i].dtype==object:
        train_data[i] = train_data[i].astype(str)
    train_data[i] = le.fit_transform(train_data[i])
train_data.head() 


# %%
X = train_data[['Credit_History','Gender','Married','Education']]
y = train_data['Loan_Status']

# %%
trained_pred=model.predict(X)


# %%
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']

for i in var_mod:
    if test_data[i].dtype==object:
        test_data[i] = test_data[i].astype(str)
    test_data[i] = le.fit_transform(test_data[i])
test_data.head() 

# %%

X_test=test_data[['Credit_History','Gender','Married','Education']]

# %%
test_pred=model.predict(X_test)
test_pred=pd.Series(test_pred)


# %%
test_data["Loan_Status"]=test_pred
test_data.sample(5)

# %%
test_data['Gender'].replace({2:"",1:"Male",0: "Female"},inplace=True)
test_data['Married'].replace({1:"Yes",0: "No"},inplace=True)
test_data['Education'].replace({1:"Not Graduate",0: "Graduate"},inplace=True)
test_data['Self_Employed'].replace({2:"",1:"Yes",0: "No"},inplace=True)
test_data['Property_Area'].replace({2:"Urban",1:"Semiurban",0: "Rural"},inplace=True)
test_data['Loan_Status'].replace({1:"Yes",0: "No"},inplace=True)
test_data.isna().sum()

# %%
output_df=test_data.set_index('Loan_ID')
output_df.to_csv('Predicted_Data.csv')
output_df.sample(10)


