
# Loading the churn modelling dataset and exploring it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import chi2_contingency
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
import streamlit as st


st.set_page_config(layout='wide')
with st.sidebar.header('1.Upload your CSV data'):
        uploaded_file= st.sidebar.file_uploader('Upload your input CSV file')
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu(
        menu_title='2.Dive into the dataset',
        options=["Home", "Exploratory Data Analysis", "Predicting Customer Churn"],
    )

if selected=="Home":
        st.markdown("<h1 style='text-align: center; color: Navy;'>Bank Customer Churn</h1>", unsafe_allow_html=True)
        st.image('https://miro.medium.com/max/640/1*RAeucVCKyFGXArObBsYnrw.png',width=400)
        st.subheader("When a customer departs or ceases an engagement with a company during a specific time period, this is referred to as customer turnover. As a result of increased rivalry, banks must preserve existing clients in order to maintain their position in society, as this is more cost-effective than gaining new ones.")
        st.subheader("Furthermore, modern technology has increased banks' data access, making data-driven client turnover analysis possible. Customer churn analysis, which investigates a set of factors in order to forecast customer attrition, is becoming increasingly popular.")

if selected=="Exploratory Data Analysis":

        #loading the dataset
        df= pd.read_csv('/Users/leazaarour/Desktop/Churn_Modelling.csv')
        #Exploratory Data Analysis
        st.title("Exploratory Data Analysis")
        st.subheader("1. Visualizing the Proportion of customer churned and retained")
        labels = 'Exited', 'Retained'
        sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        fig= plt.show()
        st.pyplot()

        #About 20% of our observations exited from the bank and approximately 80% of our observations did not exit the bank.

        df2 = df.drop(columns = 'Exited')
        st.code(df2.head(5))

        correlation= df2.corrwith(df['Exited']).plot.bar(figsize=(16,9),title='correlated with Exited',rot=45,grid=True)

        #The features with the highest correlation with our dependent variable are the Age, if the Member is Active or not and The Balance of the customer.
        #To be more precise and be sure if these features are highly correlated with our dependent variable , exited, we will do a correlation heatmap matrix.

        #heatmap
        st.subheader('2. Visualizing the correlation of our independent features with our target variable')
        corr = df.corr()
        plt.figure(figsize=(16,9))
        sns.heatmap(corr,annot=True)
        st.pyplot()

        #As we can see, looking at the exited row, the features with the highest correlation with our targeted variable are Age, Balance and if the customer is active or not.

        #let's look at the barchart of the Geography feature
        st.subheader('3. Customers Churn count of different countries')
        plt.figure(figsize=(8,6))
        ax = sns.countplot(data=df, x='Geography', hue='Exited')
        plt.title('Geography wise Churn count')
        plt.show()
        st.pyplot()
        st.subheader('4. Age distribution of our customers churning and existing')
        df["Age"].hist()
        plt.xlabel("Age")
        plt.ylabel("Amount of customers")
        plt.title("Age distribution", fontsize=15)
        plt.show()
        st.pyplot()

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #The ages with the highest frequency are between 30 and 40. As we can see, the Age of distribution is skewed right

        churned = df[df['Exited'] == 1]
        nonchurned = df[df['Exited'] == 0]

        churnedtarget = churned["Gender"].value_counts()
        nonchurnedtarget = nonchurned["Gender"].value_counts()
        st.subheader('5.Customers churn with respect to the gender')
        fig1, axs = plt.subplots(1, 2)

        axs[0].pie(churnedtarget, labels=churnedtarget.index, autopct='%1.1f%%', shadow=None)
        axs[0].axis('equal')
        axs[0].set_title('Existing customers')

        axs[1].pie(nonchurnedtarget, labels=nonchurnedtarget.index, autopct='%1.1f%%', shadow=None)
        axs[1].axis('equal')
        axs[1].set_title('Churning customers')

        plt.show()
        st.pyplot()
        #As seen in the above piechart, Male customers exited more than Female customers, around 57% of churned customers are Male customers.
        st.subheader('6.Credit Score Group of customer exiting the bank or not')
        st.write('After grouping our customers with respect of their credit score group, we visualized the count of the groups')
        df['AgeGroup'] = pd.cut(df.Age,bins=[17, 65, 93],labels=['Adult','Elderly'])
        df['CreditScoreGroup'] = pd.cut(df.CreditScore,bins=[300, 579, 669, 739, 799, 900],labels=[0, 1, 2, 3, 4])
        df['CreditScoreGroup'] = df.CreditScoreGroup.astype(int)

        # Draw count plot to check relation between Exited and continous features in dataset
        for feature in [ 'NumOfProducts', 'HasCrCard', 
                        'IsActiveMember','AgeGroup', 'CreditScoreGroup']:
                plt.figure(figsize=(5, 5))
        sns.countplot(x = feature, data = df, hue = 'Exited')
        st.pyplot()

        #From the bar plots already did, we can see that Adult have higher customer churn that Elderly customers.

        ##Preprocessing and Feature Engineering

        df.isnull().sum()
        df.nunique()
        df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
        df_train = df.sample(frac=0.8,random_state=200)
        df_test = df.drop(df_train.index)
        print(len(df_train))
        print(len(df_test))

        df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
        sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
        plt.ylim(-1, 5)

if selected == "Predicting Customer Churn":

# Loading the churn modelling dataset and exploring it
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.pipeline import Pipeline
        import xgboost as xgb
        from xgboost.sklearn import XGBClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.preprocessing import Normalizer
        from sklearn.impute import SimpleImputer
        from numpy import mean
        from numpy import std
        from sklearn.model_selection import KFold
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from scipy.stats import chi2_contingency
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_selection import RFECV
        from sklearn.feature_selection import RFE
        from sklearn.feature_selection import RFECV
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_predict, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVC
        import streamlit as st
        st.title('Predicting Customer Churn')
        st.subheader('1.We first start by splitting our data into training and testing')
        df= pd.read_csv('/Users/leazaarour/Desktop/Churn_Modelling.csv')

        ##Preprocessing and Feature Engineering

        df.isnull().sum()
        df.nunique()
        df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
        df_train = df.sample(frac=0.8,random_state=200)
        df_test = df.drop(df_train.index)
        st.write(len(df_train))
        st.write('The length of the training dataset is 8000')
        st.write(len(df_test))
        st.write('The length of the testing data set is 2000')

        st.subheader('2.Checking the number of null values and of duplicates')
        df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
        sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
        plt.ylim(-1, 5)
        
        #Comparing the two boxplots, we can see that the salary has little effect on the customer's churn.

        ##One hot encoding technique

        df = pd.get_dummies(data = df , drop_first = True)
        df.head()

        #CHOOSING THE BEST MODEL WITH THE HIGHEST ACCURACY
        #matrix of feature
        x = df.drop(columns=['Exited'])

        # target / dependant variable 
        y = df[ 'Exited']

        #training and testing the model
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x_train.select_dtypes(include= 'object').columns
        x_test.select_dtypes(include= 'object').columns

        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        st.subheader('3.Using Logistic Regression Model to predict customer churn')
        #Logistic Regression
        from sklearn.linear_model import LogisticRegression
        clr = LogisticRegression(random_state=0 )
        clr.fit(x_train,y_train)
        y_pred = clr.predict(x_test)
        from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score 

        acc = accuracy_score(y_test,y_pred)
        f1  = f1_score(y_test,y_pred)
        pr  =  precision_score(y_test,y_pred)
        re  =  recall_score(y_test,y_pred)

        results = pd.DataFrame([['Logistic Regression',acc,f1,pr,re]],columns=['Model','Accuracy','F1','precision','recall'])

        st.write(results)
        st.write('After modeling with logistic regression, we got an accuracy of 0.81')
        st.subheader('Confusion matrix of Logistic Regression')
        #confusion matrix
        cm = confusion_matrix(y_test,y_pred)
        st.write(cm)

        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator=clr,X=x_train,y=y_train,cv=10)
        st.subheader('4.Now moving to the Random Forest Classifier Model')
        #Random Forest Classifier

        from sklearn.ensemble import RandomForestClassifier
        clr_rf = RandomForestClassifier(random_state=0)
        clr_rf.fit(x_train,y_train)
        y_pred = clr_rf.predict(x_test)

        from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score 
        acc = accuracy_score(y_test,y_pred)
        f1  = f1_score(y_test,y_pred)
        pr  =  precision_score(y_test,y_pred)
        re  =  recall_score(y_test,y_pred)

        results_2 = pd.DataFrame([['Random forest classifier',acc,f1,pr,re]],columns=['Model','Accuracy','F1','precision','recall'])

        results = results.append(results_2,ignore_index=True)

        st.write(results)
        st.write('After using the Random Forest Classifier, we got an accuracy higher than the Logistic Regression model, the accuracy is approximately 0.87')
        st.subheader('Confusion Matrix for Random Forest Classification')
        #Confusion Matrix
        cm = confusion_matrix(y_test,y_pred)
        st.write(cm)
        st.subheader('5.Ending with the last model that we tried, which is XGBoost')
        #XGBoost
        from xgboost import XGBClassifier
        clr_xgb = XGBClassifier()
        clr_xgb.fit(x_train,y_train)

        y_pred = clr_xgb.predict(x_test)

        acc = accuracy_score(y_test,y_pred)
        f1  = f1_score(y_test,y_pred)
        pr  =  precision_score(y_test,y_pred)
        re  =  recall_score(y_test,y_pred)

        results_2 = pd.DataFrame([['XGBoost classifier',acc,f1,pr,re]],columns=['Model','Accuracy','F1','precision','recall'])

        results = results.append(results_2,ignore_index=True)

        st.write(results)

        st.subheader('The model with the highest accuracy is the Random Forest Classifier with an accuracy of 86%.')
        st.subheader('6.Final comments:')

        st.write('Results show that Random Forest Classifier, presented the best results. The final selection of the best model is not a task that can be generalized. It depends on several factors, like what kind of error your application wants to minimize, feature importance, model interpretation etc. This last factor is gaining importance nowadays.')
        st.write('In my experience, I have seen big credit companies choosing a simpler model with worse performance because it was simplier, or more intuitive. The final seleciton definetly should be done with a joint effort of data and domains knowledge specialists.')
        st.write('Generally speaking, AUC of 0.86 is already a satisfactory result, but again, it depends on the application needs and scope. Also, there are a few more steps we could do in order to check if we could improve performance, like feature selection, feature engineering and a mixed strategy of over and under sampling, for example.')




    



