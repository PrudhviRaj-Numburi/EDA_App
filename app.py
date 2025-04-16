import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
import itertools


def cramers_corrected_stat(confusion_matrix):
	""" calculate Cramers V statistic for categorical-categorical association.
	uses correction from Bergsma and Wicher, 
	Journal of the Korean Statistical Society 42 (2013): 323-328
	"""
	chi2 = ss.chi2_contingency(confusion_matrix)[0]
	n = confusion_matrix.sum().sum()
	phi2 = chi2/n
	r,k = confusion_matrix.shape
	phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) 
	rcorr = r - ((r-1)**2)/(n-1)
	kcorr = k - ((k-1)**2)/(n-1)
	return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# Page layout
## Page expands to full width
st.set_page_config(page_title='Data Science App',
    layout='wide')



# Model building
def build_model(data):
	sns.set_style('darkgrid')
	global target_variable


	st.markdown('**1.2- Dataset general info**')
	st.text('Dataset shape:')
	st.text(df.shape)

	categorical_attributes = list(data.select_dtypes(include=['object']).columns)
	st.text("Categorical Variables:")
	st.text(categorical_attributes)

	numerical_attributes = list(data.select_dtypes(include=['float64', 'int64']).columns)
	st.text("Numerical Variables:")
	st.text(numerical_attributes)


	st.markdown('**1.3- Duplicated values**')
	st.text(data.duplicated().sum())

	st.markdown('**1.4- Missing values**')
	st.text(data.isnull().sum())

	st.markdown('**1.5- Unique values in the Categorical Variables**')
	for col_name in data.columns:
		 if data[col_name].dtypes == 'object':
		 	unique_cat = len(data[col_name].unique())
 			st.text("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
	

	st.subheader('2- Exploratory Data Analysis (EDA)')
	hue = target_variable

	st.markdown('**2.1- Descriptive Statistics**')
	st.text(data.describe())

	st.markdown('**2.2- Outlier detectetion by Boxplot**')
	if len(numerical_attributes) == 0:
		st.text('There is no numerical variable')
	else:	
		for a in numerical_attributes:
			st.text(a)
			fig = plt.figure(figsize = (20,10))
			sns.boxplot(data[a])
			st.pyplot(fig)


	if data[target_variable].dtypes == 'O':
		catplots(data)
	else:
		if len(data[target_variable].unique()) > 5:
			numplots(data)
		else:
			catplots(data)

	

def catplots(data):
	sns.set_style('darkgrid')
	global target_variable
	hue = target_variable

	categorical_attributes = list(data.select_dtypes(include=['object']).columns)
	numerical_attributes = list(data.select_dtypes(include=['float64', 'int64']).columns)

	st.markdown('**2.3- Target Variable plot**')
	st.text("Target variable:" + hue)
	fig = plt.figure(figsize = (20,10))
	ax = sns.countplot(data[hue])
	for p in ax.patches:
		height = p.get_height()
		ax.text(x = p.get_x()+(p.get_width()/2), y  = height*1.01, s = '{:.0f}'.format(height), ha = 'center')
	st.pyplot(fig)


	st.markdown('**2.4- Numerical Variables**')
	#fig = plt.figure(figsize = (5,5))
	#sns.pairplot(data, hue = hue)
	#st.pyplot(fig)

	st.markdown('***2.4.1- Correlation***')

	try:
		fig = plt.figure(figsize = (20,10))
		sns.heatmap(data.corr(), cmap = 'Blues', annot = True)
		st.pyplot(fig)
	except:
		st.text('There is no numerical variable')


	st.markdown('***2.4.2- Distributions***')
	for a in numerical_attributes:
		st.text(a)
		fig = plt.figure(figsize = (20,10))
		sns.histplot(data = data , x =a , kde = True, hue = hue)
		st.pyplot(fig)

	

	st.markdown('**2.5- Categorical Variables**')

	if len(categorical_attributes) == 0:
		st.text('There is no categorical variable')

	else:
		for a in categorical_attributes:
			if a == hue:
				pass
			else:
				if len(data[a].unique()) < 13:

					st.text(a)
					fig = plt.figure()
					g = sns.catplot(data = data, x = a, kind = 'count', col = hue, sharey=False)
					for i in range(data[hue].nunique()):
						ax = g.facet_axis(0,i)
						for p in ax.patches:
							height = p.get_height()
							ax.text(x = p.get_x()+(p.get_width()/2),  y  = height * 1.01 , s = '{:.0f}'.format(height), ha = 'center')
					
					g.set_xticklabels(rotation=90)
					st.pyplot(g)

		st.markdown('***2.5.1 - Correlation between categorical***')
		corrM = np.zeros((len(categorical_attributes),len(categorical_attributes)))
		for col1, col2 in itertools.combinations(categorical_attributes, 2):
			idx1, idx2 = categorical_attributes.index(col1), categorical_attributes.index(col2)
			corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(data[col1], data[col2]))
			corrM[idx2, idx1] = corrM[idx1, idx2]

		corr = pd.DataFrame(corrM, index=categorical_attributes, columns=categorical_attributes)
		fig = plt.figure(figsize=(20, 10))
		sns.heatmap(corr, annot=True, cmap = 'Blues')
		plt.title("Cramer V Correlation between Variables")
		st.pyplot(fig)


def numplots(data):
	sns.set_style('darkgrid')
	global target_variable
	hue = target_variable

	categorical_attributes = list(data.select_dtypes(include=['object']).columns)
	numerical_attributes = list(data.select_dtypes(include=['float64', 'int64']).columns)

	st.markdown('**2.3- Target Variable plot**')
	st.text("Target variable:" + hue)
	fig = plt.figure(figsize = (20,10))
	sns.histplot(data = data , x = hue , kde = True)
	st.pyplot(fig)

	
	st.markdown('**2.4- Numerical Variables**')
	if len(numerical_attributes) == 0:
		st.text('There is no categorical variable')

	else:
		for a in numerical_attributes:
			if a == hue:
				pass

			else:
				st.text(a)
				fig = plt.figure(figsize = (20,10))
				fig = sns.lmplot(data = data, x = a, y = hue)
				st.pyplot(fig)



	st.markdown('**2.5- Categorical Variables**')

	if len(categorical_attributes) == 0:
		st.text('There is no categorical variable')

	else:
		for a in categorical_attributes:
			if a == hue:
				pass
			else:
				if len(data[a].unique()) < 13:
					st.text(a)
					fig = plt.figure(figsize = (20,10))
					sns.kdeplot(data = data, x = hue ,hue = a)
					st.pyplot(fig)

		st.markdown('***2.5.1 - Correlation between categorical***')
		corrM = np.zeros((len(categorical_attributes),len(categorical_attributes)))
		for col1, col2 in itertools.combinations(categorical_attributes, 2):
			idx1, idx2 = categorical_attributes.index(col1), categorical_attributes.index(col2)
			corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(data[col1], data[col2]))
			corrM[idx2, idx1] = corrM[idx1, idx2]

		corr = pd.DataFrame(corrM, index=categorical_attributes, columns=categorical_attributes)
		fig = plt.figure(figsize=(20, 10))
		sns.heatmap(corr, annot=True, cmap = 'Blues')
		plt.title("Cramer V Correlation between Variables")
		st.pyplot(fig)



st.write("""
	# Data Science App
	""")

st.image('data.jpg')

st.write("""
In this implementation, you can do the EDA of our dataset to speed-up your analysis! \n
To use this app, follow the steps: \n 
1º - Import your dateset in the sidebar on the left. \n
2º - Choose the target variable on sidebar. \n
3º - Click on the confirmation button to run the app and just wait for the results.
""")




# In[ ]:


# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")


# Main panel

# Displays the dataset
st.subheader('1- Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Dataset info:**')
    st.write('First 5 rows of the dataset:')
    st.write(df.head())
    st.write('Last 5 rows of the dataset:')
    st.write(df.tail())
    checker = False


    with st.sidebar.header('2. Select the target variable'):
    	target_variable = st.sidebar.selectbox('Select the target variable', list(df.columns))

    	if st.sidebar.button("Click to confirm the target variable"):
	    	checker = True

    if checker == True:
    	build_model(df)

else:
	st.write("Please, input a dataset")

