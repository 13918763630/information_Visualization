import csv
import os
filename='menu.csv'
with open(filename)as file:
    reader=csv.reader(file)
    header_row=next(reader)
    for index,column_header in enumerate(header_row):
        print(index,column_header)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')

#导入数据
menu = pd.read_csv('menu.csv')

from matplotlib import cm as cm

sns.set_style("whitegrid")
plt.figure(figsize = (12,12))
cmap = sns.diverging_palette(400,10, as_cmap = True)

sns.heatmap(menu[menu.columns[3:]].corr(), vmax = 1, vmin = -0.5, center = 0, cmap=cmap,
           annot = True, linewidth = 0.1, cbar_kws = {'shrink':.5}, fmt= '.2f')
plt.title("Nutrition Value Correlation", size = 32, color = 'Pink')
plt.xticks(color='Pink', size=18)
plt.yticks(color='Pink', size=18)

plt.show()

plt.savefig('heatmap.png')

sns.set_style("whitegrid")

cols = ['Calories','Total Fat','Saturated Fat','Trans Fat','Cholesterol','Sodium','Carbohydrates','Dietary Fiber','Sugars','Protein',
        'Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)','Iron (% Daily Value)']
cm = np.corrcoef(menu[cols].values.T)

mask = np.zeros_like(cm, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (14,12))
cmap = sns.diverging_palette(400,10, as_cmap = True)

sns.heatmap(cm, mask=mask, vmax = 1, vmin = -0.5, center = 0, cmap=cmap,
           annot = True, linewidth = 1.0, cbar_kws = {'shrink':.5}, fmt= '.2f',
            xticklabels = cols, yticklabels = cols)
plt.title("Nutrition Value Correlation", size = 32, color = 'Pink')
plt.xticks(color='Pink', size=18)
plt.yticks(color='Pink', size=18)
#plt.savefig('Correlation Matrix'+'.png', bbox_inches = 'tight')

sns.set(style='whitegrid')
cols = ['Category','Calories', 'Total Fat', 'Sodium', 'Protein', 'Iron (% Daily Value)']
sns.pairplot(menu[cols], hue='Category', height=2.8)
plt.show()

plt.figure(figsize=(12,6))
menu.groupby('Category')['Item'].count().plot(kind='bar')

plt.title("Count\n", size=24)
plt.xlabel('Category', color='gray', fontsize=20)
plt.ylabel('Count', color='gray', fontsize=20)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

pyplt = py.offline.plot
x = menu['Category']
y = menu.groupby('Category')['Item'].count()
data = [go.Histogram(x=x)]

layout = go.Layout(
    title = 'Count',
    xaxis = {'title':'Category'}
)

fig = go.Figure(data, layout)

fig.show()

plt.figure(figsize=(12,6))
menu.groupby('Category')['Item'].count().plot(kind='pie', radius=1.8, autopct='%.2f')

plt.title("Count\n\n\n\n", size=24)
plt.gca().set_aspect('equal')

plt.show()

plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Calories'], color='Red')

plt.title("Calories Distribution\n", size=24)
plt.xlabel('Calories', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu.Calories.mean())
print(menu.Calories.median())

plt.figure(figsize=(20,8))
sns.swarmplot(x='Category', y='Calories', data=menu, size=12)

plt.title("Calories\n", size=32, color='gray')
plt.xlabel('Category', color='gray', fontsize=24)
plt.ylabel('Calories', color='gray', fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Calories', data=menu)

plt.title("Calories\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Calories',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Total Fat'], color='Orange')

plt.title("Total Fat Distribution\n", size=24)
plt.xlabel('Total Fat', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Total Fat'].mean())
print(menu['Total Fat'].median())plt.figure(figsize=(20,8))
sns.swarmplot(x='Category', y='Total Fat', data=menu, size=12)

plt.title("Total Fat\n", size=32, color='gray')
plt.xlabel('Category', color='gray', fontsize=24)
plt.ylabel('Total Fat', color='gray', fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Total Fat', data=menu)

plt.title("Total Fat\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Total Fat',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Saturated Fat'], color='c')

plt.title("Saturated Fat Distribution\n", size=24)
plt.xlabel('Saturated Fat', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Saturated Fat'].mean())
print(menu['Saturated Fat'].median())

plt.figure(figsize=(20,8))
sns.swarmplot(x='Category', y='Saturated Fat', data=menu, size=12)

plt.title("Saturated Fat\n", size=32, color='gray')
plt.xlabel('Category', color='gray', fontsize=24)
plt.ylabel('Saturated Fat', color='gray', fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Saturated Fat', data=menu)

plt.title("Saturated Fat\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Saturated Fat',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Trans Fat'], color='rosybrown')

plt.title("Trans Fat Distribution\n", size=24)
plt.xlabel('Trans Fat', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Trans Fat'].mean())
print(menu['Trans Fat'].median())

plt.figure(figsize=(20,8))
sns.swarmplot(x='Category', y='Trans Fat', data=menu, size=12)

plt.title("Trans Fat\n", size=32, color='gray')
plt.xlabel('Category', color='gray', fontsize=24)
plt.ylabel('Calories', color='gray', fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Trans Fat', data=menu)

plt.title("Trans Fat\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Trans Fat',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Cholesterol'], color='purple')

plt.title("Cholesterol Distribution\n", size=24)
plt.xlabel('Cholesterol', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Cholesterol'].mean())
print(menu['Cholesterol'].median())


plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Cholesterol', data=menu)

plt.title("Cholesterol\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Cholesterol',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Sodium'], color='Blue')

plt.title("Sodium Distribution\n", size=24)
plt.xlabel('Sodium', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Sodium'].mean())
print(menu['Sodium'].median())

plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Sodium', data=menu)

plt.title("Sodium\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Sodium',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Carbohydrates'], color='plum')

plt.title("Carbohydrates Distribution\n", size=24)
plt.xlabel('Carbohydrates', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Carbohydrates'].mean())
print(menu['Carbohydrates'].median())

plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Carbohydrates', data=menu)

plt.title("Carbohydrates\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Carbohydrates',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Dietary Fiber'], color='green')

plt.title("Dietary Fiber Distribution\n", size=24)
plt.xlabel('Dietary Fiber', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Dietary Fiber'].mean())
print(menu['Dietary Fiber'].median())

plt.figure(figsize=(20,8))
sns.swarmplot(x='Category', y='Dietary Fiber', data=menu, size=12)

plt.title("Dietary Fiber\n", size=32, color='gray')
plt.xlabel('Category', color='gray', fontsize=24)
plt.ylabel('Dietary Fiber', color='gray', fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(20,8))
sns.boxenplot(x='Category', y='Dietary Fiber', data=menu)

plt.title("Dietary Fiber\n", size=32, color='Gray')
plt.xlabel('Category',color='Gray',fontsize=24) 
plt.ylabel('Dietary Fiber',color='Gray',fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()plt.figure(figsize=(12,5))
ax = sns.distplot(menu['Sugars'], color='Pink')

plt.title("Sugars Distribution\n", size=24)
plt.xlabel('Sugars', fontsize=20)
plt.xticks(size=16)
plt.yticks(size=16)

print(menu['Sugars'].mean())
print(menu['Sugars'].median())

plt.figure(figsize=(20,8))
sns.swarmplot(x='Category', y='Sugars', data=menu, size=12)

plt.title("Sugars\n", size=32, color='gray')
plt.xlabel('Category', color='gray', fontsize=24)
plt.ylabel('Sugars', color='gray', fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

plt.figure(figsize=(20,8))
sns.swarmplot(x='Category', y='Protein', data=menu, size=12)

plt.title("Protein\n", size=32, color='gray')
plt.xlabel('Category', color='gray', fontsize=24)
plt.ylabel('Protein', color='gray', fontsize=24)
plt.xticks(size=16)  
plt.yticks(size=16)

plt.show()

%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')

trace = go.Scatter(
    y = menu['Cholesterol (% Daily Value)'].values,
    x = menu['Item'].values,
    mode='markers',
    marker=dict(
        size= menu['Cholesterol (% Daily Value)'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = menu['Cholesterol (% Daily Value)'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = menu['Item'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of Cholesterol (% Daily Value) per Item on the Menu',
    hovermode= 'closest',
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Cholesterol (% Daily Value)',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterChol')

trace = go.Scatter(
    y = menu['Dietary Fiber (% Daily Value)'].values,
    x = menu['Item'].values,
    mode='markers',
    marker=dict(
        size= menu['Dietary Fiber (% Daily Value)'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = menu['Dietary Fiber (% Daily Value)'].values,
        colorscale='Portland',
        reversescale = True,
        showscale=True
    ),
    text = menu['Item'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of Dietary Fiber (% Daily Value) per Item on the Menu',
    hovermode= 'closest',
        xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Dietary Fiber (% Daily Value)',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterChol')

# Calcium Scatter plots
trace = go.Scatter(
    y = menu['Calcium (% Daily Value)'].values,
    x = menu['Item'].values,
    mode='markers',
    marker=dict(
        size= menu['Calcium (% Daily Value)'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = menu['Calcium (% Daily Value)'].values,
        colorscale='Portland',
        reversescale = True,
        showscale=True
    ),
    text = menu['Item'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of Calcium (% Daily Value) per Item on the Menu',
    hovermode= 'closest',
        xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Calcium (% Daily Value)',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterChol')

# Iron Scatter plots
trace = go.Scatter(
    y = menu['Iron (% Daily Value)'].values,
    x = menu['Item'].values,
    mode='markers',
    marker=dict(
        size= menu['Iron (% Daily Value)'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = menu['Iron (% Daily Value)'].values,
        colorscale='Portland',
        reversescale = True,
        showscale=True
    ),
    text = menu['Item'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of Iron (% Daily Value) per Item on the Menu',
    hovermode= 'closest',
        xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Iron (% Daily Value)',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterChol')

# Vitamin A (% Daily Value) Scatter plots
trace = go.Scatter(
    y = menu['Vitamin A (% Daily Value)'].values,
    x = menu['Item'].values,
    mode='markers',
    marker=dict(
        size= menu['Vitamin A (% Daily Value)'].values,
        color = menu['Vitamin A (% Daily Value)'].values,
        colorscale='Portland',
        reversescale = True,
        showscale=True
    ),
    text = menu['Item'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Vitamin A (% Daily Value) per Item on the Menu',
    hovermode= 'closest',
        xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Vitamin A (% Daily Value)',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterChol')

# Vitamin C (% Daily Value) Scatter plots
trace = go.Scatter(
    y = menu['Vitamin C (% Daily Value)'].values,
    x = menu['Item'].values,
    mode='markers',
    marker=dict(
        size= menu['Vitamin C (% Daily Value)'].values,
        color = menu['Vitamin C (% Daily Value)'].values,
        colorscale='Portland',
        reversescale = True,
        showscale=True
    ),
    text = menu['Item'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Vitamin C (% Daily Value) per Item on the Menu',
    hovermode= 'closest',
        xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Vitamin C (% Daily Value)',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterChol')