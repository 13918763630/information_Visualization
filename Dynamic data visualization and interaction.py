# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output         # 回调
from matplotlib.animation import FuncAnimation
import missingno as msno
#数据处理部分 把营养统一成每克食物的量
menu = pd.read_csv('menu.csv')
for indexs in menu.columns:
    if indexs!='Category' and indexs!='Item' and indexs!='Serving':
     menu[indexs]=menu[indexs]/menu['Serving']/28.349523125
menu.to_csv("nutripergram.csv")
nutritype_copy=list(['Total Fat (% Daily Value)','Saturated Fat (% Daily Value)','Cholesterol (% Daily Value)','Sodium (% Daily Value)','Carbohydrates (% Daily Value)','Dietary Fiber (% Daily Value)','Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)','Iron (% Daily Value)'])
FoodClassNutri=pd.DataFrame(menu[menu['Category']=='Breakfast'][nutritype_copy].mean()).T
FoodType=['Beef & Pork','Chicken & Fish','Salads','Snacks & Sides','Desserts','Beverages','Coffee & Tea','Smoothies & Shakes']
for x in FoodType:
    tempfcn=pd.DataFrame(menu[menu['Category']==x][nutritype_copy].mean()).T
    FoodClassNutri=FoodClassNutri.append(tempfcn)
FoodType=['Breakfast','Beef & Pork','Chicken & Fish','Salads','Snacks & Sides','Desserts','Beverages','Coffee & Tea','Smoothies & Shakes']
FoodClassNutri.index=FoodType
WeightSeriesEuclid=pd.DataFrame(columns=['Breakfast','Beef & Pork','Chicken & Fish','Salads','Snacks & Sides','Desserts','Beverages','Coffee & Tea','Smoothies & Shakes']) 
tempdf=pd.DataFrame(columns=['Breakfast','Beef & Pork','Chicken & Fish','Salads','Snacks & Sides','Desserts','Beverages','Coffee & Tea','Smoothies & Shakes']) 
for x in range(1,1000,10):
    temp=[]
    for y in FoodType:
        temp.append(np.linalg.norm(FoodClassNutri.loc[[y]]*x-100))
    WeightSeriesEuclid.loc[x]=temp
#WeightSeriesEuclid.plot(kind="line")
color = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'black', 'gray', 'pink']
fig = plt.figure()
plt.xticks(rotation=45, ha="right", rotation_mode="anchor") 
#rotate the x-axis values
plt.subplots_adjust(bottom = 0.2, top = 0.9) 
#ensuring the dates (on the x-axis) fit in the screen
plt.ylabel('Euclid_Value')
plt.xlabel('Intake(g)')
def buildmebarchart(i=int):
    plt.legend(WeightSeriesEuclid.columns)    
    p = plt.plot(WeightSeriesEuclid[:i].index, WeightSeriesEuclid[:i].values) 
    #note it only returns the dataset, up to the point i    
    for i in range(0,9):        
        p[i].set_color(color[i]) 
        #set the colour of each curve
anianimator = FuncAnimation(fig, buildmebarchart, interval = 1)
#plt.show()
anianimator.save("WeightSeriesEuclid.gif",writer="pillow")

app = dash.Dash()
app.layout = html.Div([
    dcc.Dropdown(
        id = 'my_dropdown',
        options=[{'label':i,'value':i} for i in menu['Category'].unique()],
        value = '北京'),
    dcc.Dropdown(
        id = 'foodname',
        value='北京'),
    dcc.Graph(id = 'foodfigure'),
    dcc.Slider(
        id = 'foodweight',
        min = 0,
        max = 1000,
        step = 1,
        marks={0: '0',1000:'1000'},
        tooltip={"placement": "bottom", "always_visible": True},
        value = 0),
    html.Table(id='output-01'),
    html.Div([
        dcc.Graph(id = 'nutripercentagepie'),
        ],
      style = dict(width = '49%', display = 'inline-block', padding = '0 20')),
     html.Div([
        dcc.Graph(id = 'nutriprecentagediv')],
        style = dict(width = '49%', display = 'inline-block')),
])

def create_table(df, max_rows=12):
    """基于dataframe，设置表格格式"""
    
    table = html.Table(
        # Header
        [
            html.Tr(
                [
                    html.Th(col) for col in df.columns
                ]
            )
        ] +
        # Body
        [
            html.Tr(
                [
                    html.Td(
                        df.iloc[i][col]
                    ) for col in df.columns
                ]
            ) for i in range(min(len(df), max_rows))
        ]   
    )
    return table

def getRGB(val,limit):
  one=(255+255)/limit
  r=0
  g=0
  b=0
  if (val<(limit/2)):
     r=(int)(one*val)
     g=255
  elif (val>=(limit/2) and val <limit):
     r=255
     g=255-(int)((val-(limit/2))*one)
  else:
     r=255
  return f'rgb({r},{g},{b})'

def get_color(temp,range):
    return 'rgb({r}, 0, {b})'.format(
        r=(temp+range)/(range*2)*255,
        b=(1-(temp+range)/(range*2))*255
    )

@app.callback(Output('foodname', 'options'), [Input('my_dropdown', 'value')])
def update_foodselect(value):
        options=[{'label':i,'value':i} for i in menu[menu['Category']==value]['Item']]
        return options

@app.callback(Output('output-01', 'children'), [Input('my_dropdown', 'value'),Input('foodname', 'value')])
def update_nutritable(value1,value2):
        nutritype=list(['Total Fat (% Daily Value)','Saturated Fat (% Daily Value)','Cholesterol (% Daily Value)','Sodium (% Daily Value)','Carbohydrates (% Daily Value)','Dietary Fiber (% Daily Value)','Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)','Iron (% Daily Value)'])
        nutritable=np.round(menu[menu['Category']==value1][menu[menu['Category']==value1]['Item']==value2][nutritype]*100,2)
        table=create_table(nutritable)
        return table

@app.callback(Output('nutripercentagepie', 'figure'), [Input('my_dropdown', 'value'),Input('foodname', 'value')])
def update_nutripiefigure(value1,value2):
        nutritype=list(['Total Fat (% Daily Value)','Saturated Fat (% Daily Value)','Cholesterol (% Daily Value)','Sodium (% Daily Value)','Carbohydrates (% Daily Value)','Dietary Fiber (% Daily Value)','Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)','Iron (% Daily Value)'])
        nutritable=list(np.ravel(np.round(menu[menu['Category']==value1][menu[menu['Category']==value1]['Item']==value2][nutritype]*100,2)))
        figure=dict(
                 data=[go.Pie(
                 labels=nutritype,
                 values=nutritable)
                          ])
        return figure

@app.callback(Output('nutriprecentagediv', 'figure'), [Input('my_dropdown', 'value'),Input('foodname', 'value')])
def update_nutriprecentagediv(value1,value2):
        nutritype=list(['Total Fat (% Daily Value)','Saturated Fat (% Daily Value)','Cholesterol (% Daily Value)','Sodium (% Daily Value)','Carbohydrates (% Daily Value)','Dietary Fiber (% Daily Value)','Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)','Iron (% Daily Value)'])
        nutritable=list(np.ravel(np.round(menu[menu['Category']==value1][menu[menu['Category']==value1]['Item']==value2][nutritype]*100,2)))
        print(nutritable)
        nutritable=[x/sum(nutritable) for x in nutritable]
        print(nutritable)
        nutritable=[x*100 for x in nutritable]
        nutritable=[round(x,2)-10 for x in nutritable]
        print(nutritable)
        color=[get_color(x,max(np.abs(nutritable))) for x in nutritable]
        print(nutritable)
        figure=dict(
                data=[go.Bar(
                x=nutritype, 
                y=nutritable,
                marker_color=color)
                       ])

        return figure

@app.callback(Output('foodfigure', 'figure'), [Input('my_dropdown', 'value'),Input('foodname', 'value'),Input('foodweight', 'value')])
def update_foodfigure(value1,value2,value3):
        nutritype=list(['Total Fat (% Daily Value)','Saturated Fat (% Daily Value)','Cholesterol (% Daily Value)','Sodium (% Daily Value)','Carbohydrates (% Daily Value)','Dietary Fiber (% Daily Value)','Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)','Iron (% Daily Value)'])
        color=[getRGB(x,200) for x in list(np.ravel(menu[menu['Category']==value1][menu[menu['Category']==value1]['Item']==value2][nutritype].values.tolist())*value3)]
        #print(list(np.ravel(menu[menu['Category']==value1][menu[menu['Category']==value1]['Item']==value2][nutritype].values.tolist())*value3))
        figure=dict(
                data=[go.Bar(
                x=nutritype, 
                y=np.ones(10)*100,
                text=[round(x,2) for x in list(np.ravel(menu[menu['Category']==value1][menu[menu['Category']==value1]['Item']==value2][nutritype].values.tolist())*value3)],
                hoverinfo="x+text",
                marker_color=color)
                       ])
        return figure

if __name__ == '__main__':
    app.run_server(debug=True)
