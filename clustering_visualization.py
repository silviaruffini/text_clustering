import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import nlp_analysis as nlp
import joblib
import pandas as pd
import numpy as np
##########################
#### Analyse clustering
##############

model = joblib.load('doc_cluster.pkl')
num_clusters = 5

clusters=model.predict(nlp.tfidf_matrix)

nlp.df_clean['cluster'] = pd.DataFrame(clusters)
df_group =nlp.df_clean.groupby(["cluster","Project country"])["Nid"].nunique().to_frame()

df_group.reset_index(inplace=True)
df_group.columns = ['cluster', 'project','tot']

# create a list of our conditions
conditions = [
    (df_group["cluster"] ==0),
    (df_group["cluster"] ==1),
    (df_group["cluster"] ==2),
    (df_group["cluster"] ==3),
    (df_group["cluster"] ==4)
    ]
# create a list of the values we want to assign for each condition
values = ["School Milk Program for Agriculture Development", 'School Fruit and Vegetables Program for Agriculture Development', 'development & innovation', 'Market emergency','Fisheries, Maritime Affairs & environment']


df_group['description_cluster'] = np.select(conditions, values)

#assign colors to cluster
colors=['pinkyl','tempo','magenta','amp','blues']



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '10%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '5%',
    'margin-right': '5%',
    'padding': '20px 10p'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}



content_first_row = dbc.Row(
    [
        dbc.Col(
             dcc.Dropdown(
        id="dropdown",
        options=[
            {'label': values[i], 'value': i}
            for i in range(num_clusters)
           ],
        value=0,
        clearable=False,
            ),
            md=12)
    ]
)

@app.callback(
    Output("graph", "figure"), 
    [Input("dropdown", "value")])

def display_topic_cluster(n):
    
    df_text_bow = nlp.tfidf_matrix.toarray()
    bow_df = pd.DataFrame(df_text_bow)

    # Map the column names to vocabulary 
    bow_df.columns = nlp.vectorizer.get_feature_names()
    bow_df['cluster'] = pd.DataFrame(clusters)

    word_freq = pd.DataFrame(bow_df[bow_df.cluster == n].sum().sort_values(ascending = False))
    word_freq.reset_index(level=0, inplace=True)
    word_freq.columns=['word','frequency']
    
    if n>0:
        word_freq.drop(index=[0],inplace=True)
        
    fig = px.treemap(word_freq[0:30], path=[px.Constant(values[n]),'word'], values='frequency',
                color='frequency', hover_data=['frequency'],
                color_continuous_scale= colors[n])
    return fig




content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph'), 
            md=12)
    ]
)





content_zero_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='dataset_title_1', children=['Text Clustering Analysis'], className='card-title',
                                style=CARD_TEXT_STYLE),
                         html.P(id='card_text_1', children=['Providing a meaningful visualization of the main topics represented in the projects funded by the EU'], style=CARD_TEXT_STYLE),
                              
                        
                        
                        
                       
                    ]
                )
            ]
        ),
        md=12
    )
])

import plotly



content_third_row = dbc.Row(
   [
       dbc.Col(  
         dcc.Graph(id="graph2", 
                  figure = px.sunburst(df_group, path=['description_cluster','project'], values='tot',color='cluster')),md=6 ),
       dbc.Col(  
          dcc.Graph(id="graph3", figure =px.scatter(df_group, x="project", y="cluster", size="tot", color="description_cluster").update_layout(legend=dict(
    orientation="h",
    x=0,
    y=1.2,
    yanchor="bottom",
    xanchor="right"
)) ), md=6)
    ]
)





content = html.Div(
    [
       # html.H2('Text Analytics Dashboard', style=TEXT_STYLE),
       # html.Hr(),
        content_zero_row,
        html.Br(),
        content_first_row,
        content_second_row,
        content_third_row,
        html.Hr()
    ],
    style=CONTENT_STYLE
)


app.layout = html.Div([ content])






if __name__ == '__main__':
    app.run_server(debug=False)
