import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(  layout = 'centered' )

st.title('Testes Parciais')
st.write('Os gráficos abaixo mostram dados obtidos a partir de testes realizados sobre o conjunto parcial de dados, o qual compreende 30% do conjunto original: ')


teste_info = pd.DataFrame()

for _ in range(10):
    hyperparameters_csv = pd.read_csv(f'hyperparameters_{_}.csv')
    teste_info = pd.concat([teste_info, hyperparameters_csv], ignore_index=True)
    
# Gráfico de proporção de anotações (barra)

st.header('Proporção de Anotações')


st.write('Abaixo, segue gráfico exibindo a quantidade de anotações por classe de pássaro: ')

df = pd.read_csv('partial_proportion.csv')

fig = go.Figure()
colors_list = ['#4f772d','#00b4d8', '#ffd60a', '#6c757d', '#370617']
colors = dict(zip(df.columns, colors_list))


for species in df:
    fig.add_trace(
        go.Bar(
            x = [species], 
            y = df[species],
            name = species,
            text = df[species], 
            showlegend= False,
            textfont = go.bar.Textfont(color='black', size=14),
            textposition = 'outside',
            marker_color = colors[species], 
            hovertemplate = f'<b>{species}</b><br>'+'Anotrações: %{y}<extra></extra>'
        )
    )

fig.update_layout(
    yaxis_title = 'Anotações',
    font_size = 14, 
    height = 600
)

st.plotly_chart(fig)


# Gráfico de histórico de métrica (line)

st.header('Faster RCNN')

st.subheader('Primeiro Teste')

st.write('O teste foi conduzido com a rede de backbone resnet50_fpn_1x. Mantiveram-se contantes o tamanho do batch em 1 e a taxa de aprendizagem em 0.0001. O total de epochs e se o conjunto de treinamento seria embaralhado ou não variaram. ')

labels = ['train_loss','valid_loss', 'COCOMetric']
color_values = ['#636efa', '#EF553B','#00cc96']
colors = dict(zip(labels, color_values))

st.write('Abaixo, podemos visualizar os resultados obtidos por cada modelo: ')


df_epochs_list = [pd.read_csv(f'epoch_history_{_}.csv') for _ in range(6)]

fig01 = make_subplots(
    cols = 2, 
    rows = 3,
    row_titles = ['Epochs: 25', 'Epochs: 50', 'Epochs: 100'],
    column_titles = ['Train Shuffle: True', 'Train Shuffle: False'],
    x_title = 'Epochs',
)

rows_cols_indexs = [(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]

for i, df in enumerate(df_epochs_list):
    row, col = rows_cols_indexs[i]
       
    for label in labels:
        fig01.add_trace(
            go.Scatter(
                x = df['epochs'],
                y = df[label], 
                line = dict(color=colors[label]), 
                name = label,
                legendgroup = 'metrics',
                showlegend = False if i>0 else True,
            ),
            row = row, col=col
        )


fig01.update_layout(
    title = 'Faster RCNN - Resnet50_fpn_1x',
    #legend_title = 'Eval:',
    hovermode='x unified',
    font_size = 14,
    width= 800,
    height=600,
    
)

st.plotly_chart(fig01)
st.caption('Evolução da loss de treinamento, de validação e da métrica AP em função do número de epochs')

st.dataframe(teste_info.loc[0:5, ['batch_size', 'learning_rate', 'num_epochs', 'train_shuffle', 'final_COCOMetric']]) 
st.caption('Métrica final alcançada por cada modelo.')

st.write('Percebe-se que os modelos que não tiveram embaralhamento do conjunto de treinamento tiveram um desempenho ligeiramento inferior.')
st.write('Além disso, treinar os modelos por mais de 50 epochs não gera ganho expressivo de desempenho.')


st.subheader('Segundo Teste')


st.write('Com base nisso, mativeram-se constantes o embaralhamento do conjunto de treinamento, bem como a taxa de aprendizagem anterior. Neste teste, treinaram-se os modelos apenas em 25 e 50 epochs, variando o tamanho do batch em 4 e 8:')

df_epochs_list = [pd.read_csv(f'epoch_history_{_}.csv') for _ in range(6, 10)]

fig02 = make_subplots(
    cols = 2, 
    rows = 2,
    row_titles = ['Epochs: 25', 'Epochs: 50'],
    column_titles = ['Batch Size: 4', 'Batch Size: 8'],
    x_title = 'Epochs',
)

rows_cols_indexs = [(1,2),(2,2),(1,1),(2,1)]

for i, df in enumerate(df_epochs_list):
    row, col = rows_cols_indexs[i]
       
    for label in labels:
        fig02.add_trace(
            go.Scatter(
                x = df['epochs'],
                y = df[label], 
                line = dict(color=colors[label]), 
                name = label,
                legendgroup = 'metrics',
                showlegend = False if i>0 else True
            ),
            row = row, col=col
        )


fig02.update_layout(
    title = 'Faster RCNN - Resnet50_fpn_1x',
    #legend_title = 'Eval:',
    hovermode='x unified',
    width = 800,
    height = 600, 
    font_size = 14,
    
)

st.plotly_chart(fig02)
st.caption('Evolução da loss de treinamento, de validação e da métrica AP em função do número de epochs')

st.dataframe(teste_info.loc[6:9, ['batch_size', 'learning_rate', 'num_epochs', 'train_shuffle', 'final_COCOMetric']]) 
st.caption('Métrica final alcançada por cada modelo.')

st.write('Percebe-se que o desempenho dos modelos, em geral, foi inferior aos resultados anteriores. Todavia, é valido lembrar que empregar o tamanho de batch unitário pode levar a instabilidades nos gradientes. Maiores investigações serão feitas.')
