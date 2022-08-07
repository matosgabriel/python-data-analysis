# Biblioteca de análise de dados
import pandas as pd
# Bibliotecas 
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
# Modelos de predição
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Importar base de dados
tabela = pd.read_csv("advertising.csv")

print(tabela)

# Mostrando correlação entre os dados
corr = tabela.corr()
print(corr)

# Criando gráfico para as correlações utilizando seaborn
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sbn.heatmap(tabela.corr(), cmap="Wistia", square='true', mask=mask)

# Utilizando o pyplot para exibir o gráfico
plt.title('Correlação entre os dados')
plt.show()

# Definindo as variáveis que serão utilizadas na predição
x = tabela[["TV", "Radio", "Jornal"]]
# Definindo a variável a ser prevista
y = tabela["Vendas"]

# Importando função que separa a base de dados em dados para treino e dados para teste
from sklearn.model_selection import train_test_split

# Separando a base de dados em dados para treino e dados para teste e atribuindo a variáveis
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2)

# Utilizando os modelos (inteligência artificial) de regressão linear e árvore de decisão
modelo_regressao_linear = LinearRegression()
modelo_arvore_decisao = RandomForestRegressor()

# Treinando os modelos com os dados da planilha (base de dados)
modelo_regressao_linear.fit(x_treino, y_treino)
modelo_arvore_decisao.fit(x_treino, y_treino)

# Utilizando os modelos treinados para prever y com dados de teste
previsao_regressao_linear = modelo_regressao_linear.predict(x_teste)
previsao_arvore_decisao = modelo_arvore_decisao.predict(x_teste)

# Importando o medidor de fator R^2 (fator de exatidão/acurácia do modelo)
from sklearn.metrics import r2_score

# Mostrando em tela o fator R^2 obtido para cada modelo treinado
print(r2_score(y_teste, previsao_regressao_linear))
print(r2_score(y_teste, previsao_arvore_decisao))

# Utilizando o pandas para criar tabela que representa a relação dos dados de teste y com as predições
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["regressao linear"] = previsao_regressao_linear
tabela_auxiliar["arvore decisao"] = previsao_arvore_decisao

# Criando gráfico com o seaborn que utiliza a tabela_auxiliar criada pelo pandas
plt.figure(figsize=(15, 6))
sbn.lineplot(data=tabela_auxiliar)
plt.title('Proximidade entre os valores de y (vendas) originais, obtidos com a regressão linear e obtidos com árvore de decisão')
plt.show()

# Lendo novos valores de outra planilha (base de dados)
tabela_novos = pd.read_csv("novos.csv")
print(tabela_novos)

# Utilizando o modelo de árvore de decisão treinado para realizar novas predições
print(modelo_arvore_decisao.predict(tabela_novos))

# Gráfico em barra da importância de cada variável no valor de interesse (correlação de cada variável)
sbn.barplot(x=x_treino.columns, y=modelo_arvore_decisao.feature_importances_)
plt.title('Importância de cada variável nas vendas')
plt.show()