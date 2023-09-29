# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 19:44:28 2023

@author: jeferson.dantas
"""
##############################################################
##############################################################
############SELEÇÃO DO PLANO ANALISADO########################
##############################################################
##############################################################
#Plano escolhido para a analise
plano_escolhido = 'plano2'

### nome da coluna considerada como dependente
plano_analise = 'plano'

#%% Carregar bibliotecas
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import math
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import norm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from statsmodels.stats.outliers_influence import variance_inflation_factor
current_directory = os.getcwd()
print(current_directory)

#%% FUNÇÕES - Todas estão localizadas nesta sessão 

#Função Predicao padrão
Vtimesteps = 6
perc_amostra = 0.8
best_model=[]

# função incrementar em  ais 1 mes a frente com base no anommes 'yyyymm'
def increment_anomes(anomes):
    """Incrementa o valor de ANOMES mensalmente."""
    year = int(int(anomes)/100)
    month = anomes % 100
    month += 1
    if month > 12:
        month = 1
        year += 1
    return year * 100 + month

# função que gera o MAE, MSE, RMSE e o r-quadrado dos valores previstos vs valores reais
def avaliacao(descricao, valores_reais, valores_preditos):

    # Cálculo do Erro Médio Absoluto (MAE)
    mae = np.mean(np.abs(valores_reais - valores_preditos))
    
    # Cálculo do Erro Quadrático Médio (MSE)
    mse = np.mean((valores_reais - valores_preditos) ** 2)
    
    # Cálculo da Raiz do Erro Quadrático Médio (RMSE)
    rmse = np.sqrt(mse)
    
    # Cálculo do Coeficiente de Determinação (R-squared - R²)
    r_squared = 1 - np.sum((valores_reais - valores_preditos) ** 2) / np.sum((valores_reais - np.mean(valores_reais)) ** 2)
    
    resultado = pd.DataFrame({
        "Descrição": [descricao],
        "Erro Médio Absoluto (MAE)": [mae],
        "Erro Quadrático Médio (MSE)": [mse],
        "Raiz do Erro Quadrático Médio (RMSE)": [rmse],
        "R-squared (R²)": [r_squared]
    })
    
    return resultado

# função que apresenta gráfico para visualização de autocorrelação
def verifica_autocorrelacao(residuos):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # ACF
    sm.graphics.tsa.plot_acf(residuos, lags=20, ax=ax[0])
    ax[0].set_title('Função de Autocorrelação (ACF) dos Resíduos')
    dw_stat = durbin_watson(residuos)
    ax[0].text(0.95, 0.95, f"Durbin-Watson statistic: {dw_stat:.4f}", transform=ax[0].transAxes, ha='right', va='top', fontsize=18, color="black")
    # PACF
    sm.graphics.tsa.plot_pacf(residuos, lags=20, ax=ax[1])
    ax[1].set_title('Função de Autocorrelação Parcial (PACF) dos Resíduos')
    plt.tight_layout()
    plt.show()

# função que faz stepwize de variavel a partir de regressão OLS
def stepwise_selection(data, formula, 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    """ 
    Implementa uma seleção stepwise para regressão OLS usando fórmula.
    
    Parâmetros:
        data - DataFrame com os dados
        formula - fórmula inicial (por exemplo, 'y ~ X1 + X2 + X3')
        threshold_in - inclusão de variável se seu valor-p < threshold_in
        threshold_out - exclusão de variável se seu valor-p > threshold_out
        verbose - se True, imprime o processo de seleção
    """
    included = formula.split("~")[1].strip().split(" + ")
    while True:
        changed = False
        
        # forward step
        excluded = list(set(data.columns) - set(included) - set(formula.split("~")[0].strip().split(" + ")))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            new_formula = formula.split("~")[0] + " ~ " + " + ".join(included + [new_column])
            model = sm.OLS.from_formula(new_formula, data).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Adicionando  {:30} com valor-p {:.6}'.format(best_feature, best_pval))

        # backward step
        current_formula = formula.split("~")[0] + " ~ " + " + ".join(included)
        model = sm.OLS.from_formula(current_formula, data).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Removendo   {:30} com valor-p {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break

    final_formula = formula.split("~")[0] + " ~ " + " + ".join(included)
    return final_formula


# função função que formata o dataframe para usar a predição RNN
def predict_rnn(dforiginal,plano, timesteps=Vtimesteps, modelo=best_model):
    yhatx = []
    for i in range(timesteps, len(dforiginal)):
        sequence = dforiginal.drop(plano, axis=1).iloc[i-timesteps:i].values
        sequence = sequence.reshape((1, timesteps, sequence.shape[1]))
        prediction = modelo.predict(sequence)
        yhatx.append(prediction[0][0])

    # Preencha as primeiras 6 previsões com NaN, pois não temos previsões para elas 
    yhatx = [np.nan]*timesteps + yhatx
    return yhatx

# função que cria um modelo de rnn simples para testes
def create_model(units=6, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(Vtimesteps, df.shape[1]-1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Função para estruturar os dados para RNN
def structure_data(df,plano, timesteps=Vtimesteps):
    X, y = [], []
    for i in range(timesteps, len(df)):
        X.append(df.drop(plano, axis=1).iloc[i-timesteps:i].values)
        y.append(df.iloc[i][plano])
    return np.array(X), np.array(y)

# função que diz o total de combinaçã de parametros para algoritimo do sarima e do rnn
def total_combinations(param_grid):
    total = 1
    for key in param_grid:
        total *= len(param_grid[key])
    return total

# função que define os otimizadores que poderão ser usados no RNN
def get_optimizer(name, learning_rate):
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop,
        'adagrad': Adagrad
    }
    
    if name in optimizers:
        return optimizers[name](learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {name} not recognized")


# função que extrai para dataframe o resutado do resumo de um modelo sarima
def extract_sarimax_info(results):
    # Extrair coeficientes e p-valores
    coef = results.params
    p_values = results.pvalues
    
    # Extrair outras estatísticas
    jb = results.test_normality(method='jarquebera')[0]
    lb = results.test_serial_correlation(method='ljungbox')[0]
    het = results.test_heteroskedasticity(method='breakvar')[0]
    
    # Criar dicionário com os resultados
    data = {}
    # Adicionar outras estatísticas ao dicionário
    data['jb_stat'] = jb[0]
    data['jb_pvalue'] = jb[1]
    data['lb_stat'] = lb[0][0]
    data['lb_pvalue'] = lb[1][0]
    data['het_stat'] = het[0]
    data['het_pvalue'] = het[1]
    data['nobs'] = results.nobs
    data['log_likelihood'] = results.llf
    data['aic'] = results.aic
    data['bic'] = results.bic
    data['hqic'] = results.hqic
    
    # Adicionar coeficientes e p-valores ao dicionário com prefixo
#    for param, value in coef.items():
#        data[f'coef_{param}'] = value
    for param, value in p_values.items():
        data[f'pvalue_{param}'] = value
    
    
    # Converter dicionário em dataframe
    df = pd.DataFrame([data])  # Note o [data] para transformar o dicionário em uma única linha de dataframe
    
    return df

# Função de busca por melhor arima encaixado no r2 de testes
def busca_melhor_sarimax(explicativas_teste,determinantes_teste, determinantes_treino, explicativas_treino, param_grid):
    #Buscando melhor modelo
    ii = len(explicativas_teste)-1
    df_resultado = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', 's', 'R2', 'R2_treino'])
    # Busca em grade
    best_aic = np.inf
    best_params = None
    best_seasonal_params = None
    best_r2 = -999999999.999
    combinacoes =  total_combinations(param_grid)
    qtd = int(combinacoes/100)
    qtd_i = 1
    perc = 0
    r2_treino = 0
    i=1
    for p in param_grid['p']:
        for d in param_grid['d']:
            for q in param_grid['q']:
                for P in param_grid['P']:
                    for D in param_grid['D']:
                        for Q in param_grid['Q']:
                            for s in param_grid['s']:
                                try:
                                    param = (p, d, q)
                                    param_seasonal = (P, D, Q, s)                                   
                                    if qtd_i >= qtd:
                                        perc = str(int((i/combinacoes)*100))
                                        qtd_i = 1
                                        print(f'Preocesso em {perc}%.')       
                                    else:
                                        qtd_i = qtd_i +1
                                    i = i+1
                                    model = sm.tsa.statespace.SARIMAX(determinantes_treino,
                                                                      exog=explicativas_treino,
                                                                      order=param,
                                                                      seasonal_order=param_seasonal,
                                                                      enforce_stationarity=False,
                                                                      enforce_invertibility=False)
                                    results = model.fit(disp=False)
                                    pred = results.predict(start=len(explicativas_treino), end=len(explicativas_treino)+ii, exog=explicativas_teste)
                                    #r2 = avaliacao('Modelo SARIMAX',determinantes_teste.iloc[:, 0],pred).at[0, 'R-squared (R²)']
                                    r2 = r2_score(determinantes_teste.iloc[:, 0], pred)
                                    fittedvalues = results.fittedvalues
                                    # Determinando o ponto de corte para os 20% iniciais
                                    corte = int(0.2 * len(determinantes_treino))
                                    # Selecionando os 80% finais dos dados
                                    determinantes_treino_80_final = determinantes_treino[corte:]
                                    fittedvalues_80_final = fittedvalues[corte:]
                                    # Calculando o R^2 para os 80% finais selecionados
                                    r2_treino = r2_score(determinantes_treino_80_final, fittedvalues_80_final)
                                    if r2 > best_r2 and r2 !=1:
                                    #if results.aic < best_aic and r2 !=1:
                                        best_r2 = r2
                                        best_aic = results.aic
                                        print(f'**** Melhor R2 de teste: {best_r2}. Parametros: ({p},{d},{q}). Parametros Sazonais: ({P},{D},{Q},{s}). AIC: {results.aic}. R2_treino: {r2_treino}')
                                        #print(f'best_AIC: {results.aic}')
                                        best_params = param
                                        best_seasonal_params = param_seasonal
                                        new_row = pd.concat([pd.DataFrame({'p': [p], 'd': [d], 'q': [q], 'P': [P], 'D': [D], 'Q': [Q], 's': [s], 'R2': [r2], 'R2_treino': [r2_treino]}), extract_sarimax_info(results)], axis=1, ignore_index=False)
                                        df_resultado = pd.concat([df_resultado, new_row], ignore_index=True)
                                    if  r2 >= 0.3 and r2 != best_r2: 
                                        print(f'*Consideravel R2 de teste: {r2}. Parametros: ({p},{d},{q}). Parametros Sazonais: ({P},{D},{Q},{s}). AIC: {results.aic}. R2_treino: {r2_treino}')
                                        new_row = pd.concat([pd.DataFrame({'p': [p], 'd': [d], 'q': [q], 'P': [P], 'D': [D], 'Q': [Q], 's': [s], 'R2': [r2], 'R2_treino': [r2_treino]}), extract_sarimax_info(results)], axis=1, ignore_index=False)
                                        df_resultado = pd.concat([df_resultado, new_row], ignore_index=True)
                                except:
                                    continue
            
    if best_params is None or best_seasonal_params is None:
        raise ValueError("Não foi possível encontrar um modelo SARIMAX adequado com os parâmetros fornecidos.")

    return best_r2, best_params[0],best_params[1],best_params[2], best_seasonal_params[0],best_seasonal_params[1],best_seasonal_params[2],best_seasonal_params[3],df_resultado

# função que cria modelo RNN nos padrões necessários para a busca por melhores hiper parâmetros
def criar_modelo_rnn(steps, shape, units,drop_out, optimizer,learning_rate,epochs, batch_size, X_train,y_train, duascamadas=False, drop_out2=0):

    vmodel = Sequential()
    if duascamadas:
        vmodel.add(LSTM(units, input_shape=(steps, shape), return_sequences=True))
    else:
        vmodel.add(LSTM(units, input_shape=(steps, shape), return_sequences=False))
    vmodel.add(Dropout(drop_out))
    if duascamadas:
        vmodel.add(LSTM(units, return_sequences=False))
        vmodel.add(Dropout(drop_out2))
    vmodel.add(Dense(1))
    vopt = get_optimizer(optimizer, learning_rate)
    vmodel.compile(optimizer=vopt, loss='mean_squared_error') 
    vmodel.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return vmodel



# função que busca melhor RNN conforme hiperparametros 
def grid_search_melhor_rnn(param, df_avaliacao, percent_amostra_teste, var_dependente, min_r2):
    r2_best = -9999999999
    best_step = -1
    df = df_avaliacao.copy()
    combinacoes =  total_combinations(param)
    qtd = int(combinacoes/100)
    qtd_i = 1
    perc = 0
    units_best = 0
    learning_rate_best = 0
    drop_out_best = 0
    drop_out2_best = 0
    optimizer_best = ""
    batch_size_best = 0
    epochs_best = 0
    r2_train_best = 0
    i=1
    df_resultado = pd.DataFrame(columns=['R2', 'R2_treino','optimizer','learning_rate','units','drop_out','batch_size','epochs','steps','duascamadas','drop_out2'])
    best_model = [] 
    

    duascamadas = param_grid['duascamadas']
    print(f'Início do processo: {combinacoes} interações')       
    for units in param_grid['units']:
        for optimizer in param_grid['optimizer']:
            for learning_rate in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    for drop_out in param_grid['drop_out']:
                        for drop_out2 in param_grid['drop_out2']:
                            for epochs in param_grid['epochs']:
                                for steps in param_grid['steps']:
                                    X, y = structure_data(df,var_dependente, timesteps=steps)
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_amostra_teste, shuffle=False)
                                    shape = (df.shape[1]-1)
                                    np.random.seed(0)
                                    tf.random.set_seed(0)
                                    vmodel = criar_modelo_rnn(steps=steps, shape=shape, units=units,drop_out=drop_out, optimizer=optimizer,learning_rate=learning_rate,epochs=epochs, batch_size=batch_size, X_train=X_train,y_train=y_train, duascamadas=duascamadas, drop_out2=drop_out2)
                                    prediction_train = vmodel.predict(X_train,  verbose=0)
                                    prediction = vmodel.predict(X_test,  verbose=0)
                                    r2_train=r2_score(y_train,prediction_train)
                                    r2=r2_score(y_test,prediction)
                                    if r2>= r2_best:
                                        r2=r2_score(y_test,prediction)
                                        r2_best = r2
                                        r2_train_best = r2_train
                                        units_best = units
                                        optimizer_best = optimizer
                                        learning_rate_best = learning_rate
                                        batch_size_best = batch_size
                                        drop_out_best = drop_out
                                        drop_out2_best = drop_out2
                                        epochs_best = epochs
                                        best_step = steps
                                        best_model = vmodel
                                        print(f"Melhor R2 encontrado: {r2_best}, R2 Treino = {r2_train_best}, com os parametros optimizer={optimizer_best}, learning_rate={learning_rate_best}, units={units_best}, drop_out={drop_out_best},  batch_size={batch_size_best}, epochs={epochs_best}, steps={best_step}, duascamadas={duascamadas},drop_out2={drop_out2_best}.")
                                        best_model_plano.save(f'{current_directory}/rnn_melhor_modelo_plano {r2_best}%.h5')
                                    if  r2>= min_r2:
                                        new_row = pd.DataFrame({'R2': [r2], 'R2_treino': [r2_train], 'optimizer': [optimizer],'learning_rate': learning_rate, 'units':[units],'drop_out':[drop_out],'batch_size':[batch_size], 'epochs':[epochs_best],'steps':[best_step],'duascamadas':[duascamadas],'drop_out2':[drop_out2_best]})
                                        df_resultado = pd.concat([df_resultado, new_row], ignore_index=True)
                                    i = (i+1)
                                    if qtd_i >= qtd:
                                        perc = str(int((i/combinacoes)*100))
                                        qtd_i = 1
                                        print(f'Preocesso em {perc}%. Interação {i} de {combinacoes}')       
                                    else:
                                        qtd_i = qtd_i +1
            
    print(f"*****FIM O PROCESSO: Melhor R2 encontrado: {r2_best}, R2 Treino = {r2_train_best}, com os parametros optimizer={optimizer_best}, learning_rate={learning_rate_best}, units={units_best}, drop_out={drop_out_best},  batch_size={batch_size_best}, epochs={epochs_best}, steps={best_step}, duascamadas={duascamadas},drop_out2={drop_out2_best}.")
    return best_model, best_step, df_resultado

# função que gera gáfico simples
def plot(df, colunay, titulo):
    n=3
    plt.figure(figsize=(26, 16))
    
    # Converter 'Ano_mes' para numérico
    df['Ano_mes'] = df['Ano_mes'].astype(str)
    label = colunay
    color = 'black'
    plt.plot(df['Ano_mes'], df[colunay], '-o', label=label, markersize=5, color=color, linewidth=3)
    
    # Remover rótulos duplicados da legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=24)  # Aumentar a fonte da legenda
    
    ax = plt.gca()
    ax.set_xticks(df['Ano_mes'][::n])  # Definir os ticks do eixo X manualmente a cada 'n' pontos
    ax.set_xticklabels(df['Ano_mes'][::n], rotation=90)  # Rotacionar os rótulos do eixo X
    ax.set_xlabel('Ano_mes', fontsize=24)  # Aumentar a fonte do rótulo do eixo x
    ax.set_ylabel(colunay, fontsize=24)      # Aumentar a fonte do rótulo do eixo y
    ax.tick_params(axis='both', labelsize=20) # Aumentar a fonte dos ticks dos eixos
    plt.title(titulo, fontsize=24)            # Aumentar a fonte do título
    plt.show()

 # função que gera gráfico de linhas e barras com eixos y distintos
def grafico_linha_barra(x_linha, y_linha, nome_variavel_linhax,nome_variavel_linhay, x_barra, y_barra, nome_variavel_barra, titulo):
    plt.figure(figsize=(26, 16))
    
    # Referência para o eixo original
    ax1 = plt.gca()
    
    # Plotando os dados originais
    line1, = ax1.plot(x_linha, y_linha, 'o-', markersize=1, linewidth=2, label=f'{nome_variavel_linhay}', color='black')
    
    # Criando um eixo secundário para a nova variável
    ax2 = ax1.twinx()
    bars = ax2.bar(x_barra, y_barra, alpha=0.6, color='black')
    ax2.set_ylabel(f'{nome_variavel_barra}', fontsize=24)
    
    # Adicionando uma legenda para as barras
    bars.set_label('Plano')
    
    # Configurações gerais do gráfico
    ax1.set_xlabel(f'{nome_variavel_linhax}', fontsize=24)
    ax1.set_ylabel(f'{nome_variavel_linhay}', fontsize=24)
    plt.title(f'{titulo}', fontsize=44)
    
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    ax2.tick_params(axis='y', labelsize=24)
    
    # Combinando as legendas
    lines = [line1, bars]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=24)
    
    plt.grid(True)
    plt.show()

#gráfico correlação plano (x)
def gcorrplano():
    df=resultados_planos_indicadores
    df=df.reset_index(drop=True)
    nome_plano = f'Plano {plano_escolhido}'
    df=df[[plano_analise, 'txdesemprego','ipca','vegetativo', 'concorrencia']]
    
    # Calcula a matriz de correlação
    corr_matrix = df.corr()
    correlacoes = corr_matrix[plano_analise].drop(plano_analise)

    # Normalizar os valores de correlação e obter as cores correspondentes
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    colors = plt.cm.coolwarm(norm(correlacoes.values))
    
    plt.figure(figsize=(10, 6))
    
    # Plotar as correlações em um gráfico de barras
    bars = sns.barplot(x=correlacoes.index, y=correlacoes.values, palette="coolwarm")
    
    plt.title('Correlação com outras variáveis ({plano_escolhido})')
    plt.ylim(-1, 1)
    
    # Adicionar rótulos nas barras
    for bar in bars.patches:
        y_value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_value, round(y_value, 2), 
                ha='center', va='center', color='black', size=24)
    
    # Remover os rótulos e ticks do eixo Y
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    
# gera gráficos de decompozição da sazonalidade    
def decompor_sazonalidade(df, periodos):
    df_sazonal = df
    result = seasonal_decompose(df, model='additive', period=periodos)
    
    # Ajuste o tamanho da figura antes de chamar result.plot()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(26, 16))
    result.observed.plot(ax=ax1, color='black', linewidth=2)
    ax1.set_ylabel('Valor Real', fontsize=24)
    result.trend.plot(ax=ax2, color='black', linewidth=2)
    ax2.set_ylabel('Tendência', fontsize=24)
    result.seasonal.plot(ax=ax3, color='black', linewidth=2)
    ax3.set_ylabel('Sazonalidade', fontsize=24)
    result.resid.plot(ax=ax4, color='black', linewidth=2)
    ax4.set_ylabel('Resíduos', fontsize=24,)
    
    print(f'Média dos resíduos: {result.resid.mean()}')
    
    # Ajuste os ticks do eixo x para mostrar todos
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(df_sazonal.index)
        ax.tick_params(axis='x', labelsize=20)  # Ajuste o valor 20 conforme necessário
        ax.tick_params(axis='y', labelsize=20)  # Ajuste o valor 20 conforme necessário    
    plt.tight_layout()
    plt.show()


    # Plotando o valor real e sazonalidade no mesmo gráfico
    fig, ax1 = plt.subplots(figsize=(26, 10))
    
    result.observed.plot(ax=ax1, label='Valor Real', color='black', linewidth=2)
    ax1.set_ylabel('Valor Real', fontsize=24)
    
    # Criando eixo secundário para sazonalidade
    ax1b = ax1.twinx()
    result.seasonal.plot(ax=ax1b, label='Sazonalidade', color='grey', linestyle='--', linewidth=2)
    ax1b.set_ylabel('Sazonalidade', fontsize=24, color='grey')
    ax1b.tick_params(axis='y', labelcolor='black', labelsize=20)
    
    # Combinando as legendas de ambos os eixos
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=20)
    
    # Ajuste os ticks do eixo x para mostrar todos
    ax1.set_xticks(df.index)
    ax1.tick_params(axis='x', labelsize=20)  # Ajuste o valor 20 conforme necessário
    ax1.tick_params(axis='y', labelsize=20)  # Ajuste o valor 20 conforme necessário
    
    plt.tight_layout()
    plt.show()

# função que gera gráfico mostrando as tendências simuladas
def gtendencia(df_tendencia, coluna, titulo):
    plt.figure(figsize=(20, 10), dpi=100)
    # Criar gráfico de linhas
    sns.lineplot(x='seq_tempo', y=coluna, data=df_tendencia, hue='modelo', 
                 palette={'Predição': 'red', 'Original': 'blue'}, marker='o')  # Adicionado 'Original': 'blue'
    plt.xlabel('Seq Tempo')
    plt.ylabel(f'{coluna}')
    plt.title(f'[{titulo}]Tendencia de {coluna}')
    plt.grid(True)
    plt.show()
    
# Função para modelar e prever usando ARIMA
def arima_forecast(df, coluna, p, d, q, vperiodos):
    model = ARIMA(df[coluna], order=(p,d,q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=vperiodos)
    return forecast

# Função para modelar e prever usando SAROIMA
def sarima_forecast(df, coluna, p, d, q, sP, sD,sQ, s, vperiodos):
    model = SARIMAX(df[coluna], order=(p,d,q), seasonal_order=(sP,sD,sQ,s))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=vperiodos)
    return forecast

# função que gera gráfico mostrando as tendências simuladas do ARIMA E SARIMA
def gera_resultados_tendencia(drx, coluna, p, d, q, sP, sD, sQ, s , vperiodos):
    # Projeções
    dr = drx[['seq_tempo',coluna]]
    arima_pred = arima_forecast(dr,coluna, p,d,q, vperiodos)
    sarima_pred = sarima_forecast(dr,coluna, p,d,q, sP, sD, sQ,s, vperiodos )
    
    # Criar sequenciais de tempo para as projeções
    future_seq_tempo = [dr['seq_tempo'].iloc[-1] + i for i in range(1, vperiodos+1)]
    
    # Criar dataframes para as projeções
    his_arima = pd.DataFrame({'seq_tempo': future_seq_tempo, coluna: arima_pred, 'modelo': 'Predição'})
    his_sarima = pd.DataFrame({'seq_tempo': future_seq_tempo, coluna: sarima_pred, 'modelo': 'Predição'})



    df_tendencia_arima = dr.copy()
    df_tendencia_arima['modelo'] = 'Original'
    df_tendencia_arima = pd.concat([df_tendencia_arima, his_arima[['seq_tempo', coluna ,'modelo']] ], axis=0)    
    gtendencia(df_tendencia_arima, coluna,'ARIMA')

    df_tendencia_sarima = dr.copy()
    df_tendencia_sarima['modelo'] = 'Original'
    df_tendencia_sarima = pd.concat([df_tendencia_sarima, his_sarima[['seq_tempo',coluna ,'modelo']] ], axis=0)
    gtendencia(df_tendencia_sarima, coluna,'SARIMA')
    
    return his_arima, df_tendencia_arima, his_sarima, df_tendencia_sarima



# função que gera gráfico mostrando as simulações dos modelos combinadas
def plot_colored_line(df, tempo_predicao, colunay, colunay2, colunay3, titulo):
    plt.figure(figsize=(26, 16))
    
    # Converter 'Ano_mes' para numérico
    df['Ano_mes'] = df['Ano_mes'].astype(str)
    
    for i in range(1, len(df)):
        color = 'red' if df['Ano_mes'].iloc[i] >= anomes_predicao else 'blue'
        label = 'Predição ' + colunay if df['Ano_mes'].iloc[i] == tempo_predicao else None
        plt.plot(df['Ano_mes'].iloc[i-1:i+1], df[colunay].iloc[i-1:i+1], color=color, label=label, linewidth=3)
        # Adicionar o valor do dado como texto se Ano_mes >= anomes_predicao
        #if df['Ano_mes'].iloc[i] >= anomes_predicao:
        plt.text(df['Ano_mes'].iloc[i], df[colunay].iloc[i], str(int(df[colunay].iloc[i])), fontsize=10, ha='right', va='bottom', color=color)
            
    for i in range(1, len(df)):
        color = 'black' if df['Ano_mes'].iloc[i] >= anomes_predicao else 'blue'
        label = 'Predição ' + colunay2 if df['Ano_mes'].iloc[i] == tempo_predicao else None
        plt.plot(df['Ano_mes'].iloc[i-1:i+1], df[colunay2].iloc[i-1:i+1], color=color, label=label, linewidth=3)
        if df['Ano_mes'].iloc[i] >= anomes_predicao:
            plt.text(df['Ano_mes'].iloc[i], df[colunay2].iloc[i], str(int(df[colunay2].iloc[i])), fontsize=10, ha='right', va='bottom', color=color)
        
    for i in range(1, len(df)):
        color = 'orange' if df['Ano_mes'].iloc[i] >= anomes_predicao else 'blue'
        label = 'Predição ' + colunay3 if df['Ano_mes'].iloc[i] == tempo_predicao else None
        plt.plot(df['Ano_mes'].iloc[i-1:i+1], df[colunay3].iloc[i-1:i+1], color=color, label=label, linewidth=3)
        if df['Ano_mes'].iloc[i] >= anomes_predicao:
            plt.text(df['Ano_mes'].iloc[i], df[colunay3].iloc[i], str(int(df[colunay3].iloc[i])), fontsize=10, ha='right', va='bottom', color=color)
    
    # Remover rótulos duplicados da legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=24)  # Aumentar a fonte da legenda
    
    ax = plt.gca()
    ax.set_xticks(df['Ano_mes'])  # Definir os ticks do eixo X manualmente
    ax.set_xticklabels(df['Ano_mes'], rotation=90)  # Rotacionar os rótulos do eixo X
    ax.set_xlabel('Ano_mes', fontsize=14)  # Aumentar a fonte do rótulo do eixo x
    ax.set_ylabel(colunay, fontsize=24)      # Aumentar a fonte do rótulo do eixo y
    ax.tick_params(axis='both', labelsize=20) # Aumentar a fonte dos ticks dos eixos
    plt.title(titulo, fontsize=24)            # Aumentar a fonte do título
    plt.show()

#Histograma - analisando normalidade
def cria_hostograma_normalidade(coluna, descricao):
    
    plt.figure(figsize=(10,6))
    
    # Histograma
    plt.hist(coluna, bins=10, density=True, color='grey', edgecolor='black', alpha=0.6, label='Dados')
    
    # Calcula a média e o desvio padrão da variável
    mu, std = coluna.mean(), coluna.std()
    
    # Gera valores para o eixo x
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    
    # Calcula a PDF para os valores x
    p = norm.pdf(x, mu, std)
    
    # Adiciona a curva de distribuição normal ao gráfico
    plt.plot(x, p, 'k', linewidth=2, label='Distribuição Normal')
    
    # Título e rótulos
    plt.title(f'Histograma {descricao} vs Distribuição Normal')
    plt.xlabel(f'Valores de {descricao}')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    plt.show()


# %% carregar dados iniciais 

# Preparar variáveis de tendencia
n_periodos = 24


serieIPCA = pd.read_csv("dados brutos/ipca.csv")
seriedesempregotrimestre = pd.read_csv("dados brutos/seriedesempregotrimestre.csv") 
seriedesemprego = pd.read_csv("dados brutos/seriedesemprego.csv") 
dados_anomes =  pd.read_csv("dados tratados/tabela_anomes.csv", encoding='ISO-8859-1', delimiter=',') 
dados_anomes = dados_anomes.drop(dados_anomes.columns[0], axis=1)
dados_planos = pd.read_csv("dados tratados/tabela_planos.csv", encoding='ISO-8859-1', delimiter=',') 
dados_planos = dados_planos.drop(dados_planos.columns[0], axis=1)

print('Meses: ',dados_anomes['Ano_mes'].nunique())
print('Planos: ',dados_planos['Plano'].nunique())

resultados_socioeconomicos = pd.read_csv("dados tratados/tabela_socioeconomicos.csv", encoding='ISO-8859-1', delimiter=',')
resultados_socioeconomicos = resultados_socioeconomicos.drop(resultados_socioeconomicos.columns[0], axis=1)
resultados_planos_indicadores = pd.read_csv("dados tratados/tabela_planos_socioeconomicos.csv", encoding='ISO-8859-1', delimiter=',')
planos_indicadores = resultados_planos_indicadores.copy()
resultados_planos_indicadores = resultados_planos_indicadores.drop(resultados_planos_indicadores.columns[0], axis=1)
resultados_beneficiarios_anomes =  pd.read_csv("dados tratados/tabela_beneficiarios_socioeconomicos.csv", encoding='ISO-8859-1', delimiter=',')
resultados_beneficiarios_anomes = resultados_beneficiarios_anomes.drop(resultados_beneficiarios_anomes.columns[0], axis=1)

# Usando a função pivot para transformar idplano em colunas
df_pivot = resultados_planos_indicadores.pivot(index='seq_tempo', columns='idplano', values='Benef_ativos').reset_index()
# Renomeando as colunas para um formato mais claro
df_pivot.columns = [f'plano{col}' if col != 'seq_tempo' else col for col in df_pivot.columns]
# Fazendo merge com o dataframe original para obter as outras colunas
cols_to_merge = ['seq_tempo',  'txdesemprego',  'ipca', 'nascimentos', 'obitos']
resultados_planos_indicadores = df_pivot.merge(resultados_planos_indicadores[cols_to_merge].drop_duplicates(), on='seq_tempo', how='left')

resultados_planos_indicadores[plano_analise] = resultados_planos_indicadores[plano_escolhido]
resultados_planos_indicadores['beneficiarios'] = (resultados_planos_indicadores['plano1']  + resultados_planos_indicadores['plano2']  + resultados_planos_indicadores['plano3'] + resultados_planos_indicadores['plano4'] + resultados_planos_indicadores['plano5']) 
resultados_planos_indicadores['concorrencia'] = ((resultados_planos_indicadores['beneficiarios'] - resultados_planos_indicadores[plano_analise]) /  resultados_planos_indicadores['beneficiarios'])*100
resultados_planos_indicadores[plano_analise] = resultados_planos_indicadores[plano_analise].diff()
resultados_planos_indicadores[plano_analise] = resultados_planos_indicadores[plano_analise].fillna(resultados_planos_indicadores[plano_analise].shift(-1))
resultados_planos_indicadores['vegetativo'] = (resultados_planos_indicadores['nascimentos'] - resultados_planos_indicadores['obitos']) 


dados = resultados_planos_indicadores.copy()
cols = dados.columns
scaler_dados_padronizados  = MinMaxScaler()
dados_padronizados  = scaler_dados_padronizados.fit_transform(dados)
dados_padronizados = pd.DataFrame(dados_padronizados, columns=cols)

resultados_planos_indicadores.to_excel(f'{current_directory}/graf_resultados_planos_indicadores.xlsx')

# %% Geral de beneficiarios

    
# Usar a função para plotar o gráfico

plot(dados_anomes, 'Benef_ativos', 'Evolução de beneficiários ativos por Ano e Mês')

dados_anomes.to_excel(f'{current_directory}/graf_dados_anomes.xlsx')

#%% beneficiarios por plano

subset = planos_indicadores

# Variáveis
x = subset['seq_tempo']
y = subset['Benef_ativos']

# Criar o gráfico
plt.figure(figsize=(12, 7))

# Lista única de idplanos
unique_idplanos = subset['idplano'].unique()

# Para cada idplano, plotar os pontos e atribuir um rótulo
for idplano in unique_idplanos:
    mask = subset['idplano'] == idplano
    plt.scatter(x[mask], y[mask], label=f'idplano {idplano}')

# Adicionar rótulos, título e legenda
plt.xlabel('Tempo', size=18)
plt.ylabel('Beneficiários', size=18)
plt.title('Beneficiários ao longo do tempo', size=18)
plt.legend()

# Mostrar o gráfico
plt.show()

planos_indicadores.to_excel(f'{current_directory}/graf_planos_indicadores.xlsx')

#%% apresentando os dados




grafico_linha_barra(resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['ipca'], 'Tempo','IPCA', resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['plano'], 'Beneficiários', 'IPCA vs Plano')
grafico_linha_barra(resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['txdesemprego'], 'Tempo','Taxa Desemprego', resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['plano'], 'Beneficiários', 'Taxa Desemprego vs Plano')
grafico_linha_barra(resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['vegetativo'], 'Tempo','crescimento vegetativo', resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['plano'], 'Beneficiários', 'Crescimento vegetativo vs Plano')
grafico_linha_barra(resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['concorrencia'], 'Tempo','Concorrência (%)', resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores['plano'], 'Beneficiários', 'Percentual de Concorrência vs Plano')


#%%
# Calcular a matriz de correlação


df = resultados_planos_indicadores[['ipca',  'txdesemprego','vegetativo','concorrencia' ]]
correlation_matrix = df.corr()

# Criar um heatmap usando Seaborn
plt.figure(figsize=(8, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='gray', vmin=-1, vmax=1,  cbar=False)
plt.title('Matriz de Correlação dos Indicadores Socieconômicos')
plt.show()

# %% Apresentando a correlação Geral


df = resultados_planos_indicadores[[plano_analise, 'seq_tempo','txdesemprego','ipca', 'concorrencia','vegetativo']]

# Calcula a matriz de correlação}
corr_matrix = df.corr()

# Cria um heatmap para visualizar a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,  cbar=False)
plt.title("Matriz de Correlação de Todos os Planos com as variáveis", size=18)
plt.show()


#%% Preparando dados do plano para modelos

colunas_todas = [plano_analise,'seq_tempo','txdesemprego','ipca','concorrencia','vegetativo']
coluna_determinante_plano, colunas_explicativas_plano = [plano_analise],['seq_tempo', 'txdesemprego','ipca','vegetativo', 'concorrencia']
## Formulas iniciais dos planos
scaler_explicativas_plano, scaler_determinante_plano  = MinMaxScaler(),MinMaxScaler()
i= int(len(dados)*perc_amostra)  # indice amostra treino e teste
dados_explicativas_plano, dados_determinante_plano = dados[colunas_explicativas_plano],dados[coluna_determinante_plano]
dados_plano = pd.concat([dados_determinante_plano, dados_explicativas_plano], axis=1)
dados_explicativas_treino_plano, dados_determinante_treino_plano = dados_explicativas_plano.iloc[:i+1],dados_determinante_plano.iloc[:i+1]
dados_explicativas_teste_plano, dados_determinante_teste_plano = dados_explicativas_plano.iloc[i+1:],dados_determinante_plano.iloc[i+1:]
dados_treino_plano, dados_teste_plano = dados_plano.iloc[:i+1],dados_plano.iloc[i+1:]
dados_padronizados_explicativas_plano = scaler_explicativas_plano.fit_transform(resultados_planos_indicadores[colunas_explicativas_plano])
dados_padronizados_determinante_plano = scaler_determinante_plano.fit_transform(resultados_planos_indicadores[coluna_determinante_plano])
dados_padronizados_explicativas_plano = pd.DataFrame(dados_padronizados_explicativas_plano)
dados_padronizados_determinante_plano = pd.DataFrame(dados_padronizados_determinante_plano)
dados_padronizados_explicativas_plano.columns = colunas_explicativas_plano
dados_padronizados_determinante_plano.columns = coluna_determinante_plano
dados_padronizados_plano = pd.concat([dados_padronizados_determinante_plano, dados_padronizados_explicativas_plano], axis=1)
dados_padronizados_treino_plano, dados_padronizados_teste_plano = dados_padronizados_plano.iloc[:i+1], dados_padronizados_plano.iloc[i+1:] 
dados_padronizados_treino_plano, dados_padronizados_teste_plano = sm.add_constant(dados_padronizados_treino_plano), sm.add_constant(dados_padronizados_teste_plano)
dados_padronizados_explicativas_treino_plano, dados_padronizados_determinante_treino_plano = dados_padronizados_explicativas_plano.iloc[:i+1], dados_padronizados_determinante_plano.iloc[:i+1] 
dados_padronizados_explicativas_teste_plano, dados_padronizados_determinante_teste_plano = dados_padronizados_explicativas_plano.iloc[i+1:], dados_padronizados_determinante_plano.iloc[i+1:] 
dados_padronizados_explicativas_treino_plano = sm.add_constant(dados_padronizados_explicativas_treino_plano)
dados_padronizados_explicativas_teste_plano = sm.add_constant(dados_padronizados_explicativas_teste_plano)


# %% Apresentando a correlação dos plano



gcorrplano()



# %% Criando o gráfico plotando todas as informaçõe padronizadas

plano = plano_analise
df = dados_padronizados

plt.figure(figsize=(26, 16))
# Plotando os dados originais
plt.plot(resultados_planos_indicadores['seq_tempo'], resultados_planos_indicadores[plano], 'o-',  markersize=5, linewidth=8, label=plano, color='red')
plt.xlabel('seq_tempo')
plt.ylabel('Valor')
plt.title('Gráfico')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(26, 16))
# Plotando os dados originais
plt.plot(df['seq_tempo'], df[plano], 'o-',  markersize=5, linewidth=8, label=plano, color='red')
plt.plot(df['seq_tempo'], df['concorrencia'], 'o-',markersize=1, linewidth=2, label='Concorrência', color='black')
plt.plot(df['seq_tempo'], df['vegetativo'], 'o-',markersize=1, linewidth=2, label='vegetativo', color='orange')
plt.xlabel('seq_tempo')
plt.ylabel('Valor')
plt.title('Gráfico')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(26, 16))
# Plotando os dados originais
plt.plot(df['seq_tempo'], df[plano], 'o-',  markersize=5, linewidth=8, label=plano, color='red')
plt.plot(df['seq_tempo'], df['txdesemprego'], 'o-', markersize=1, linewidth=2,label='txdesemprego', color='blue')
plt.plot(df['seq_tempo'], df['concorrencia'], 'o-',markersize=1, linewidth=2, label='% Concorrência', color='black')
plt.xlabel('seq_tempo')
plt.ylabel('Valor')
plt.title('Gráfico')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(26, 16))
# Plotando os dados originais
plt.plot(df['seq_tempo'], df[plano], 'o-',  markersize=5, linewidth=8, label=plano, color='red')
plt.plot(df['seq_tempo'], df['ipca'], 'o-',markersize=1, linewidth=2, label='ipca', color='green')
plt.xlabel('seq_tempo')
plt.ylabel('Valor')
plt.title('Gráfico')
plt.legend()
plt.grid(True)
plt.show()

# %% Análise de regressão linear do plano
np.random.seed(0)

#plano

formula_plano = f"{plano_analise} ~ txdesemprego + ipca   + vegetativo + concorrencia"
modelo_linear_plano = sm.OLS.from_formula(formula_plano, dados_padronizados_treino_plano).fit()
print(modelo_linear_plano.summary())

formula_plano = f"{plano_analise} ~ txdesemprego + ipca + concorrencia"
print('Fórmula final:', formula_plano)
modelo_linear_plano = sm.OLS.from_formula(formula_plano, dados_padronizados_treino_plano).fit()
print(modelo_linear_plano.summary())
verifica_autocorrelacao(modelo_linear_plano.resid)

# Calculando VIF para cada variável independente
vif_data = pd.DataFrame()
vif_data["Variável"] = modelo_linear_plano.model.exog_names[1:]  # Excluindo o intercepto
vif_data["VIF"] = [variance_inflation_factor(modelo_linear_plano.model.exog, i) 
                   for i in range(1, modelo_linear_plano.model.exog.shape[1])]  # Começando de 1 para excluir o intercepto

print(vif_data)


# Execute o teste de Breusch-Pagan
bp_test = het_breuschpagan(modelo_linear_plano.resid, modelo_linear_plano.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))
##Se o valor-p do teste for significativamente baixo (por exemplo, menor que 0,05), 
##você rejeitaria a hipótese nula de homocedasticidade, indicando a presença de heterocedasticidade nos dados.


### em caso de Caso de autocorrelação criar matriz de poderação 
'''
residuos_quadrados = modelo_linear_plano.resid ** 2
# 3. Estime a variância dos erros (aqui, usamos uma média móvel para suavizar os resíduos quadrados)
# Isso é apenas um exemplo; outras técnicas podem ser usadas para estimar a variância
variancia_estimada = residuos_quadrados.rolling(window=3).mean()

# 4. Crie uma matriz de ponderação
# A matriz de ponderação é a inversa da variância estimada
W = 1 / variancia_estimada
W = W.fillna(1)

# 5. Ajuste um modelo GLS usando a matriz de ponderação
#modelo_linear_plano = sm.GLSAR.from_formula(formula_plano, dados_padronizados_treino_plano, rho=1, weights=W).fit()
#modelo_linear_plano = sm.GLSAR.from_formula(formula_plano, dados_padronizados_treino_plano, rho=1).fit()
#verifica_autocorrelacao(modelo_linear_plano.resid)
#print(modelo_linear_plano.summary())
'''

dados_padronizados_treino_plano['yhat_lin'] = modelo_linear_plano.predict()
dados_padronizados_teste_plano['yhat_lin'] = modelo_linear_plano.predict(dados_padronizados_explicativas_teste_plano)
dados_padronizados_treino_plano['erro_lin'] = modelo_linear_plano.resid
dados_padronizados_teste_plano['erro_lin'] = (dados_padronizados_teste_plano[plano_analise] - dados_padronizados_teste_plano['yhat_lin'])
comp = pd.concat([dados_padronizados_treino_plano[['seq_tempo', plano_analise, 'yhat_lin','erro_lin']], dados_padronizados_teste_plano[['seq_tempo', plano_analise, 'yhat_lin','erro_lin']]], axis=0)
dados_padronizados_plano['yhat_lin'] = modelo_linear_plano.predict(dados_padronizados_plano)
dados_padronizados_plano['erro_lin'] = (dados_padronizados_plano[plano_analise] - dados_padronizados_plano['yhat_lin'])
comp = dados_padronizados_plano[['seq_tempo', plano_analise, 'yhat_lin','erro_lin']]
plt.figure(figsize=(26, 16))
# Plotando os dados originais
plt.plot(comp['seq_tempo'], comp[plano_analise], 'o-',  markersize=3, linewidth=3, label=plano_analise, color='black')
plt.plot(comp['seq_tempo'], comp['yhat_lin'], 'o--', markersize=3, linewidth=4,label='yhat_lin', color='gray')
plt.xlabel('Tempo', size=24)
plt.ylabel('Beneficiários', size=24)
plt.title('Regressão Linear [OLS] (Treino e Teste)', size=44)
plt.tick_params(axis='both', labelsize=24) # Aumentar a fonte dos ticks dos eixos
plt.legend(prop={'size': 34})
plt.grid(True)
plt.axvspan(perc_amostra, comp['seq_tempo'].max(), facecolor='lightgray', alpha=0.5)
plt.text(0.3, 0.9, 'Treino', fontsize=40, color='black')
plt.text(0.85, 0.9, 'Teste', fontsize=40, color='black')
plt.show()

cria_hostograma_normalidade(dados_padronizados_treino_plano['erro_lin'], 'Erros Regressão Linear - Treino')
cria_hostograma_normalidade(dados_padronizados_treino_plano['erro_lin'], 'Erros Regressão Linear - Teste')

print("R-quadrado de treino: " + str(r2_score(dados_padronizados_treino_plano[plano_analise], modelo_linear_plano.predict(dados_padronizados_treino_plano))))
print("R-quadrado de teste: " + str(r2_score(dados_padronizados_teste_plano[plano_analise],  modelo_linear_plano.predict(dados_padronizados_teste_plano))))
print("R-quadrado geral : " + str(r2_score(dados_padronizados_plano[plano_analise], modelo_linear_plano.predict(dados_padronizados_plano))))

aval_linear_treino = avaliacao('Modelo Linear plano',dados_padronizados_treino_plano[plano_analise],modelo_linear_plano.predict(dados_padronizados_treino_plano))
aval_linear_teste = avaliacao('Modelo Linear plano',dados_padronizados_teste_plano[plano_analise],modelo_linear_plano.predict(dados_padronizados_teste_plano))


dados_padronizados_plano.to_excel(f'{current_directory}/graf_dados padronizados_linear.xlsx')

#exibindo equação:
# Acessando os coeficientes
intercepto = modelo_linear_plano.params[0]
coef_txdesemprego = modelo_linear_plano.params[1]
coef_ipca = modelo_linear_plano.params[2]
coef_concorrencia = modelo_linear_plano.params[3]

# Formatando a equação linear
equacao = f"{plano_analise} = {intercepto:.2f} + {coef_txdesemprego:.2f}*txdesemprego + {coef_ipca:.2f}*ipca + {coef_concorrencia:.2f}*concorrencia"

print(equacao)
print(modelo_linear_plano.summary())    


#%% verificando a sazonalidade


    
decompor_sazonalidade(dados_padronizados_plano[[plano_analise]], 12)



#%% buscando melhor modelo com base no R2 de teste - Executar somente a primeira vez para enconcontrar parametros que tradezem um melhor 2 de testes

ii = len(dados_padronizados_explicativas_teste_plano)-1
colunas_explicativas_plano_sar = ['txdesemprego', 'ipca', 'concorrencia','vegetativo']

           
param_grid = {

    'p': [0,1,2,4,5,6,7,8,9,10],
    'd': [1],
    'q': [0,1,2,3,4],
    'P': [0,1,2,4,5,6,7,8,9,10],
    'D': [0],
    'Q': [0,1,2,3,4],
    's': [12]
}

#buscando melhor modelo com base no R2 de teste
R2, p, d, q, P, D, Q, s,df_gridserash_sarima = busca_melhor_sarimax(dados_padronizados_explicativas_teste_plano[colunas_explicativas_plano_sar],dados_padronizados_determinante_teste_plano, dados_padronizados_determinante_treino_plano, dados_padronizados_explicativas_treino_plano[colunas_explicativas_plano_sar], param_grid)
#Usando melhor SARIMA para plano
print(f"Melhores R2: {R2}")
print(f"Melhores parâmetros: {p, d, q }")
print(f"Melhores parâmetros sazonais: {P, D, Q, s}")

# grava resultados em arquivo excel de apoio
df_gridserash_sarima.to_excel(f'{current_directory}\gridsearsh_sarima.xlsx', index=False, engine='openpyxl')




#%% aplicando modelo com as melhores variáveis

#Otimo 53%
p, d, q, P, D, Q,s=6,	1,	0,	1,	0,	0,	12

ii = len(dados_padronizados_explicativas_teste_plano)-1
colunas_explicativas_plano_sar = ['txdesemprego', 'ipca', 'concorrencia','vegetativo']

modelo_sarimax_plano = sm.tsa.statespace.SARIMAX(dados_padronizados_determinante_treino_plano, 
                                  exog=dados_padronizados_explicativas_treino_plano[colunas_explicativas_plano_sar], 
                                  order=(p,d,q), 
                                  seasonal_order=(P,D,Q,s), 
                                  enforce_stationarity=False, 
                                  enforce_invertibility=False)

modelo_sarimax_plano = modelo_sarimax_plano.fit(disp=False)

print(modelo_sarimax_plano.summary())
# Previsão para os próximos 12 pontos de dados, por exemplo
forecast = modelo_sarimax_plano.predict(start=len(dados_padronizados_explicativas_treino_plano), end=len(dados_padronizados_explicativas_treino_plano)+ii, exog=dados_padronizados_explicativas_teste_plano[colunas_explicativas_plano_sar])

dados_padronizados_treino_plano['yhat_sar'] = modelo_sarimax_plano.fittedvalues
dados_padronizados_teste_plano['yhat_sar'] = forecast.copy()
dados_padronizados_treino_plano['erro_sar'] = modelo_sarimax_plano.resid
dados_padronizados_teste_plano['erro_sar'] = (dados_padronizados_teste_plano[plano_analise] - dados_padronizados_teste_plano['yhat_sar'])
comp = pd.concat([dados_padronizados_treino_plano[['seq_tempo', plano_analise, 'yhat_sar','erro_sar']], dados_padronizados_teste_plano[['seq_tempo', plano_analise, 'yhat_sar','erro_sar']]], axis=0)
dados_padronizados_plano['yhat_sar'] = comp[['yhat_sar']]
dados_padronizados_plano['erro_sar'] = (dados_padronizados_plano[plano_analise] - dados_padronizados_plano['yhat_sar'])

comp_sarima = comp[comp['seq_tempo']>= 0.2] 
plt.figure(figsize=(26, 16))
# Plotando os dados originais
plt.plot(comp_sarima['seq_tempo'], comp_sarima[plano_analise], 'o-',  markersize=3, linewidth=3, label=plano_analise, color='black')
plt.plot(comp_sarima['seq_tempo'], comp_sarima['yhat_sar'], 'o--', markersize=3, linewidth=4,label='yhat_sar', color='gray')
plt.xlabel('Tempo', size=24)
plt.ylabel('Beneficiários', size=24)
plt.title('Modelo SARIMA (Treino e Teste)', size=44)
plt.tick_params(axis='both', labelsize=24) # Aumentar a fonte dos ticks dos eixos
plt.legend(prop={'size': 34})
plt.grid(True)
plt.axvspan(perc_amostra, comp_sarima['seq_tempo'].max(), facecolor='lightgray', alpha=0.5)
plt.text(0.3, 0.9, 'Treino', fontsize=40, color='black')
plt.text(0.85, 0.9, 'Teste', fontsize=40, color='black')
plt.show()

verifica_autocorrelacao(dados_padronizados_treino_plano['erro_sar'])

cria_hostograma_normalidade(dados_padronizados_treino_plano['erro_sar'], 'Erros Regressão Linear - treino')
cria_hostograma_normalidade(dados_padronizados_teste_plano['erro_sar'], 'Erros Regressão Linear - teste')

print("R-quadrado de treino: " + str(r2_score(dados_padronizados_treino_plano[dados_padronizados_treino_plano['seq_tempo']>= 0.2][plano_analise], dados_padronizados_treino_plano[dados_padronizados_treino_plano['seq_tempo']>= 0.2]['yhat_sar'])))
print("R-quadrado de teste: " + str(r2_score(dados_padronizados_teste_plano[plano_analise], dados_padronizados_teste_plano['yhat_sar'])))
print("R-quadrado geral : " + str(r2_score(comp_sarima[plano_analise], comp_sarima[['yhat_sar']])))

aval_sarima_treino = avaliacao('Modelo Linear plano',dados_padronizados_treino_plano[dados_padronizados_treino_plano['seq_tempo']>= 0.2][plano_analise],dados_padronizados_treino_plano[dados_padronizados_treino_plano['seq_tempo']>= 0.2]['yhat_sar'])
aval_sarima_teste = avaliacao('Modelo Linear plano',dados_padronizados_teste_plano[plano_analise],dados_padronizados_teste_plano['yhat_sar'])


dados_padronizados_plano.to_excel(f'{current_directory}/graf_dados padronizados.xlsx')


# %% Escolha do melhor modelo plano - exectar somente 1 vez para encontrar melhores hiperparametros e arquitetura para encontrar melhor r2


Vtimesteps =6
np.random.seed(0)
tf.random.set_seed(0)

colunas_rnn_plano = [plano_analise,'txdesemprego','ipca','concorrencia', 'vegetativo']
df_final = dados_padronizados_plano[colunas_rnn_plano].copy()



param_grid = {
    'duascamadas': [True],
    'units': [5, 10, 50],
    'learning_rate': [0.1, 0.01, 0.001],
    'drop_out': [0.1,0.2, 0.3],
    'drop_out2': [0.1,0.2, 0.3],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [8, 16, 32],
    'epochs': [100,200, 400],
    'steps': [6, 9]
}

# a função salva o melhor encontrado em um arquivo com extensão .h5
best_model_plano, Vtimesteps, df_gridserach_Rnn = grid_search_melhor_rnn(param=param_grid, df_avaliacao=df_final, percent_amostra_teste=1-perc_amostra, var_dependente=plano_analise, min_r2=0.3)
best_model_plano.summary()

# arnazenando resultados em arquivo
df_gridserach_Rnn.to_excel(f'{current_directory}\gridsearch_Rnn.xlsx', index=False, engine='openpyxl')





#%% Testando o mesmo modelo mais de uma vez um configuração - usar quando necessário
'''
Vtimesteps = 5
np.random.seed(0)
tf.random.set_seed(0)

colunas_rnn_plano = [plano_analise,'txdesemprego','ipca','concorrencia', 'vegetativo']
df_final = dados_padronizados_plano[colunas_rnn_plano].copy()

r2 = -9999999
while 1==1:  
    df = df_final.copy()
    np.random.seed(0)
    tf.random.set_seed(0)
    X, y = structure_data(df,plano_analise, timesteps=Vtimesteps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-perc_amostra, shuffle=False)
    shape = (df.shape[1]-1)
    best_model_plano = criar_modelo_rnn(steps=5, shape=shape, units=25,drop_out=0.2, optimizer='adam',learning_rate=0.01,epochs=400, batch_size=5, X_train=X_train,y_train=y_train, duascamadas=True, drop_out2=0.1)
    a_treino = best_model_plano.predict(X_train, verbose=0)
    a_teste = best_model_plano.predict(X_test, verbose=0)
    r_treino = r2_score(y_train,a_treino)
    r_teste = r2_score(y_test, a_teste)
    print(f"R-quadrado de teste: {r_teste}. R2 de treino: {r_treino}")
    if r_teste >= r2:
        r2 = r_teste
        print(f"***********Melhor R-quadrado de teste encontrado: {r_teste}. R2 de treino: {r_treino}")
        best_model_plano.save(f'{current_directory}/rnn_melhor_modelo_plano.h5')
'''

# %% Previsões plano RNN 

##***********Melhor R-quadrado de teste encontrado: 0.39056479803199995. R2 de treino: 0.9083468374824983
##best_model_plano = criar_modelo_rnn(steps=5, shape=shape, units=25,drop_out=0.2, optimizer='adam',learning_rate=0.01,epochs=400, batch_size=5, X_train=X_train,y_train=y_train, duascamadas=True, drop_out2=0.1)

colunas_rnn_plano = [plano_analise,'txdesemprego','ipca','concorrencia', 'vegetativo']
df_final = dados_padronizados_plano[colunas_rnn_plano].copy()

Vtimesteps = 5

np.random.seed(0)
tf.random.set_seed(0)

df = df_final.copy()
X, y = structure_data(df,plano_analise,timesteps=Vtimesteps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-perc_amostra, shuffle=False)

best_model_plano = load_model(f'{current_directory}/rnn_melhor_modelo_plano.h5')
print(best_model_plano)

df_scaled = df_final.copy()

yhat =  np.concatenate((df_scaled[plano_analise].iloc[:Vtimesteps].values.reshape(-1, 1),best_model_plano.predict(X)), axis=0)
df_scaled['yhat'] = yhat
df_scaled['yhat'].fillna(df_scaled[plano_analise], inplace=True)
dados_padronizados_plano['yhat_rnn'] = df_scaled['yhat']
dados_padronizados_plano['erro_rnn'] = (dados_padronizados_plano[plano_analise] - dados_padronizados_plano['yhat_rnn'])
dados_padronizados_treino_plano['yhat_rnn'] =  dados_padronizados_plano['yhat_rnn'].iloc[:int(len(df) * perc_amostra)+1]
dados_padronizados_teste_plano['yhat_rnn'] =  dados_padronizados_plano['yhat_rnn'].iloc[int(len(df) * perc_amostra)+1:]
dados_padronizados_treino_plano['erro_rnn'] =  dados_padronizados_plano['erro_rnn'].iloc[:int(len(df) * perc_amostra)+1]
dados_padronizados_teste_plano['erro_rnn'] =  dados_padronizados_plano['erro_rnn'].iloc[int(len(df) * perc_amostra)+1:]
comp = dados_padronizados_plano[['seq_tempo', plano_analise, 'yhat_rnn','erro_rnn']]

plt.figure(figsize=(26, 16))
# Plotando os dados originais
plt.plot(comp['seq_tempo'], comp[plano_analise], 'o-',  markersize=3, linewidth=3, label=plano_analise, color='black')
plt.plot(comp['seq_tempo'], comp['yhat_rnn'], 'o--', markersize=3, linewidth=4,label='yhat_rnn', color='gray')
plt.xlabel('Tempo', size=24)
plt.ylabel('Beneficiários', size=24)
plt.title('Modelo RNN (Treino e Teste)', size=44)
plt.tick_params(axis='both', labelsize=24) # Aumentar a fonte dos ticks dos eixos
plt.legend(prop={'size': 34})
plt.grid(True)
plt.axvspan(perc_amostra, comp['seq_tempo'].max(), facecolor='lightgray', alpha=0.5)
plt.text(0.3, 0.9, 'Treino', fontsize=40, color='black')
plt.text(0.85, 0.9, 'Teste', fontsize=40, color='black')
plt.show()


#avaliacao('Modelo RNN plano',dados_padronizados_teste_plano[plano_analise],dados_padronizados_teste_plano['yhat_box'])

cria_hostograma_normalidade(dados_padronizados_treino_plano['erro_rnn'], 'Erros RNN - treino')
cria_hostograma_normalidade(dados_padronizados_teste_plano['erro_rnn'], 'Erros RNN Linear - teste')


a_treino = best_model_plano.predict(X_train)
a_teste = best_model_plano.predict(X_test)
a_total = np.concatenate((a_treino,a_teste), axis=0) 

print("R-quadrado de treino: " + str(r2_score(y_train,a_treino)))
print("R-quadrado de teste: " + str(r2_score(y_test, a_teste)))
print("R-quadrado geral : " +str(r2_score(y, a_total)))

dados_padronizados_plano.to_excel(f'{current_directory}/graf_dados padronizados.xlsx')



# %% Armazena em dataframes o resultados dos modelos

Resultados_comparativos_treino = pd.concat([
                                     avaliacao('Modelo Linear plano',dados_padronizados_treino_plano[plano_analise],modelo_linear_plano.predict(dados_padronizados_treino_plano))
                                    ,avaliacao('Modelo SARIMA Plano3',dados_padronizados_treino_plano[dados_padronizados_treino_plano['seq_tempo']>= 0.2][plano_analise], dados_padronizados_treino_plano[dados_padronizados_treino_plano['seq_tempo']>= 0.2]['yhat_sar'])
                                    ,avaliacao('Modelo RNN Plano3',pd.DataFrame(y_train).iloc[:, 0], pd.DataFrame(a_treino).iloc[:, 0])
                                    ], ignore_index=True)

Resultados_comparativos_teste = pd.concat([
                                     avaliacao('Modelo Linear plano',dados_padronizados_teste_plano[plano_analise],modelo_linear_plano.predict(dados_padronizados_teste_plano))
                                    ,avaliacao('Modelo SARIMA plano',dados_padronizados_teste_plano[plano_analise], dados_padronizados_teste_plano['yhat_sar'])
                                    ,avaliacao('Modelo RNN Plano3',pd.DataFrame(y_test).iloc[:, 0], pd.DataFrame(a_teste).iloc[:, 0])
                                    ], ignore_index=True)

Resultados_comparativos_geral = pd.concat([
                                     avaliacao('Modelo Linear plano',dados_padronizados_plano[plano_analise], modelo_linear_plano.predict(dados_padronizados_plano))
                                    ,avaliacao('Modelo SARIMA plano',comp_sarima[plano_analise], comp_sarima['yhat_sar'])
                                    ,avaliacao('Modelo RNN Plano3',pd.DataFrame(y).iloc[:, 0], pd.DataFrame(a_total).iloc[:, 0])
                                    ], ignore_index=True)



# %% treinando os modelos para toda a base com Regressão Liner
np.random.seed(0)
#plano
modelo_linear_plano = sm.OLS.from_formula(formula_plano, dados_padronizados_plano).fit()
dados_padronizados_plano['yhat_sar'] = modelo_linear_plano.predict()
dados_padronizados_plano['erro_sar'] = (dados_padronizados_plano[plano_analise] - dados_padronizados_plano['yhat_sar'])

#%% treinando Sarimax com toda a base
#adicionando dados de teste ao SARIMA
#modelo_sarimax_plano = modelo_sarimax_plano.append(dados_padronizados_determinante_teste_plano, exog=dados_padronizados_explicativas_teste_plano[colunas_explicativas_plano_sar], refit=True)
p, d, q, P, D, Q,s=6,	1,	0,	1,	0,	0,	12

modelo_sarimax_plano = sm.tsa.statespace.SARIMAX(dados_padronizados_determinante_plano, 
                                  exog=dados_padronizados_explicativas_plano[colunas_explicativas_plano_sar], 
                                  order=(p,d,q), 
                                  seasonal_order=(P,D,Q,s), 
                                  enforce_stationarity=False, 
                                  enforce_invertibility=False)

modelo_sarimax_plano = modelo_sarimax_plano.fit(disp=False)
print(modelo_sarimax_plano.summary())


dados_padronizados_plano['yhat_sar'] = modelo_sarimax_plano.fittedvalues
dados_padronizados_plano['erro_sar'] = (dados_padronizados_plano[plano_analise] - dados_padronizados_plano['yhat_sar'])

# %% treinando os modelos para toda a base com rnn


Vtimesteps = 5

np.random.seed(0)
tf.random.set_seed(0)

#plano
df =  dados_padronizados_plano[colunas_rnn_plano].copy()
X, y = structure_data(df,plano_analise,Vtimesteps)
best_model_plano.fit(X,y, epochs=400, batch_size=5, verbose=0)

yhat = predict_rnn(df, plano_analise, Vtimesteps, best_model_plano)
df['yhat'] = yhat
df['yhat'].fillna(df[plano_analise], inplace=True)
dados_padronizados_plano['yhat_rnn'] = df['yhat']
dados_padronizados_plano['erro_rnn'] = (dados_padronizados_plano[plano_analise] - dados_padronizados_plano['yhat_rnn'])
#salva o modelotreinando em um novo arquivo .h5
best_model_plano.save(f'{current_directory}/rnn_melhor_modelo_plano_toda_base.h5')


#%% Resultado comparativo com treino de toda a base

Resultados_comparativos_treino_toda_base = pd.concat([
                                     avaliacao('Modelo Linear plano',dados_padronizados_plano[plano_analise],dados_padronizados_plano['yhat_lin'])
                                    ,avaliacao('Modelo SARIMA Plano3',dados_padronizados_plano[dados_padronizados_plano['seq_tempo']>= 0.2][plano_analise],dados_padronizados_plano[dados_padronizados_plano['seq_tempo']>= 0.2]['yhat_sar'])
                                    ,avaliacao('Modelo RNN Plano3',dados_padronizados_plano[plano_analise],dados_padronizados_plano['yhat_rnn'])
                                    ], ignore_index=True)


# %% lendo arquivos de simulações criados manualmente simulações de dataframe de taxa de desemprego

df_sim_txdesemprego = pd.read_excel(f'{current_directory}/simulacoes/sim_txdesemprego_sobe.xlsx')
df_sim_ipca = pd.read_excel(f'{current_directory}/simulacoes/sim_ipca_sobe.xlsx')
df_sim_vegetativo = pd.read_excel(f'{current_directory}/simulacoes/sim_vegetativo_sobe.xlsx')
df_sim_concorrencia = pd.read_excel(f'{current_directory}/simulacoes/sim_concorrencia_sobe.xlsx')



# %% Prediões a partir das bases simuladas

##### OLS
res_sim_txdesemprego_linear = modelo_linear_plano.predict(df_sim_txdesemprego[df_sim_txdesemprego["Tipo"]=="Simulação"]).to_frame(name='yhat_lin')
res_sim_txdesemprego_linear["Tipo"] = "Predição"
res_sim_ipca_linear = modelo_linear_plano.predict(df_sim_ipca[df_sim_ipca["Tipo"]=="Simulação"]).to_frame(name='yhat_lin')
res_sim_ipca_linear["Tipo"] = "Predição"
res_sim_vegetativo_linear = modelo_linear_plano.predict(df_sim_vegetativo[df_sim_vegetativo["Tipo"]=="Simulação"]).to_frame(name='yhat_lin')
res_sim_vegetativo_linear["Tipo"] = "Predição"
res_sim_concorrencia_linear = modelo_linear_plano.predict(df_sim_concorrencia[df_sim_concorrencia["Tipo"]=="Simulação"]).to_frame(name='yhat_lin')
res_sim_concorrencia_linear["Tipo"] = "Predição"

##### SARIMA
vstart = len(dados_padronizados_explicativas_plano)
vend = vstart +10-1
res_sim_txdesemprego_sarimax =  modelo_sarimax_plano.predict(start=vstart, end=vend, exog=df_sim_txdesemprego[df_sim_txdesemprego["Tipo"]=="Simulação"][colunas_explicativas_plano_sar]).to_frame(name='yhat_sar')
res_sim_txdesemprego_sarimax["Tipo"] = "Predição"
res_sim_ipca_sarimax = modelo_sarimax_plano.predict(start=vstart, end=vend, exog=df_sim_ipca[df_sim_ipca["Tipo"]=="Simulação"][colunas_explicativas_plano_sar]).to_frame(name='yhat_sar')
res_sim_ipca_sarimax["Tipo"] = "Predição"
res_sim_vegetativo_sarimax = modelo_sarimax_plano.predict(start=vstart, end=vend, exog=df_sim_vegetativo[df_sim_vegetativo["Tipo"]=="Simulação"][colunas_explicativas_plano_sar]).to_frame(name='yhat_sar')
res_sim_vegetativo_sarimax["Tipo"] = "Predição"
res_sim_concorrencia_sarimax = modelo_sarimax_plano.predict(start=vstart, end=vend, exog=df_sim_concorrencia[df_sim_concorrencia["Tipo"]=="Simulação"][colunas_explicativas_plano_sar]).to_frame(name='yhat_sar')
res_sim_concorrencia_sarimax["Tipo"] = "Predição"

##### RNN
best_model_plano = load_model(f'{current_directory}/rnn_melhor_modelo_plano_toda_base.h5')
print(best_model_plano)
inicio_coleta_rnn = vstart = len(dados_padronizados_explicativas_plano)- Vtimesteps
fim_coleta_rnn =  len(dados_padronizados_explicativas_plano)+ 10 

df = df_sim_txdesemprego.iloc[inicio_coleta_rnn:fim_coleta_rnn][colunas_rnn_plano]
X, y = structure_data(df,plano_analise,timesteps=Vtimesteps)
res_sim_txdesemprego_RNN =pd.Series(best_model_plano.predict(X, verbose=0).flatten()).to_frame(name='yhat_rnn')
res_sim_txdesemprego_RNN["Tipo"] = "Predição"
res_sim_txdesemprego_RNN.index = res_sim_txdesemprego_RNN.index + len(dados_padronizados_explicativas_plano)

df = df_sim_ipca.iloc[inicio_coleta_rnn:fim_coleta_rnn][colunas_rnn_plano]
X, y = structure_data(df,plano_analise,timesteps=Vtimesteps)
res_sim_ipca_RNN =pd.Series(best_model_plano.predict(X, verbose=0).flatten()).to_frame(name='yhat_rnn')
res_sim_ipca_RNN["Tipo"] = "Predição"
res_sim_ipca_RNN.index = res_sim_ipca_RNN.index + len(dados_padronizados_explicativas_plano)

df = df_sim_vegetativo.iloc[inicio_coleta_rnn:fim_coleta_rnn][colunas_rnn_plano]
X, y = structure_data(df,plano_analise,timesteps=Vtimesteps)
res_sim_vegetativo_RNN =pd.Series(best_model_plano.predict(X, verbose=0).flatten()).to_frame(name='yhat_rnn')
res_sim_vegetativo_RNN["Tipo"] = "Predição"
res_sim_vegetativo_RNN.index = res_sim_vegetativo_RNN.index + len(dados_padronizados_explicativas_plano)

df = df_sim_concorrencia.iloc[inicio_coleta_rnn:fim_coleta_rnn][colunas_rnn_plano]
X, y = structure_data(df,plano_analise,timesteps=Vtimesteps)
res_sim_concorrencia_RNN =pd.Series(best_model_plano.predict(X, verbose=0).flatten()).to_frame(name='yhat_rnn')
res_sim_concorrencia_RNN["Tipo"] = "Predição"
res_sim_concorrencia_RNN.index = res_sim_concorrencia_RNN.index + len(dados_padronizados_explicativas_plano)

sim_txdesemprego = pd.concat([dados_padronizados_plano[plano_analise], res_sim_txdesemprego_linear["yhat_lin"],res_sim_txdesemprego_sarimax["yhat_sar"],res_sim_txdesemprego_RNN["yhat_rnn"]], axis=1)
sim_ipca = pd.concat([dados_padronizados_plano[plano_analise], res_sim_ipca_linear["yhat_lin"],res_sim_ipca_sarimax["yhat_sar"],res_sim_ipca_RNN["yhat_rnn"]], axis=1)
sim_vegetativo = pd.concat([dados_padronizados_plano[plano_analise], res_sim_vegetativo_linear["yhat_lin"],res_sim_vegetativo_sarimax["yhat_sar"],res_sim_vegetativo_RNN["yhat_rnn"]], axis=1)
sim_concorrencia = pd.concat([dados_padronizados_plano[plano_analise], res_sim_concorrencia_linear["yhat_lin"],res_sim_concorrencia_sarimax["yhat_sar"],res_sim_concorrencia_RNN["yhat_rnn"]], axis=1)

sim_txdesemprego[plano_analise] = scaler_determinante_plano.inverse_transform(sim_txdesemprego[[plano_analise]])
sim_txdesemprego["yhat_lin"] = scaler_determinante_plano.inverse_transform(sim_txdesemprego[["yhat_lin"]])
sim_txdesemprego["yhat_sar"] = scaler_determinante_plano.inverse_transform(sim_txdesemprego[["yhat_sar"]])
sim_txdesemprego["yhat_rnn"] = scaler_determinante_plano.inverse_transform(sim_txdesemprego[["yhat_rnn"]])

sim_ipca[plano_analise] = scaler_determinante_plano.inverse_transform(sim_ipca[[plano_analise]])
sim_ipca["yhat_lin"] = scaler_determinante_plano.inverse_transform(sim_ipca[["yhat_lin"]])
sim_ipca["yhat_sar"] = scaler_determinante_plano.inverse_transform(sim_ipca[["yhat_sar"]])
sim_ipca["yhat_rnn"] = scaler_determinante_plano.inverse_transform(sim_ipca[["yhat_rnn"]])

sim_vegetativo[plano_analise] = scaler_determinante_plano.inverse_transform(sim_vegetativo[[plano_analise]])
sim_vegetativo["yhat_lin"] = scaler_determinante_plano.inverse_transform(sim_vegetativo[["yhat_lin"]])
sim_vegetativo["yhat_sar"] = scaler_determinante_plano.inverse_transform(sim_vegetativo[["yhat_sar"]])
sim_vegetativo["yhat_rnn"] = scaler_determinante_plano.inverse_transform(sim_vegetativo[["yhat_rnn"]])

sim_concorrencia[plano_analise] = scaler_determinante_plano.inverse_transform(sim_concorrencia[[plano_analise]])
sim_concorrencia["yhat_lin"] = scaler_determinante_plano.inverse_transform(sim_concorrencia[["yhat_lin"]])
sim_concorrencia["yhat_sar"] = scaler_determinante_plano.inverse_transform(sim_concorrencia[["yhat_sar"]])
sim_concorrencia["yhat_rnn"] = scaler_determinante_plano.inverse_transform(sim_concorrencia[["yhat_rnn"]])



sim_txdesemprego[colunas_explicativas_plano] = scaler_explicativas_plano.inverse_transform(df_sim_txdesemprego[colunas_explicativas_plano])
sim_ipca[colunas_explicativas_plano] = scaler_explicativas_plano.inverse_transform(df_sim_ipca[colunas_explicativas_plano])
sim_vegetativo[colunas_explicativas_plano] = scaler_explicativas_plano.inverse_transform(df_sim_vegetativo[colunas_explicativas_plano])
sim_concorrencia[colunas_explicativas_plano] = scaler_explicativas_plano.inverse_transform(df_sim_concorrencia[colunas_explicativas_plano])

# gravando resultado das simulações em arquivos
sim_txdesemprego.to_excel(f'{current_directory}/simulacoes/resultado_simulacoes_txdesemprego.xlsx')
sim_ipca.to_excel(f'{current_directory}/simulacoes/resultado_simulacoes_ipca.xlsx')
sim_vegetativo.to_excel(f'{current_directory}/simulacoes/resultado_simulacoes_vegetativo.xlsx')
sim_concorrencia.to_excel(f'{current_directory}/simulacoes/resultado_simulacoes_concorrencia.xlsx')


#%% Simulando projeções

#Análise de tendencia de desemprego

n_periodos= 10

np.random.seed(0)
his_arima_desemprego, df_final_desemprego_arima, his_sarima_desemprego, df_final_desemprego_sarima \
= gera_resultados_tendencia(drx=resultados_socioeconomicos,coluna="txdesemprego",p=1,d=1,q=6,sP=4,sD=0,sQ=0,s=12,vperiodos=n_periodos)



his_arima_ipca, df_final_ipca_arima, his_sarima_ipca, df_final_ipca_sarima \
= gera_resultados_tendencia(drx=resultados_socioeconomicos,coluna="ipca",p=1,d=1,q=6,sP=4,sD=0,sQ=0,s=12,vperiodos=n_periodos)



his_arima_concorrencia, df_final_concorrencias_arima, his_sarima_concorrencia, df_final_concorrencia_sarima \
= gera_resultados_tendencia(drx=dados_plano,coluna="concorrencia",p=1,d=1,q=6,sP=5,sD=0,sQ=0,s=12,vperiodos=n_periodos)

his_arima_vegetativo, df_final_vegetativo_arima, his_sarima_vegetativo, df_final_vegetativo_sarima \
= gera_resultados_tendencia(drx=dados_plano,coluna="vegetativo",p=1,d=1,q=6,sP=0,sD=0,sQ=0,s=12,vperiodos=n_periodos)


# %% montando base do previsto e do realizado
np.random.seed(0)

df_explicativas_final = pd.merge(df_final_desemprego_sarima, df_final_ipca_sarima[['seq_tempo','ipca']], on='seq_tempo', how='inner')
df_explicativas_final = pd.merge(df_explicativas_final, df_final_concorrencia_sarima[['seq_tempo','concorrencia']], on='seq_tempo', how='inner')
df_explicativas_final = pd.merge(df_explicativas_final, df_final_vegetativo_sarima[['seq_tempo','vegetativo']], on='seq_tempo', how='inner')
df_explicativas_final = df_explicativas_final[['seq_tempo', 'txdesemprego','ipca', 'concorrencia','vegetativo', 'modelo']]


# %% montando df_explicativas_final_escalada

df_explicativas_final_escalada = df_explicativas_final.copy()
df_explicativas_final_escalada[colunas_explicativas_plano] = scaler_explicativas_plano.transform(df_explicativas_final_escalada[colunas_explicativas_plano])
df_explicativas_final_escalada[plano_analise] = 0
df_explicativas_final_escalada_plano = df_explicativas_final_escalada.copy()


# %% Predição da base original e da simulada com tendencias para OLS
df_explicativas_final_escalada_plano['yhat_lin'] = modelo_linear_plano.predict(df_explicativas_final_escalada)



# %% predicao da base original e simulada com tendencia para SARIMA

dados_explicativas_sarima_plano = df_explicativas_final_escalada[df_explicativas_final_escalada['modelo'] == 'Predição'][colunas_explicativas_plano_sar]
vstart = len(dados_padronizados_explicativas_plano)
vend = vstart +len(dados_explicativas_sarima_plano)-1
df_explicativas_final_escalada_plano['yhat_sar']  = pd.concat([modelo_sarimax_plano.fittedvalues, modelo_sarimax_plano.predict(start=vstart, end=vend, exog=dados_explicativas_sarima_plano)])
# %% Predição da base original e da simulada com tendencias para RNN
np.random.seed(0)

inicio_coleta_rnn = vstart = len(dados_padronizados_explicativas_plano)- Vtimesteps
fim_coleta_rnn =  len(dados_padronizados_explicativas_plano)+ n_periodos


dados_explicativas_rnn_plano = df_explicativas_final_escalada.iloc[inicio_coleta_rnn:fim_coleta_rnn][colunas_rnn_plano]

#plano
yhat = predict_rnn(dados_padronizados_plano[colunas_rnn_plano], plano_analise, Vtimesteps, best_model_plano)
yhat = [x if not math.isnan(x) else 0.0 for x in yhat]
yhat_previsao = predict_rnn(dados_explicativas_rnn_plano, plano_analise, Vtimesteps, best_model_plano)
yhat_previsao = [x for x in yhat_previsao if not math.isnan(x)]
df_explicativas_final_escalada_plano['yhat_rnn']  = yhat+yhat_previsao
tempo_predicao = df_explicativas_final_escalada.loc[df_explicativas_final_escalada['modelo'] == 'Predição', 'seq_tempo'].min()



df_variaveis_simulacao = scaler_explicativas_plano.inverse_transform(df_explicativas_final_escalada_plano[colunas_explicativas_plano]) 
df_variaveis_simulacao = pd.DataFrame(df_variaveis_simulacao, columns=colunas_explicativas_plano)
df_variaveis_simulacao.to_excel(f'{current_directory}/simulacoes/sim_projecao_variaveis.xlsx')

# %% Resultado final com realizado e previsto
np.random.seed(0)

df_determinante_escalada_plano = df_explicativas_final_escalada_plano[['seq_tempo']]

## Linear

#plano
df_filtered = df_explicativas_final_escalada_plano[df_explicativas_final_escalada_plano['seq_tempo'] >= tempo_predicao][['seq_tempo','yhat_lin']]
df_filtered = df_filtered.rename(columns={'yhat_lin': plano_analise})
df_concatenada = pd.concat([dados_padronizados_plano[['seq_tempo', plano_analise]], df_filtered[['seq_tempo', plano_analise]]], ignore_index=True)
df_concatenada = df_concatenada.rename(columns={plano_analise: 'lin'})
df_determinante_escalada_plano['lin'] = df_concatenada[['lin']]

## SARIMA

# plano
df_filtered = df_explicativas_final_escalada_plano[df_explicativas_final_escalada_plano['seq_tempo'] >= tempo_predicao][['seq_tempo','yhat_sar']]
df_filtered = df_filtered.rename(columns={'yhat_sar': plano_analise})
df_concatenada = pd.concat([dados_padronizados_plano[['seq_tempo', plano_analise]], df_filtered[['seq_tempo', plano_analise]]], ignore_index=True)
df_concatenada = df_concatenada.rename(columns={plano_analise: 'sar'})
df_determinante_escalada_plano['sar'] = df_concatenada[['sar']]

## RNN


# plano
df_filtered = df_explicativas_final_escalada_plano[df_explicativas_final_escalada_plano['seq_tempo'] >= tempo_predicao][['seq_tempo','yhat_rnn']]
df_filtered = df_filtered.rename(columns={'yhat_rnn': plano_analise})
df_concatenada = pd.concat([dados_padronizados_plano[['seq_tempo', plano_analise]], df_filtered[['seq_tempo', plano_analise]]], ignore_index=True)
df_concatenada = df_concatenada.rename(columns={plano_analise: 'rnn'})
df_determinante_escalada_plano['rnn'] = df_concatenada[['rnn']]

df_determinante_escalada_plano['lin'],df_determinante_escalada_plano['sar'],df_determinante_escalada_plano['rnn'] = scaler_determinante_plano.inverse_transform(df_determinante_escalada_plano[['lin']]),  scaler_determinante_plano.inverse_transform(df_determinante_escalada_plano[['sar']]),  scaler_determinante_plano.inverse_transform(df_determinante_escalada_plano[['rnn']])
posicao_index_projecoes = df_determinante_escalada_plano.index[df_determinante_escalada_plano['seq_tempo'] == tempo_predicao].tolist()
df_determinante_escalada_plano['seq_tempo'] = df_explicativas_final[['seq_tempo']]
# %% Gerando tabela de anomes com meses previstos

tab_tempo = dados_anomes[['seq_tempo', 'Ano_mes']]


# Quantidade de novos registros
num_new_records = n_periodos

last_seq_tempo = tab_tempo['seq_tempo'].iloc[-1]
last_anomes = int(tab_tempo['Ano_mes'].iloc[-1])

# Adicionando novos registros
for _ in range(num_new_records):
    last_seq_tempo += 1
    last_anomes = increment_anomes(last_anomes)
    new_row = pd.DataFrame({'seq_tempo': [last_seq_tempo], 'Ano_mes': [last_anomes]})
    tab_tempo = pd.concat([tab_tempo, new_row], ignore_index=True)

df_determinante_escalada_plano = df_determinante_escalada_plano.merge(tab_tempo[['seq_tempo', 'Ano_mes']], on='seq_tempo', how='left')

anomes_predicao = df_determinante_escalada_plano.iloc[posicao_index_projecoes][['Ano_mes']].iloc[0,0]
anomes_predicao= str(anomes_predicao)
# %% plotando os resultados Finais
#plot_colored_line(df_determinante_escalada_plano[df_determinante_escalada_plano['seq_tempo'] >= 81], anomes_predicao, 'lin', 'sar', 'rnn', 'Gráfico de Simulação OLS vs SARIMA vs RNN para plano')
plot_colored_line(df_determinante_escalada_plano, anomes_predicao, 'lin', 'sar', 'rnn', 'Gráfico de Simulação OLS vs SARIMA vs RNN para plano')

dados_determinante_plano

simulacao_final = pd.merge(df_explicativas_final, df_determinante_escalada_plano, on='seq_tempo', how='inner')

simulacao_final = pd.concat([simulacao_final, dados_determinante_plano], axis=1)

simulacao_final.to_excel(f'{current_directory}/simulacoes/resultado_simulacao_final.xlsx')

