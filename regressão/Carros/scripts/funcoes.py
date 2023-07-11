import seaborn           as sns
import matplotlib.pyplot as plt
import pandas            as pd

from sklearn.metrics     import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error



def validacao(y_true,y_predicted,modelo = None):
    """Exibe as métricas de validação estabelecidas"""

    mae = mean_absolute_error(y_true,y_predicted)

    mape = mean_absolute_percentage_error(y_true, y_predicted)

    rmse = mean_squared_error(y_true,y_predicted,squared=False)


    fig, ax = plt.subplots(1,2,constrained_layout=True,figsize=(15,8))

    fig.suptitle('Validação',size=16)

    ax[0].set_ylabel('Predito',size=14)
    ax[0].set_xlabel('Real',size=14)
    


    ax[1].set_ylabel(' ')
    ax[1].set(yticklabels=[])  # remove the tick labels
    ax[1].tick_params(bottom=False)  # remove the tick



    sns.scatterplot(x = y_true,y=y_predicted, ax=ax[0])
    sns.lineplot(x=y_true,y=y_true, ax=ax[0],color='black',linewidth=3)



    sns.histplot(x=y_true,ax=ax[1],label='Preço real',bins=50)
    sns.histplot(x=y_predicted,ax=ax[1],label='Preço predito',bins=50)
 


    texto = f"Modelo: {modelo}\n\n"+ f"MAE: {mae:.2f}\n\nMAPE: {mape:.2f}\n\nRMSE: {rmse:.2f}"
    fig.text(1.01,0.76,texto,size=14,bbox=dict(boxstyle='square',ec= (0,0,0), fc= (9/255, 232/255, 102/255)))

    ax[1].legend()


def outlier(serie: pd.Series) -> list:
    """Retorna os indices dos registros classificados como outlier"""

    q3 = serie.quantile(0.75)
    q1 = serie.quantile(0.25)
    
    dq = q3-q1
    
    lim_sup = q3 + (1.5*dq)
    
    lim_inf = q1 - (1.5*dq)
    
    idx = []
    
    for ind, value in enumerate(serie):
        if value < lim_inf or value > lim_sup:
            idx.append(ind)
            
    return idx
