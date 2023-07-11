import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import seaborn               as sns

from sklearn.metrics 	     import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error
from sklearn.compose 	     import ColumnTransformer
from sklearn.pipeline 	     import Pipeline
from sklearn.model_selection import KFold, train_test_split


class Modelos:

    def __init__(self) -> None:
        
        self.models = dict()

        self.results = dict()

        self.results_per_fold = dict()

        self.kf = KFold(n_splits=5) 

    def AddModel(self, modelos : list = []) -> None:
        """Método para adicionar modelos ao objeto a estrutura é uma lista de tuplas onde a tupla segue o seguinte esquema: (nome do modelo, modelo instanciado)"""
        
        for modelo in modelos:
            self.models[modelo[0]] = modelo[1]

    def RemoveModel(self, nome: str = None, tipo: str = None) -> None:
        """Remove modelos do objeto"""
        
        if tipo == 'all':
            self.models=dict()
            return

        del self.models[nome]

          
    def FitoneModel(self,X: pd.DataFrame, y: pd.Series, pipe: Pipeline ,nome: str):
        """Treina somente um modelo dentre os que estão no objeto"""
        if nome not in self.models.keys():
            print('Modelo invalido')
            return
        
        if len(pipe.steps) > 1:
            pipe.steps.pop()
            
        pipe.steps.append((nome,self.models[nome]))

        modelo = pipe

        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

        modelo.fit(x_train,y_train)
        predito = modelo.predict(x_test)

        aux_df = x_test.copy()
    

        return predito, y_test, aux_df


    def FitModels(self, X: np.ndarray, y: np.array, pipe: Pipeline = None, log_y: bool = False) -> None:
        """Treina todos os modelos inseridos no objeto através de validação cruzada"""
		
        if len(self.models) == 0:
            return "Nenhum modelo adicionado na estrutura"


        for aux in self.models.items():
            nome_modelo = aux[0]
            modelo = aux[1]

            if len(pipe.steps) > 1:
                pipe.steps.pop()

            mae = 0
            mape = 0
            rmse = 0

            resultados_aux = []

            print(f"-----{nome_modelo}-----")
            pipe.steps.append(("Model",modelo))
            modelo = pipe
            for i, (train_index, test_index) in enumerate(self.kf.split(X)):
                #print(f"Fold {i}:")
                
                #print(f"  Train: index={train_index}")
                #print(f"  Test:  index={test_index}")

                X_train = X.loc[train_index,:]
                y_train = y.loc[train_index]

                
                X_test = X.loc[test_index,:]
                y_test = y.loc[test_index]


                if log_y:
                    y_train = np.log(y_train)
                    

                
                modelo.fit(X_train,y_train)

                predito = modelo.predict(X_test)
                
                if log_y:
                    predito = np.exp(predito)
                    
                
                
                resultados_aux.append((y_test,predito)) 
                
                mae  += mean_absolute_error(y_test,predito)
                mape += mean_absolute_percentage_error(y_test,predito)
                rmse += mean_squared_error(y_test,predito,squared=True)

                #print(f"MAE: {mae:.2f} || MAPE: {mape:.2f} || RMSE: {rmse:.2f}")
            self.results[nome_modelo]=[mae/5, mape/5, rmse/5]
            self.results_per_fold[nome_modelo] = resultados_aux

        self._gerar_resultado()

    def _gerar_resultado(self) -> None:
        """Gera os resultados em uma estrutura DataFrame"""
        
        indices = ['mae','mape','rmse']
        display(pd.DataFrame(self.results,index=indices).T)
    

 
