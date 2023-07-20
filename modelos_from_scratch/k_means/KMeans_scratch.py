import numpy as np
import pandas as pd


class KMeansScratch:

    def __init__(self, n_clusters = 2, tol = 1e-5, rodadas=10):
        """Método construtor"""

        self.n_clusters = n_clusters

        self.centroids = []

        self.tol = tol

        self.atribuicoes = None

        self.inertia = None

        self.rodadas = rodadas

        self.best = None
    

    def _CalcularInercia(self, cols: int, matriz: np.ndarray) -> float:
        
        soma_quadrado = 0
        for cluster in range(self.n_clusters):
            
            rows = self.atribuicoes[cluster]

             
            
            centroide = self.centroids[cluster].reshape((1,cols))[0]
            for row in rows:
                vetor = matriz[row].reshape((1,cols))[0]
                soma_quadrado += (centroide - vetor )**2
        
        return soma_quadrado.sum()


    def _IniciarCentroid(self,cols: int) -> None:
        """Inicia os centroids aleatoriamente no intervalo [0,1]"""

        # Resetando os centroides
        self.centroids = []

        for cluster in range(self.n_clusters):
            self.centroids.append(np.random.random(cols))

        

    def _Distancia(self, vetor: np.array, centroide: np.array) -> float:
        '''Calcula a distância euclidiana entre dois vetores'''
        return np.sqrt(((vetor-centroide)**2).sum())
    
    def _RecalculaCentroide(self, rows: list ,matriz: np.ndarray, cluster: int, cols: int) -> None:
        '''Recalculando os centroides'''
        
        novo_centroide = []
        for col in range(cols):
            sum = 0
            for row in rows:
                sum += matriz[row,col]
        
            try:
                media = sum / len(rows)
                novo_centroide.append(media)
        
                self.centroids[cluster] = np.array(novo_centroide)
            except:
                pass#print(f'Cluster {cluster} com divisao por zero')


    def _DiferencaCentroides(self, centroide_antigo: list) -> float:
        """Calcula a média do produto escalar de todos os centroides"""
        
        diferenca = []
        for cluster in range(self.n_clusters):
            vnovo = self.centroids[cluster]
            vantigo = centroide_antigo[cluster]
            diferenca.append(np.abs(vnovo-vantigo))

        
        return np.array(diferenca).mean()

    def _Armazenarhistorico(self, rodada: int):

        if rodada == 0:
            self.best['centroides'] = self.centroids.copy()
            self.best['atribuidos'] = self.atribuicoes.copy()
            self.best['inertia'] = self.inertia.copy()

        else:
            if(self.inertia < self.best['inertia']):
                self.best['centroides'] = self.centroids.copy()
                self.best['atribuidos'] = self.atribuicoes.copy()
                self.best['inertia'] = self.inertia.copy()          
            

    def fit(self,X: pd.DataFrame) -> list:
        """Essa função é responsável por treinar o modelo"""

        rows = X.shape[0]
        cols = X.shape[-1]

        matriz = X.to_numpy()


        self.best = {'centroides': None, 'atribuidos': None, 'inertia': None}

        for rodada in range(self.rodadas):
           
            # Iniciando os centroids 
            self._IniciarCentroid(cols)

            # Determinando os centroides
            diff = 1
            while(diff > self.tol):
                
                distancias = {} # conterá a distância de cada linha com todos os centroides

                # Calcula a distância de cada vetor com relação a cada centroide
                # Cada linha do dataset representa um vetor

                # percorre cada linha do dataset
                for row in range(rows):

                    # Para cada linha (vetor) é efetuado o calculo de distância com relação aos centroides
                    dis = []
                    for centroide in self.centroids:
                        vetor = matriz[row].reshape((1,cols))[0]
                        centroide = centroide.reshape((1,cols))[0]
                    
                        dis.append(self._Distancia(vetor,centroide))
                    
                    distancias[row] = dis
                
                # Atribuindo as linhas aos clusters
                self.atribuicoes = {cluster : list() for cluster in range(self.n_clusters)}
                
                for row , dists in distancias.items():
                    cluster = np.argmin(np.array(dists))

                    self.atribuicoes[cluster].append(row)

                #Centroide antigo
                centroide_antigo = self.centroids.copy()


                # Recalculando os centroides
                for cluster, indices in self.atribuicoes.items():
                    self._RecalculaCentroide(indices, matriz, cluster,cols)

                # Calculando a diferença entre os centroides atualizados e antigos
                diff = self._DiferencaCentroides(centroide_antigo)

            
            self.inertia = self._CalcularInercia(cols,matriz)

            self._Armazenarhistorico(rodada)


        self.centroids = self.best['centroides']
        self.atribuicoes = self.best['atribuidos']
        self.inertia = self.best['inertia']    
