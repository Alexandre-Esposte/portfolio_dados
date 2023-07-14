# Introdução

Bem-vindo(a) ao meu projeto de portfólio na área de Ciência de Dados! Neste trabalho, tenho o prazer de apresentar um projeto que realizei utilizando uma base de dados encontrada no Kaggle sobre carros e esses dados podem ser encontrados através do seguinte link: <a href="https://www.kaggle.com/datasets/lepchenkov/usedcarscatalog" target="_blank">Carros usados</a>

A base de dados foi obtida por meio de web scraping em concessionárias localizadas na Bielorrússia. Ela contém informações detalhadas sobre vários veículos, incluindo diversos atributos tais como quilometragem rodada, estado do veículo, ano de fabricação, cor do veículo, fabricante e entre muitos outros atributos. Nosso objetivo principal é desenvolver um modelo de predição capaz de aprender os padrões presentes nesses dados e fornecer estimativas precisas dos preços dos veículos com base em suas características.

Esse projeto foi dividido em alguns arquivos, os arquivos de código estão no diretório scripts, nesses arquivos são definidas algumas funções e classes utilizadas para treinar e validar modelos de forma mais ágil. Temos também dois notebooks sendo um deles voltado para a análise da base (analise.ipynb) e outro para modelagem (modelagem.ipynb).

# Contextualização

A Esposte consultorias é uma empresa de consultoria de dados altamente especializada, temos o prazer de fornecer serviços sob medida para uma ampla gama de empresas. Uma de nossas empresas clientes mais destacadas é a ABC Automóveis, uma renomada concessionária de veículos usados. A ABC Automóveis possui uma vasta rede de revendas e se destaca por sua ampla seleção de carros de qualidade.

No entanto, mesmo com seu sucesso no mercado, a ABC Automóveis enfrenta um desafio recorrente e crítico: a dificuldade em precificar corretamente seus carros usados. A equipe de vendas, composta por profissionais experientes, muitas vezes se depara com a tarefa complexa de estabelecer preços justos e competitivos para cada veículo em sua ampla gama de estoque.

Analisando o processo atual da ABC Automóveis, identificamos que o desafio reside na falta de uma metodologia estruturada para precificar carros usados. Os profissionais da empresa, embora tenham conhecimento e intuição sobre o mercado, muitas vezes cometem erros na precificação, o que pode levar a prejuízos financeiros e à perda de oportunidades de vendas.

Para solucionar essa dor específica da ABC Automóveis, nossa equipe de consultores de dados entrou em ação. Colaboramos diretamente com a equipe da concessionária, trabalhando em estreita parceria para entender a dinâmica do mercado automotivo e coletar um conjunto abrangente de dados sobre carros usados.

Em nosso processo de consultoria, utilizamos técnicas avançadas de ciência de dados e aprendizado de máquina para desenvolver um modelo de precificação personalizado para a ABC Automóveis. Esse modelo leva em consideração uma série de fatores, como marca, modelo, ano de fabricação, quilometragem, condições do veículo e outros atributos relevantes.

Com base nesses dados, treinamos o modelo para aprender as correlações e padrões ocultos que influenciam os preços de carros usados no mercado. Isso nos permitiu desenvolver uma solução que fornece à equipe da ABC Automóveis uma estimativa precisa do valor de mercado de cada veículo em seu estoque.

Ao implementar nossa solução de precificação, a ABC Automóveis agora pode tomar decisões embasadas em dados sólidos. Nossa metodologia personalizada ajuda a empresa a estabelecer preços justos e competitivos, minimizando erros de precificação e otimizando suas oportunidades de vendas.

Com essa abordagem baseada em dados, nossos serviços de consultoria de dados trouxeram à ABC Automóveis uma melhoria significativa em sua capacidade de precificar carros usados.

Estamos orgulhosos de destacar esse projeto em nosso portfólio, pois ele ilustra nossa capacidade de resolver desafios específicos da indústria, fornecendo soluções personalizadas. Nossa parceria com a ABC Automóveis exemplifica o poder da análise de dados para impulsionar o sucesso dos negócios e alcançar resultados reais no mercado automotivo.


# Definições

Antes de iniciarmos é necessário definirmos como os nossos modelos serão avaliados. Para esse projeto vamos utilizar as três métricas a seguir:

1. **Mean absolute percentage error (MAPE)**
2. **Mean absolute error (MAE)**
3. **Root mean squared error (RMSE)**

A nossa principal métrica sera o MAPE e tomaremos como erro aceitável até no máximo 20%, entretanto o desejável são erros menores que 15%. Além disso, esses erros são apenas uma estimativa através da média, almejamos uma análise mais aprofundada para podemos entender melhor os nossos modelos treinados.

# Arquivos

Esse projeto foi divido em dois notebooks e são eles:

1. analise.ipynb

 * Neste arquivo apresentamos uma breve análise dos dados onde buscamos por inconsistências nos dados e como as variáveis correlacionam-se com a variável alvo. Não exploramos afundo questões de análise visto que nosso foco é a modelagem.

2. modelagem.ipynb

 * Neste arquivo apresentamos todos os passos que utilizamos para modelar nosso modelo de machine learning.

 Além disso temos os códigos funcoes.py e modelos.py no diretório  scripts. No primeiro apresentamos algumas funções utilizadas tanto na análise quanto na modelagem e no segundo trazemos uma classe que nos auxiliar a treinar e validar vários modelos ao mesmo tempo.

