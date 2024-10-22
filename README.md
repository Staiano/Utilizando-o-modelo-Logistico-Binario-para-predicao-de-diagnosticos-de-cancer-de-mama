# Utilizando o modelo Logistico Binário para predição de diagnósticos de câncer de mama

11 de outubro de 2024

Olá pessoal!

A detecção precoce e precisa do câncer de mama é fundamental para o tratamento e a sobrevivência dos pacientes.

Este é um mini-artigo de estudo one utilizei o Python para construir um  **modelo logístico binário**  como ferramenta para a análise e predição de diagnósticos, permitindo uma compreensão mais profunda das características associadas a tumores malignos e benignos.

O modelo demonstrou sua eficácia, alcançando uma  **acurácia**  de quase  **97%**  e maximizando a  **sensibilidade**  com quase **99%**  de precisão nas predições, ao adotar um  **cutoff**  de  **0,2**. Esse alto nível de sensibilidade foi priorizado para reduzir o risco de diagnósticos incorretos.

No contexto de saúde, identificar corretamente os casos de câncer maligno é crucial, pois diagnósticos incorretos, especialmente os falsos negativos, podem impedir que pacientes recebam tratamento adequado de maneira oportuna.

### Descrição do Conjunto de Dados

Os dados utilizados foram obtidos pela  [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  e disponibilizados pela University of Wisconsin Hospitals. O conjunto de dados contém 569 observações com 30 variáveis preditoras (X) e uma variável dependente (Y). As características incluem medições de propriedades celulares, como:

-   **Raio**  (radius)
-   **Textura**  (texture)
-   **Perímetro**  (perimeter)
-   **Área**  (area)
-   **Suavidade**  (smoothness)
-   **Compactação**  (compactness)
-   **Concavidade**  (concavity)
-   **Pontos Côncavos**  (concave points)
-   **Simetria**  (symmetry)
-   **Dimensão Fractal**  (fractal dimension)

A variável  **dependente (Y)**  é o diagnóstico (Diagnosis), com valores dicotômicos que indicam se o tumor é maligno ou benigno.

## Análise Exploratória

### Boxplot para Análise das Observações

Foi realizado um boxplot das variáveis X para entender seu comportamento. Observou-se que muitas variáveis apresentaram outliers. Considerando que tumores malignos podem modificar células de forma significativa, foi decidido manter o conjunto de dados original, sem remoção de outliers, para preservar essas informações.

![](https://media.licdn.com/dms/image/v2/D4D12AQHuPQeAtzbBGw/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1728649701916?e=1735171200&v=beta&t=UxsNbQHQPESTupm4IOFdsuLymWC9VccM4BT_ubBx6Fk)

### Correlação entre Versões das Variáveis X

A análise de correlação entre as diferentes versões das variáveis revelou correlações altas entre as versões 1 e 3, sugerindo multicolinearidade. Assim, foram desenvolvidos modelos separados para cada versão das variáveis e, posteriormente, um modelo combinado.

![](https://media.licdn.com/dms/image/v2/D4D12AQEis_R1gasopA/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1728649732174?e=1735171200&v=beta&t=2pTNGMfAdKUlkfYZ4a9IgJyq740Y6TltsV4K1zkiI8s)

### Criação do Modelo

Desenvolveram-se três modelos logísticos distintos para cada versão das variáveis e um modelo final que combina as versões 2 e 3. Cada modelo foi construído com atenção à seleção de variáveis, buscando otimizar a sensibilidade e a precisão.

-   **Versão 3:**  Após o processo stepwise, o modelo manteve apenas quatro variáveis de um conjunto inicial de 10. Este modelo teve um log-likelihood de aproximadamente -44, indicando um bom ajuste, e alcançou uma acurácia robusta.

![](https://media.licdn.com/dms/image/v2/D4D12AQHE51sSF762SA/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1728649910961?e=1735171200&v=beta&t=fkrfIQo3HVMNbH-08dXyGtuOBI0epw1QmQsl7N1NsPQ)

Resultados do modelo X3

-   **Versão 2:**  Apresentou um beta significativo para a variável "fractal_dimension2", ressaltando a importância dessa medida na identificação de malignidade. Esse beta elevado sinalizou que a dimensão fractal é um fator chave para distinguir entre tumores malignos e benignos.

![](https://media.licdn.com/dms/image/v2/D4D12AQH-X4lS4S_Y5Q/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1728649768105?e=1735171200&v=beta&t=3slU7Ra17CkfLSXJlkhCvC5N6bxqkI_Z24WJqM6RaCA)

Resultados do modelo X2

### Modelo Combinado

![](https://media.licdn.com/dms/image/v2/D4D12AQHtbsclz4hmfQ/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1728649853850?e=1735171200&v=beta&t=caoPoM3UrQ98GJ8klPTcgOUN_SnkXbEt3YRtP_wxurU)

Resultados do modelo combinado

Optou-se por combinar as versões 2 e 3, uma vez que ambas demonstraram características preditivas complementares. O modelo combinado resultou em um log-likelihood de -30, superando os modelos individuais e alcançando uma acurácia de quase 97% e uma sensibilidade de 99%, e uma especificidade de 95% com um cutoff de 0,2.

![](https://media.licdn.com/dms/image/v2/D4D12AQFFmDQBJvr8LA/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1728649843905?e=1735171200&v=beta&t=-6jkAYiNXYuvNjmaHoPy4MJnkSHEnfW2Xt54CBXTSHo)

Matriz de confusão do modelo combinado de um cutoff de 0,2

Para maximizar a sensibilidade e reduzir o risco de falsos negativos, adotou-se um cutoff mais baixo.

Isso é crucial na detecção precoce de câncer, onde a segurança do paciente exige uma alta taxa de identificação de casos malignos. Essa escolha estratégica permite que mais casos de câncer sejam corretamente classificados como malignos, mesmo que haja um aumento nos falsos positivos. Em diagnósticos de saúde, o impacto positivo de reduzir diagnósticos incorretos e atrasos no tratamento supera o risco de avaliações adicionais causadas pelos falsos positivos.

### Conclusão

O modelo logístico binário utilizado neste estudo simulado revelou-se eficaz para a predição da malignidade em tumores mamários. Com uma acurácia de quase 97% e uma sensibilidade de 99%, o modelo mostrou-se uma ferramenta robusta para transformar dados complexos em informações acionáveis, de valor potencial para a prática clínica. Este estudo ressalta a importância da seleção cuidadosa de variáveis e da maximização da sensibilidade,mesmo perdendo um pouco da acurácia, evidenciando o papel dos modelos logísticos no campo da ciência de dados e sua relevância para o diagnóstico de câncer de mama.

Obrigado pessoal por me acompanharem até o final e até a próxima!