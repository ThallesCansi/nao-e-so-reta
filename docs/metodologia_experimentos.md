# Metodologia de experimentos

Este documento descreve como transformar o app em uma ferramenta de investigação quantitativa.

## 1. Escolha da região

A região deve ser suficientemente pequena para que:

- a projeção métrica seja adequada;
- o grafo seja carregado em tempo razoável;
- a malha viária seja coerente com o objetivo do estudo.

Exemplos:

- bairro;
- campus universitário;
- região administrativa;
- centro urbano.

## 2. Escolha do tipo de rede

O OSMnx permite redes como:

- `drive`: vias dirigíveis;
- `walk`: rede caminhável;
- `bike`: rede ciclável.

Cada tipo responde a uma pergunta diferente. Não se deve comparar resultados de `drive` e `walk` como se fossem a mesma geometria.

## 3. Amostragem de pares

A calibração usa pares de nós \((u_i,v_i)\). Recomendações:

- usar semente fixa para reprodutibilidade;
- amostrar pares da maior componente conectada;
- remover pares sem caminho;
- registrar número de pares válidos;
- repetir o experimento com sementes diferentes.

## 4. Métricas de avaliação

Para cada valor de \(p\), ajusta-se:

\[
d_G \approx \alpha d_p.
\]

Depois calculam-se:

- MAE: erro absoluto médio em metros;
- RMSE: penaliza erros grandes;
- MAPE: erro percentual médio;
- distorção média: \(d_G/d_p\).

## 5. Interpretação do melhor p

O melhor \(p\) indica qual bola \(L_p\), após escala, mais se aproxima da distância média da rede.

Interpretações típicas:

- \(p\) próximo de 1: estrutura fortemente ortogonal;
- \(p\) próximo de 2: deslocamentos mais isotrópicos;
- \(p\) alto: presença de comportamento mais próximo de max(|dx|,|dy|), o que pode ocorrer em certas geometrias restritas, mas deve ser interpretado com cuidado.

## 6. Cuidados estatísticos

- Um único par origem-destino não caracteriza a cidade.
- Pares muito curtos podem ser dominados por erro de discretização.
- Pares muito longos podem capturar barreiras regionais, não a geometria local.
- O melhor \(p\) pode mudar por escala espacial.
- A calibração deve ser comparada entre diferentes amostras.

## 7. Experimentos recomendados

1. Calibrar \(p\) para `drive`, `walk` e `bike`.
2. Comparar bairros planejados e bairros orgânicos.
3. Avaliar distribuição de tortuosidade.
4. Separar pares por distância euclidiana curta, média e longa.
5. Avaliar se \(p\) ótimo muda por orientação do deslocamento.
