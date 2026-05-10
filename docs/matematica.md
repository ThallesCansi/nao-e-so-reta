# Fundamentos matemáticos

Este projeto investiga a distância urbana como comparação entre duas estruturas métricas.

## 1. Métrica intrínseca da rede

Uma rede viária pode ser modelada como um grafo ponderado:

$$
G=(V,E,\ell),
$$

onde $V$ é o conjunto de interseções ou pontos discretizados da via, $E$ é o conjunto de segmentos viários e $\ell(e)>0$ é o comprimento do segmento $e$.

A distância intrínseca entre dois nós $u,v\in V$ é:

$$
d_G(u,v)=\min_{\gamma:u\to v}\sum_{e\in\gamma}\ell(e),
$$

onde $\gamma$ percorre todos os caminhos possíveis entre $u$ e $v$. Computacionalmente, essa distância é calculada por algoritmos de menor caminho, como Dijkstra, quando os pesos são não negativos.

Em uma rede direcionada, $d_G(u,v)$ pode ser diferente de $d_G(v,u)$, por causa de ruas de mão única, acessos restritos ou modelagem direcional. Portanto, a rede viária pode induzir uma quase-métrica direcionada.

## 2. Métricas extrínsecas $L_p$

Cada nó da rede também possui uma posição espacial. Após projetar latitude/longitude para um sistema de coordenadas em metros, cada nó passa a ter coordenadas:

$$
x_u=(x_u^1,x_u^2)\in\mathbb{R}^2.
$$

A distância $L_p$ entre dois pontos é:

$$
\|x_u-x_v\|_p=
\left(|x_u^1-x_v^1|^p+|x_u^2-x_v^2|^p\right)^{1/p},
\quad p\ge 1.
$$

Casos especiais:

$$
\|x\|_1=|x_1|+|x_2|
$$

$$
\|x\|_2=\sqrt{x_1^2+x_2^2}
$$

$$
\|x\|_\infty=\max(|x_1|,|x_2|).
$$

Para $p<1$, a expressão ainda pode ser calculada, mas não define uma norma porque viola a desigualdade triangular. Por isso, o projeto restringe $p\ge 1$.

## 3. Relações entre normas

Em dimensão finita, todas as normas $L_p$ são equivalentes, mas não iguais. Em $\mathbb{R}^2$, para $1\le p\le q\le\infty$, vale:

$$
\|x\|_q\le \|x\|_p \le 2^{(1/p-1/q)}\|x\|_q.
$$

Isso significa que mudar $p$ altera a geometria percebida, mas de forma controlada. No contexto urbano, essa variação ajuda a investigar se uma cidade se comporta mais como uma malha ortogonal, associada a $L_1$, ou como um espaço mais isotrópico, associado a $L_2$.

## 4. Bolas unitárias

A bola unitária de $L_p$ é:

$$
B_p=\{x\in\mathbb{R}^2:\|x\|_p\le 1\}.
$$

- $p=1$: losango;
- $p=2$: círculo;
- $p\to\infty$: quadrado.

A visualização da fronteira da bola $L_p$ no mapa é didática: ela mostra quais pontos têm mesma distância $L_p$ em relação à origem no plano projetado.

## 5. Distorção métrica

A rede viária define uma métrica diferente da métrica plana. Para um par de pontos, podemos definir a razão:

$$
\rho_p(u,v)=\frac{d_G(u,v)}{\|x_u-x_v\|_p}.
$$

Quando $p=2$, essa razão é frequentemente chamada de tortuosidade:

$$
\tau(u,v)=\frac{d_G(u,v)}{\|x_u-x_v\|_2}.
$$

Valores altos indicam que o caminho real é muito maior que a reta euclidiana. Isso pode ocorrer por barreiras físicas, desenho urbano, baixa conectividade, vias sinuosas, rios, ferrovias, condomínios fechados ou restrições de direção.

## 6. Calibração empírica de $p$

Uma pergunta natural é: qual valor de $p$ melhor aproxima a geometria da rede?

Para uma amostra de pares $(u_i,v_i)$, ajustamos:

$$
d_G(u_i,v_i)\approx \alpha \|x_{u_i}-x_{v_i}\|_p.
$$

Para cada $p$, o melhor fator de escala $\alpha$ por mínimos quadrados sem intercepto é:

$$
\alpha^\*(p)=
\frac{\sum_i \|x_{u_i}-x_{v_i}\|_p d_G(u_i,v_i)}
{\sum_i \|x_{u_i}-x_{v_i}\|_p^2}.
$$

Depois avaliamos erros como MAE, RMSE e MAPE. O $p$ ótimo depende do local, do tipo de rede, da amostra de pares e do objetivo da comparação.

## 7. Interpretação geométrica urbana

- Redes em grade ortogonal tendem a se aproximar de $L_1$, especialmente quando deslocamentos seguem eixos predominantes.
- Redes radiais, irregulares ou com caminhos diagonais podem se aproximar mais de $L_2$.
- Redes com restrições fortes, gargalos ou barreiras podem apresentar tortuosidade elevada em todas as normas planas.
- A calibração de $p$ não substitui a rede; ela é uma aproximação agregada da anisotropia urbana.

## 8. Limitações

1. A comparação depende da qualidade dos dados OpenStreetMap.
2. A projeção cartográfica precisa ser adequada à região estudada.
3. A distância real usa o nó mais próximo do clique, não necessariamente o ponto exato clicado.
4. A visualização de curvas auxiliares não deve ser confundida com rotas reais.
5. O melhor $p$ é descritivo, não causal.
