PROJECT_EXPLANATION = r"""
Este app compara duas geometrias:

1. **Geometria intrínseca da cidade**: a distância real é o menor caminho no grafo de ruas.
2. **Geometria extrínseca do plano**: distâncias \(L_p\) calculadas em coordenadas projetadas em metros.

A diferença entre elas mede, de forma prática, a distorção causada pela malha urbana.
"""

MATH_PANEL = r"""
### Métricas usadas

Para dois nós \(u,v\) da rede, com posições projetadas \(x_u,x_v \in \mathbb{R}^2\):

\[
d_G(u,v)=\min_{\gamma:u\to v}\sum_{e\in\gamma}\ell(e)
\]

é a distância intrínseca do grafo. Já a métrica \(L_p\) é:

\[
\|x_u-x_v\|_p =
\left(|\Delta x|^p+|\Delta y|^p\right)^{1/p}.
\]

Casos especiais:

- \(p=1\): Manhattan;
- \(p=2\): Euclidiana;
- \(p=\infty\): Chebyshev, \(\max(|\Delta x|,|\Delta y|)\).

O índice de tortuosidade usado aqui é:

\[
\tau(u,v)=\frac{d_G(u,v)}{\|x_u-x_v\|_2}.
\]

Valores próximos de 1 indicam que a rede se aproxima de uma reta. Valores maiores indicam barreiras, baixa conectividade, vias sinuosas ou restrições direcionais.
"""

CALIBRATION_EXPLANATION = r"""
### Calibração empírica de \(p\)

Para uma amostra de pares de nós, o app busca qual valor de \(p\) faz com que uma métrica plana escalada aproxime melhor a distância real:

\[
d_G(u,v) \approx \alpha \|x_u-x_v\|_p.
\]

Para cada \(p\), o fator \(\alpha\) é calculado por mínimos quadrados sem intercepto:

\[
\alpha^\*=\frac{\sum_i x_i y_i}{\sum_i x_i^2},
\]

onde \(x_i=\|x_{u_i}-x_{v_i}\|_p\) e \(y_i=d_G(u_i,v_i)\).

O melhor \(p\) não deve ser interpretado como uma "verdade universal" da cidade. Ele resume, para aquela amostra e aquele tipo de rede, qual norma plana mais se aproxima da métrica de ruas.
"""
