# Arquitetura do sistema

O projeto foi estruturado para separar matemática, acesso a dados, roteamento, visualização e interface.

## Camadas

### 1. Interface

Arquivo: `app.py`

Responsável por:

- configurar a página Streamlit;
- capturar cliques no mapa;
- controlar estado de origem/destino;
- chamar funções de cálculo;
- renderizar tabelas e gráficos;
- executar calibração interativa.

A interface não contém a lógica matemática principal. Ela apenas orquestra os módulos.

### 2. Domínio matemático

Arquivo: `src/nao_e_so_reta/norms.py`

Contém:

- norma $L_p$;
- distância $L_p$;
- casos especiais $L_1$, $L_2$, $L_\infty$;
- fronteira da bola $L_p$;
- polilinhas Manhattan;
- curva visual didática.

Arquivo: `src/nao_e_so_reta/analysis.py`

Contém:

- comparação entre métricas;
- erro relativo;
- tortuosidade;
- ajuste de escala $\alpha$;
- calibração de $p$;
- seleção de melhor $p$.

### 3. Dados e grafo

Arquivo: `src/nao_e_so_reta/graph_io.py`

Responsável por:

- carregar GraphML local;
- baixar grafo do OSM se necessário;
- projetar grafo para CRS métrico;
- retornar transformador para WGS84.

### 4. Roteamento

Arquivo: `src/nao_e_so_reta/routing.py`

Responsável por:

- encontrar nó mais próximo do clique;
- calcular menor caminho;
- retornar rota, nós, coordenadas e distância.

### 5. Amostragem

Arquivo: `src/nao_e_so_reta/sampling.py`

Responsável por:

- selecionar pares de nós;
- evitar componentes desconectadas;
- montar pares válidos para calibração.

### 6. Visualização

Arquivo: `src/nao_e_so_reta/visualization.py`

Responsável por:

- criar mapa base;
- adicionar marcadores;
- adicionar rota real;
- adicionar camadas $L_p$;
- adicionar legenda.

## Decisões técnicas

### Uso de GraphML local

O download do OSM pode ser lento e variável. Por isso, a aplicação prioriza `data/graph.graphml`, se o arquivo existir. Isso melhora:

- reprodutibilidade;
- velocidade;
- estabilidade de deploy.

### Cache do Streamlit

O grafo é carregado com `st.cache_resource`, pois é um recurso pesado e compartilhável entre reruns da interface.

### CRS projetado

As métricas $L_p$ precisam ser calculadas em metros. Por isso, o grafo é projetado com OSMnx antes de calcular distâncias planas.

### Testabilidade

As funções matemáticas não dependem de Streamlit, Folium ou OSMnx. Isso permite testes unitários rápidos e confiáveis.

## Evoluções sugeridas

- Persistir resultados de calibração em banco leve, como SQLite.
- Adicionar comparação entre múltiplas regiões.
- Adicionar análise por orientação de ruas.
- Adicionar estimativa de anisotropia por setores angulares.
- Adicionar cache local dos experimentos.
- Criar API FastAPI para separar backend e frontend.
