# Calculadora Métrica Urbana (Streamlit)

App em Streamlit para comparar distância real (menor caminho em ruas via OpenStreetMap/OSMnx) com distâncias teóricas de normas \(L_p\) (Euclidiana, Manhattan, Chebyshev e Minkowski).

## Rodando localmente

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app_refatorado.py
```

Na primeira execução, o app baixa a malha viária do OpenStreetMap (pode demorar).

## Como usar

1. Clique no mapa para definir **origem** (verde).
2. Clique de novo para definir **destino** (vermelho).
3. Veja a rota real (azul) e as comparações teóricas.

## Notas

- A “curva/bola” Minkowski é gerada no plano projetado (metros) e reprojetada para o mapa (apenas visual/ilustrativa).
- A distância real é calculada como menor caminho no grafo viário (peso `length` em metros).
