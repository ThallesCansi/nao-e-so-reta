# Deploy

## Streamlit Community Cloud

1. Suba o repositório para o GitHub.
2. Configure o arquivo principal como `app.py`.
3. Garanta que `requirements.txt` esteja na raiz.
4. Para evitar download lento do OSM a cada deploy, gere `data/graph.graphml` e inclua o arquivo no repositório se o tamanho permitir.

## Docker

Build:

```bash
docker build -t nao-e-so-reta .
```

Run:

```bash
docker run --rm -p 8501:8501 nao-e-so-reta
```

Abra:

```text
http://localhost:8501
```

## Variáveis de ambiente

`GRAPH_PATH` pode ser usada para apontar para um GraphML específico:

```bash
GRAPH_PATH=/app/data/graph.graphml streamlit run app.py
```

## Recomendações de produção

- Preferir GraphML local a download sob demanda.
- Fixar versões quando o projeto for apresentado ou publicado.
- Documentar a data de extração do OpenStreetMap.
- Salvar resultados de calibração usados em relatórios.
