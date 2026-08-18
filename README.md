# ⚽ Football-Stats OCR

Um projeto em **Python** que utiliza **OCR (Optical Character Recognition)** para extrair estatísticas de jogadores diretamente de imagens e convertê-las em um arquivo **CSV**, facilitando a análise e o tratamento dos dados.

Atualmente o projeto é voltado para imagens do **FIFA**, porém o objetivo é torná-lo compatível com outros jogos e interfaces semelhantes no futuro.

---

# 📌 Objetivo

Automatizar a coleta de estatísticas presentes em imagens, eliminando a necessidade de preenchimento manual de planilhas.

O programa percorre automaticamente todas as imagens presentes nas pastas dos jogadores ou diretamente na pasta, realiza a leitura via OCR e gera um arquivo CSV contendo os dados encontrados.

---

# 🚀 Tecnologias Utilizadas

- Python 3
- OpenCV
- Tesseract OCR
- Pytesseract
- Glob
- OS
- RE (Regex)
- CSV
- Time
- Difflib
- Unicodedata

---

# 📂 Estrutura do Projeto

```text
Football-Stats OCR/
│
├── Imagens/
│   ├── Jogador 1/img.png
│   ├── Jogador 2/img.png
│   └── ...
│        OR
├── Imagens/
│   ├── img.png
│   ├── img.png
│   └── ...
│
│
│
│
├── output/
│   └── jogadores.csv
│
└── main.py
```

---

# ⚙ Funcionamento

O código executa as seguintes etapas:

1. Percorre todas as pastas de jogadores.
2. Localiza automaticamente todas as imagens.
3. Pré-processa cada imagem para melhorar o OCR.
4. Extrai os dados da tabela.
5. Organiza os resultados.
6. Exporta tudo para um arquivo CSV.

---

# 📊 Testes

Até o momento foram realizados testes utilizando:

- **298 imagens**
- **36 tabelas individuais**
- **10.728 dados coletados**
<br></br>
- **499 imagens**
- **36 tabelas individuais**
- **19.760 dados coletados**
- 4 imagens descartadas no total
<br></br>
- **612 imagens**
- **36 tabelas individuais**
- **22.291 dados coletados**
- 8 imagens descartadas no total

Os resultados têm sido satisfatórios para um projeto em desenvolvimento, embora ainda existam limitações inerentes ao OCR.

---
# ⚡ Performance

Atualmente o projeto apresenta os seguintes tempos médios de execução:

| Quantidade de imagens | Tempo médio |
|-----------------------:|------------:|
| 298 imagens | ~17 minutos |
| 499 imagens | ~45 minutos |
| 612 imagens | ~43 minutos |

Durante esse processamento são realizadas operações como:

- Leitura automática de todas as imagens;
- Pré-processamento com OpenCV;
- Execução do OCR via Tesseract;
- Extração dos dados;
- Organização das informações;
- Geração do arquivo CSV.

O tempo de execução depende principalmente da velocidade do **Tesseract OCR**, que representa a maior parte do processamento, além da resolução das imagens e do desempenho do computador utilizado.

Futuramente estão previstas otimizações para reduzir significativamente esse tempo de execução, incluindo melhorias no processamento das imagens, redução de chamadas ao OCR e paralelização das leituras.
---

# ⚠ Limitações Conhecidas

Os principais problemas encontrados atualmente são:

- Alguns valores podem retornar como **0** devido a falhas de leitura do OCR.
- Alguns campos podem permanecer vazios quando o texto não é identificado.
- Alguns nomes de jogadores ainda são lidos de forma incorreta.
    - Letras ausentes;
    - Letras extras ("fantasmas");
    - Palavras quebradas.

Esses problemas ocorrem principalmente devido à fonte utilizada pelo FIFA e às limitações naturais do reconhecimento óptico de caracteres.

Melhorias futuras serão implementadas para aumentar a precisão das leituras.

---

# 📈 Versões

## Version 0.5

A primeira grande versão do projeto.

Possuía suporte para cinco tabelas diferentes do FIFA:

- Resumo
- Passes
- Finalizações
- Defesa
- Goleiro (GL)

A proposta era identificar automaticamente a aba aberta, coletar todas as estatísticas e unificá-las em um único CSV.

Durante os testes surgiram diversos problemas relacionados ao acúmulo de dados, sincronização entre tabelas e erros de escrita provocados pelo OCR.

---

## Version 1.0 (Atual)

Após diversas tentativas de estabilização da arquitetura, o projeto foi simplificado.

Nesta versão o foco passou a ser exclusivamente a tabela **Resumo**, permitindo uma coleta muito mais consistente e reduzindo significativamente a complexidade do processamento.

Essa decisão tornou o código mais estável, organizado e servirá como base para futuras expansões.

---

# 🔮 Roadmap

Algumas melhorias planejadas:

- Melhor reconhecimento de nomes de jogadores.
- Aumento da precisão do OCR.
- Redução de leituras nulas.
- Suporte para outras tabelas do FIFA.
- Compatibilidade com diferentes resoluções.
- Compatibilidade com outros jogos esportivos.
- Melhor organização do código.
- Interface gráfica para facilitar a utilização.

---

# 📌 Observações

Este projeto foi desenvolvido principalmente como forma de estudo e aprendizado em:

- Python
- Visão Computacional
- OCR
- Processamento de Imagens
- Manipulação de arquivos CSV
- Manipulação de dados

Todas as imagens utilizadas durante os testes foram obtidas no **FIFA**, sendo usadas apenas para fins de desenvolvimento e experimentação.

---

# 📄 Licença

Este projeto possui fins educacionais e de aprendizado.

Contribuições, sugestões e melhorias são sempre bem-vindas.
