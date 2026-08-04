import cv2
import pytesseract
import csv
import os
import re
import glob
import time
from collections import Counter
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.makedirs("output", exist_ok=True)

inicio = time.perf_counter()


# ---------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------

def recortar(imagem, x, y, w, h):
    return imagem[y:y + h, x:x + w]


def para_float(valor):
    """Converte string do OCR (com vírgula ou ponto) pra float.
    Retorna None se não der pra converter."""
    try:
        return float(str(valor).replace(",", "."))
    except (ValueError, TypeError):
        return None


def limpar_nome(texto):
    """Extrai a MAIOR palavra em Title Case (primeira letra
    maiúscula, resto minúsculo, com direito a hífen/apóstrofo pra
    nomes compostos tipo Jean-Philippe ou Rak-Sakyi) de UMA linha
    de texto. Retorna '' se não achar nenhuma."""
    candidatos = re.findall(r"[A-Za-zÀ-ÿ'\-]+", texto)
    validos = [
        p for p in candidatos
        if re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ]*(?:[-'][A-ZÀ-Ýa-zà-ÿ]+)*", p) and len(p) > 1
    ]
    if not validos:
        return ""
    return max(validos, key=len)


def _variantes_para_ocr_nome(crop_gray):
    """Gera 4 versões binarizadas do recorte do nome pra tentar.
    O CLAHE (realce de contraste local) entra porque o primeiro
    nome é escrito numa cor bem mais fraca/escura que o sobrenome
    -- um threshold só (fixo ou OTSU simples) muitas vezes lê o
    sobrenome certinho mas perde o primeiro nome inteiro ou lê
    ele como lixo. Com CLAHE, o contraste do primeiro nome fica
    realçado o suficiente pra aparecer."""
    big = cv2.resize(crop_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    variantes = []

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    realce = clahe.apply(big)
    _, otsu_c = cv2.threshold(realce, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu_c)
    variantes.append(cv2.bitwise_not(otsu_c))

    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu)
    _, fixo = cv2.threshold(big, 190, 255, cv2.THRESH_BINARY)
    variantes.append(cv2.bitwise_not(fixo))

    return variantes


def ocr_nome(gray, x, y, w, h, nome_pasta):
    """OCR do nome do jogador (2 linhas: primeiro nome embaixo o
    sobrenome).

    A ideia central: deixa o TESSERACT separar as linhas sozinho
    (ele já faz isso bem, mesmo com ruído -- cada ruído fica na
    MESMA linha do nome real, nunca sozinho numa linha própria) e,
    IGNORANDO ordem entre tentativas diferentes, pega a MAIOR
    palavra válida de CADA linha -- como o lixo (ícones, chevrons
    lidos como texto) é sempre mais curto que o nome de verdade,
    isso naturalmente descarta o lixo e mantém o nome.

    Tenta 4 binarizações diferentes (ver _variantes_para_ocr_nome)
    e, entre as que renderam as 2 linhas esperadas, fica com a de
    maior comprimento total (o nome certo quase sempre "ganha" das
    leituras parciais/erradas por ser mais completo)."""
    crop = recortar(gray, x, y, w, h)

    candidatos_2_linhas = []
    melhor_qualquer = ""

    for variante in _variantes_para_ocr_nome(crop):
        texto = pytesseract.image_to_string(variante, lang="por+eng", config="--oem 3 --psm 6").strip()
        linhas = [l for l in texto.split("\n") if l.strip()]
        tokens = [limpar_nome(l) for l in linhas]
        tokens = [t for t in tokens if t]

        if len(tokens) >= 2:
            candidatos_2_linhas.append(" ".join(tokens[:2]))

        texto_completo = " ".join(tokens)
        if len(texto_completo) > len(melhor_qualquer):
            melhor_qualquer = texto_completo

    if candidatos_2_linhas:
        return max(candidatos_2_linhas, key=len)
    if melhor_qualquer:
        return melhor_qualquer
    return nome_pasta


def ocr_overall(gray, x1, y1, x2, y2, escala=6, limiares=(170, 150, 190, 210)):
    """OCR específico pro distintivo de overall (sempre 2 dígitos).

    Descoberta importante: pedir pro tesseract ler os 2 dígitos
    JUNTOS (ex: "79") às vezes lê errado de forma CONSISTENTE (não
    é ruído aleatório) -- em várias imagens reais, "79" sai sempre
    como "19" em toda tentativa de threshold/PSM, porque a fonte
    do jogo desenha o "7" de um jeito que o modelo confunde com
    "1" quando vê os dois dígitos juntos.

    A correção: cortar a célula ao MEIO (metade esquerda = dígito
    1, metade direita = dígito 2) e pedir pro tesseract ler CADA
    METADE SOZINHA como um único caractere (psm 8/10/13). Isoladas,
    a forma de cada dígito fica sem ambiguidade e o "7" deixa de
    ser confundido com "1". Testado contra 4 imagens reais
    (incluindo casos que davam "19" no lugar de "79") e bateu 100%
    nas 4, sem quebrar o caso que já funcionava.
    """
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    largura = crop.shape[1]
    metade_esquerda = crop[:, :largura // 2]
    metade_direita = crop[:, largura // 2:]

    def ler_um_digito(pedaco):
        big = cv2.resize(pedaco, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
        candidatos = []
        for limiar in limiares:
            _, fixo = cv2.threshold(big, limiar, 255, cv2.THRESH_BINARY)
            fixo_inv = cv2.bitwise_not(fixo)
            for psm in (8, 13, 10):
                texto = pytesseract.image_to_string(
                    fixo_inv, config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
                ).strip()
                if len(texto) == 1 and texto.isdigit():
                    candidatos.append(texto)
            if candidatos:
                break  # esse threshold já deu um resultado de 1 dígito, não precisa dos outros
        if not candidatos:
            return ""
        return Counter(candidatos).most_common(1)[0][0]

    digito1 = ler_um_digito(metade_esquerda)
    digito2 = ler_um_digito(metade_direita)

    if digito1 and digito2:
        return digito1 + digito2
    return ""


def ocr_numero(gray, x1, y1, x2, y2, escala=6, exigir_digitos=None, limiares=(170, 150, 190, 210)):
    """OCR de uma célula numérica. Tenta os thresholds na ordem de
    'limiares' -- 170 primeiro, porque é o que calibramos como
    melhor pra essa tabela. Só escala pros próximos thresholds se
    170 não achar nenhum dígito. Dentro de cada threshold, faz uma
    mini-votação entre os 4 PSMs."""
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    big = cv2.resize(crop, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)

    for limiar in limiares:
        _, fixo = cv2.threshold(big, limiar, 255, cv2.THRESH_BINARY)
        fixo_inv = cv2.bitwise_not(fixo)

        candidatos = []
        for psm in (7, 8, 10, 13):
            texto = pytesseract.image_to_string(
                fixo_inv, config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789,"
            ).strip()
            numeros = re.findall(r"\d+[.,]?\d*", texto)
            if not numeros:
                continue
            valor = numeros[0].strip(",.")
            if exigir_digitos is not None:
                so_digitos = re.sub(r"\D", "", valor)
                if len(so_digitos) != exigir_digitos:
                    continue
            candidatos.append(valor)

        if candidatos:
            contagem = Counter(candidatos)
            # em caso de empate na votação, prefere o valor mais
            # CURTO -- o artefato conhecido dessa tabela (rótulo
            # comprido encostando no número, ex: "(km)0,4") sempre
            # ADICIONA um dígito a mais na frente do valor certo,
            # nunca remove um dígito. Então entre "40,4" e "0,4"
            # empatados, "0,4" (mais curto) é o correto.
            melhor = min(contagem.items(), key=lambda kv: (-kv[1], len(kv[0])))
            return melhor[0]

    return ""


# ---------------------------------------------------------------
# Filtro de sanidade (novo na v3)
# ---------------------------------------------------------------

CAMPOS_CONTAGEM_PEQUENA = {
    "gols", "assistencias", "finalizacoes", "divididas",
    "impedimentos", "faltas_cometidas",
}


def normalizar_valor(chave, valor):
    """Corrige 2 padrões de erro conhecidos (ver explicação no
    topo do arquivo). 'chave' é o nome da coluna do CSV, tipo
    'gols_jogador' ou 'precisao_finalizacoes_pct_time'."""
    if not valor:
        return valor

    inteiro, _, decimal = valor.partition(",")
    if not inteiro.isdigit():
        return valor

    base, _, coluna = chave.rpartition("_")  # 'gols_jogador' -> ('gols','_','jogador')

    if coluna == "jogador" and base in CAMPOS_CONTAGEM_PEQUENA and len(inteiro) == 2 and inteiro[0] == inteiro[1]:
        inteiro = inteiro[0]

    if base.endswith("_pct") and int(inteiro) > 100:
        inteiro = inteiro[-2:]

    return inteiro + ("," + decimal if decimal else "")


# ---------------------------------------------------------------
# Rótulos da tabela RESUMO, na ordem fixa em que aparecem no jogo
# ---------------------------------------------------------------

ROTULOS_RESUMO = [
    "gols", "assistencias", "finalizacoes", "precisao_finalizacoes_pct",
    "passes", "precisao_passes_pct", "dribles", "taxa_dribles_certos_pct",
    "divididas", "taxa_divididas_ganhas_pct", "impedimentos", "faltas_cometidas",
    "posses_bola_ganhas", "perdas_posse_bola", "minutos_jogados_media_time",
    "distancia_percorrida_media_time_km", "distancia_corrida_media_time_km",
]

# ---------------------------------------------------------------
# Coordenadas calibradas por resolução
# ---------------------------------------------------------------

COORDENADAS = {
    (1920, 1080): {
        "nome_crop": (170, 150, 260, 130),
        "overall_crop": (92, 208, 42, 41),
        "tabela": {"x": 1258, "y": 226, "w": 594, "h": 753},
        "linha0_topo": 71,
        "passo_linha": 39,
        "col_jogador": (410, 512),
        "col_time": (512, 594),
    },
}


def extrair_stats_resumo_rapido(gray, coords):
    """Lê as 17 linhas x 2 colunas da tabela RESUMO com UMA chamada
    de OCR na tabela inteira, localizando cada número pela posição
    (linha esperada = topo + índice*passo; coluna = jogador ou
    time pela faixa de x). Só recorre à célula individual pros
    poucos casos em que a chamada em massa não achou nada."""
    tabela = coords["tabela"]
    linha0_topo = coords["linha0_topo"]
    passo_linha = coords["passo_linha"]
    col_jogador = coords["col_jogador"]
    col_time = coords["col_time"]

    crop = gray[tabela["y"]:tabela["y"] + tabela["h"], tabela["x"]:tabela["x"] + tabela["w"]]
    escala = 4
    big = cv2.resize(crop, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    _, fixo = cv2.threshold(big, 170, 255, cv2.THRESH_BINARY)
    fixo = cv2.bitwise_not(fixo)

    dados = pytesseract.image_to_data(fixo, config="--oem 3 --psm 6", output_type=Output.DICT)

    registro = {}
    for i in range(len(dados["text"])):
        texto = dados["text"][i].strip()
        if not re.fullmatch(r"[0-9,.]+", texto):
            continue

        x_centro = (dados["left"][i] + dados["width"][i] / 2) / escala
        y_topo = dados["top"][i] / escala

        indice = round((y_topo - linha0_topo) / passo_linha)
        if not (0 <= indice < len(ROTULOS_RESUMO)):
            continue

        esperado = linha0_topo + indice * passo_linha
        if abs(y_topo - esperado) > passo_linha / 2:
            continue

        rotulo = ROTULOS_RESUMO[indice]
        if col_jogador[0] <= x_centro < col_jogador[1]:
            chave = f"{rotulo}_jogador"
        elif col_time[0] <= x_centro < col_time[1]:
            chave = f"{rotulo}_time"
        else:
            continue

        registro[chave] = texto.strip(",.")

    # os 2 campos de distância (_km) sempre têm vírgula (formato
    # "X,Y" -- nunca é um número inteiro puro). Se a passada em
    # massa leu sem vírgula (ex: "44" em vez de "4,4"), o valor
    # está incompleto -- descarta pra forçar o fallback abaixo a
    # tentar de novo nessa célula específica.
    for rotulo in ("distancia_percorrida_media_time_km", "distancia_corrida_media_time_km"):
        for coluna in ("jogador", "time"):
            chave = f"{rotulo}_{coluna}"
            if registro.get(chave) and "," not in registro[chave]:
                registro[chave] = ""

    # fallback pontual só pras células que ficaram faltando
    for indice, rotulo in enumerate(ROTULOS_RESUMO):
        topo = tabela["y"] + linha0_topo + indice * passo_linha
        y1, y2 = topo - 4, topo + passo_linha - 16
        for coluna, faixa in (("jogador", col_jogador), ("time", col_time)):
            chave = f"{rotulo}_{coluna}"
            if not registro.get(chave):
                x1 = tabela["x"] + faixa[0]
                x2 = tabela["x"] + faixa[1]
                registro[chave] = ocr_numero(gray, x1, y1, x2, y2)

    # aplica o filtro de sanidade em cada valor antes de devolver
    for chave, valor in list(registro.items()):
        registro[chave] = normalizar_valor(chave, valor)

    return registro


# ---------------------------------------------------------------
# Localização das imagens -- aceita as duas estruturas de pasta
# ---------------------------------------------------------------

PASTA_PLAYERS = os.path.join("Imagens")
EXTENSOES = ("*.png", "*.jpg", "*.jpeg")


def listar_imagens(pasta):
    return sorted(f for ext in EXTENSOES for f in glob.glob(os.path.join(pasta, ext)))


def coletar_imagens():
    """Retorna uma lista de (jogador_pasta, caminho_da_imagem).

    Junta as duas estruturas possíveis:
      a) Imagens/<jogador>/*.png|jpg|jpeg  -> jogador_pasta = nome da subpasta
      b) Imagens/*.png|jpg|jpeg (soltas)   -> jogador_pasta = "" (sem pasta pra confiar)
    """
    resultado = []

    subpastas = sorted(
        pasta for pasta in glob.glob(os.path.join(PASTA_PLAYERS, "*"))
        if os.path.isdir(pasta)
    )
    for pasta in subpastas:
        nome_pasta = os.path.basename(pasta)
        for imagem_path in listar_imagens(pasta):
            resultado.append((nome_pasta, imagem_path))

    for imagem_path in listar_imagens(PASTA_PLAYERS):
        resultado.append(("", imagem_path))

    return resultado


# ---------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------

jogadores = []

todos_campos = ["jogador_pasta", "img", "nome", "overall"]
for rotulo in ROTULOS_RESUMO:
    todos_campos.append(f"{rotulo}_jogador")
    todos_campos.append(f"{rotulo}_time")

itens = coletar_imagens()
total = len(itens)
print(f"\n{total} imagem(ns) encontrada(s)")

for indice_atual, (nome_pasta, imagem_path) in enumerate(itens, start=1):

    img = cv2.imread(imagem_path)
    if img is None:
        print(f"  [aviso] não consegui abrir {imagem_path}")
        continue

    altura, largura = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coords = COORDENADAS.get((largura, altura))
    if coords is None:
        print(f"  [aviso] resolução {largura}x{altura} sem calibração, pulando {imagem_path}")
        continue

    # ---------------- NOME ----------------
    nx, ny, nw, nh = coords["nome_crop"]
    nome = ocr_nome(gray, nx, ny, nw, nh, nome_pasta)

    # ---------------- OVERALL ----------------
    ox, oy, ow, oh = coords["overall_crop"]
    overall = ocr_overall(gray, ox, oy, ox + ow, oy + oh)

    registro = {
        "jogador_pasta": nome_pasta,
        "img": os.path.basename(imagem_path),
        "nome": nome,
        "overall": overall,
    }

    # ---------------- TABELA RESUMO ----------------
    registro.update(extrair_stats_resumo_rapido(gray, coords))

    jogadores.append(registro)

    if indice_atual % 20 == 0 or indice_atual == total:
        decorrido = time.perf_counter() - inicio
        media = decorrido / indice_atual
        restante = media * (total - indice_atual)
        print(f"  [{indice_atual}/{total}] {nome_pasta or '(solta)'}/{os.path.basename(imagem_path)} "
              f"-- {decorrido:.0f}s decorridos, ~{restante:.0f}s restantes")


with open("output/jogadores.csv", "w", newline="", encoding="utf-8") as arquivo:
    writer = csv.DictWriter(arquivo, fieldnames=todos_campos, restval="")
    writer.writeheader()
    writer.writerows(jogadores)

print("\nCSV criado com sucesso em output/jogadores.csv!")

fim = time.perf_counter()
tempo = fim - inicio
minutos = int(tempo // 60)
segundos = tempo % 60
print(f"Tempo de processamento: {minutos} min {segundos:.2f} s")