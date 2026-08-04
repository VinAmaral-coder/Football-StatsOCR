import cv2
import pytesseract
import csv
import os
import re
import glob
import time

from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.makedirs("debug", exist_ok=True)
os.makedirs("output", exist_ok=True)

inicio = time.perf_counter()

def recortar(imagem, x, y, w, h):
    return imagem[y:y + h, x:x + w]


def preparar_ocr(imagem, escala=4):
    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem = cv2.resize(imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imagem


def ocr_texto(imagem, config):
    return pytesseract.image_to_string(imagem, lang="por+eng", config=config).strip()


def para_float(valor):
    """Tenta converter algo (string vinda do OCR, número, etc.) pra float.
    Retorna None se não der pra converter, em vez de deixar o erro se
    propagar ou virar texto cru salvo no lugar do número."""
    try:
        return float(str(valor).replace(",", "."))
    except (ValueError, TypeError):
        return None


def ocr_overall(imagem_overall):
    """Tenta ler o overall com vários PSMs diferentes até achar um
    valor de 2 dígitos plausível. Antes, se o primeiro PSM não desse
    2 dígitos, o campo ficava vazio -- essa é a causa da maioria dos
    overalls em branco no CSV."""
    for psm in (8, 10, 7, 13):
        texto = ocr_texto(
            imagem_overall,
            f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789",
        ).replace("\n", "")
        texto = re.sub(r"\D", "", texto)
        if len(texto) == 2:
            return texto
    return ""


def limpar_nome(texto):
    """Remove lixo que o OCR às vezes captura junto do nome (dois
    pontos, dígitos, símbolos de overlay), evitando coisas como
    'Michael TE:' em vez de 'Michael Olise'."""
    texto = re.sub(r"[^A-Za-zÀ-ÿ'\- ]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    # se sobrou uma "palavra" isolada e curta demais (ex: "TE"),
    # provavelmente é ruído de overlay -- descarta pra não sujar o nome
    partes = [p for p in texto.split(" ") if len(p) > 1]
    return " ".join(partes)


# Detecta a aba pelo CONTEÚDO da imagem: acha o centro x de cada
# rótulo de aba (RESUMO / POSSE DE BOLA / FINALIZAÇÕES / PASSES /
# DEFESA / GL) via OCR, acha o sublinhado rosa que marca a aba ativa
# e vê de qual rótulo ele está mais perto. Não depende do nome do
# arquivo (os prints do Windows não têm essa informação no nome).

GRUPOS_ABA = {
    "resumo": ["RESUMO"],
    "posse": ["POSSE", "DE", "BOLA"],
    "finalizacoes": ["FINALIZACOES", "FINALIZAÇÕES", "FINALIZAÇOES", "FINALIZACOES."],
    "passes": ["PASSES"],
    "defesa": ["DEFESA"],
    "gl": ["GL"],
}


def detectar_aba(img):
    altura_img, largura_img = img.shape[:2]
    # região aproximada da barra de abas, escalada pela resolução
    # (medida em cima de screenshots 1920x1080)
    fx, fy = largura_img / 1920, altura_img / 1080
    bx, by = int(700 * fx), int(150 * fy)
    bw, bh = int(1200 * fx), int(70 * fy)
    faixa = img[by:by + bh, bx:bx + bw]

    if faixa.size == 0:
        return "resumo"

    # 1) OCR pra achar o centro x de cada rótulo de aba
    gray = cv2.cvtColor(faixa, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dados = pytesseract.image_to_data(
        otsu, lang="por+eng", config="--oem 3 --psm 6", output_type=Output.DICT
    )

    palavras = []
    for i in range(len(dados["text"])):
        t = dados["text"][i].strip().upper()
        if t:
            x_centro = bx + (dados["left"][i] + dados["width"][i] / 2) / 3
            palavras.append((x_centro, t))

    centros = {}
    for aba, alvos in GRUPOS_ABA.items():
        xs = [x for x, t in palavras if t in alvos]
        if xs:
            centros[aba] = sum(xs) / len(xs)

    # se não achou pelo menos a maioria dos rótulos, algo saiu muito
    # errado (imagem cortada, resolução não prevista) -- não arrisca
    if len(centros) < 4:
        return "resumo"

    # 2) acha a linha rosa (sublinhado da aba ativa) por cor, dentro
    # da mesma faixa
    melhor_y, melhor_mask, melhor_qtd = None, None, 0
    for y in range(faixa.shape[0]):
        linha = faixa[y]
        mask = (linha[:, 2] > 180) & (linha[:, 1] < 90) & (linha[:, 0] > 90) & (linha[:, 0] < 200)
        qtd = int(mask.sum())
        if qtd > melhor_qtd:
            melhor_qtd, melhor_y, melhor_mask = qtd, y, mask

    if melhor_y is None or melhor_qtd < 10:
        return "resumo"

    xs_rosa = [bx + x for x in range(len(melhor_mask)) if melhor_mask[x]]
    centro_rosa = sum(xs_rosa) / len(xs_rosa)

    return min(centros, key=lambda aba: abs(centros[aba] - centro_rosa))


# Rótulos de cada aba, na ordem em que aparecem na tela.
# None = linha de cabeçalho de seção (ex: "POSSE (GERAL)"), não tem valor.

ROTULOS_RESUMO = [
    "gols", "assistencias", "finalizacoes", "precisao_finalizacoes_pct",
    "passes", "precisao_passes_pct", "dribles", "taxa_dribles_certos_pct",
    "divididas", "taxa_divididas_ganhas_pct", "impedimentos", "faltas_cometidas",
    "posses_bola_ganhas", "perdas_posse_bola", "minutos_jogados_media_time",
    "distancia_percorrida_media_time_km", "distancia_corrida_media_time_km",
]

ROTULOS_POSSE = [
    None, "posse_de_bola_pct", "dribles", "dribles_completos", "taxa_dribles_certos_pct",
    "distancia_conduzindo_km", "faltas_sofridas", "penaltis_sofridos",
    "conducao_normal_pct", "conducao_protecao_pct", "conducao_lateral_pct",
    None, "superando_oponentes", "superando_oponentes_fintas", "no_meio_das_pernas", "adiantadas",
]

ROTULOS_FINALIZACOES = [
    None, "gols", "gols_esperados", "finalizacoes", "certas", "erradas", "bloqueadas", "precisao_pct",
    None, "normal", "colocada", "cabeceio", "rasteira", "de_voleio", "por_cobertura", "bola_parada",
]

ROTULOS_PASSES = [
    None, "assistencias", "assistencias_esperadas", "passes", "concluidos", "interceptados",
    "precisao_passe_pct", "para_impedimento",
    None, "rasteiro", "por_cobertura", "enfiada", "enfiada_por_cima", "cruzamento", "bola_parada",
]

ROTULOS_DEFESA = [
    None, "divididas_em_pe", "divididas_em_pe_ganhas", "taxa_divididas_em_pe_ganhas_pct",
    "carrinhos", "carrinhos_certos", "taxa_carrinhos_certos_pct", "interceptacoes",
    "bloqueios", "chutoes", "disputas_aereas_ganhas", "duelos_perdidos",
    None, "faltas_cometidas", "penaltis_cometidos", "gols_contra",
]

ROTULOS_GL = [
    None, "finalizacoes_contra", "finalizacoes_certas", "defesas", "gols_sofridos",
    "taxa_defesas_pct", "defesas_penaltis", "gols_sofridos_penaltis",
    None, "normal", "colocada", "cabeceio", "rasteira", "de_voleio", "por_cobertura", "bola_parada",
]

ABAS = {
    "resumo": ROTULOS_RESUMO,
    "posse": ROTULOS_POSSE,
    "finalizacoes": ROTULOS_FINALIZACOES,
    "passes": ROTULOS_PASSES,
    "defesa": ROTULOS_DEFESA,
    "gl": ROTULOS_GL,
}

# Coordenadas da tabela de estatísticas por resolução
COORDENADAS_TABELA = {
    (1920, 1080): {"x": 1258, "y": 226, "w": 594, "h": 753},
    (1360, 768): {"x": 884, "y": 158, "w": 428, "h": 536},
}


# Acha a posição Y de cada linha da tabela usando o OCR da coluna de RÓTULOS

def linhas_da_tabela(gray, x, y, w_label, h):
    crop = gray[y:y + h, x:x + w_label]
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dados = pytesseract.image_to_data(
        otsu, lang="por+eng", config="--oem 3 --psm 6", output_type=Output.DICT
    )

    linhas = {}
    for i in range(len(dados["text"])):
        texto = dados["text"][i].strip()
        if not texto:
            continue
        chave = (dados["block_num"][i], dados["par_num"][i], dados["line_num"][i])
        topo, altura = dados["top"][i], dados["height"][i]
        if chave not in linhas:
            linhas[chave] = {"top": topo, "bottom": topo + altura}
        linhas[chave]["bottom"] = max(linhas[chave]["bottom"], topo + altura)
        linhas[chave]["top"] = min(linhas[chave]["top"], topo)
        linhas[chave].setdefault("texto", []).append(texto)

    # descarta linhas sem nenhuma letra (ruído/pontuação solta que
    # o OCR às vezes "inventa" no topo da tabela e desalinha tudo)
    validas = [l for l in linhas.values() if re.search(r"[A-Za-zÀ-ÿ]", " ".join(l["texto"]))]

    ordenadas = sorted(validas, key=lambda l: l["top"])

    # funde linhas muito próximas verticalmente: o OCR às vezes quebra
    # uma única linha da tabela em dois blocos de texto (ex: rótulo
    # longo com espaçamento irregular), o que fazia a contagem de
    # linhas "sobrar" e desalinhar tudo daquele ponto em diante.
    if ordenadas:
        alturas = [l["bottom"] - l["top"] for l in ordenadas]
        altura_media = sum(alturas) / len(alturas)
        fundidas = [dict(ordenadas[0])]
        for linha in ordenadas[1:]:
            anterior = fundidas[-1]
            gap = linha["top"] - anterior["bottom"]
            if gap < altura_media * 0.35:
                anterior["bottom"] = max(anterior["bottom"], linha["bottom"])
                anterior["top"] = min(anterior["top"], linha["top"])
            else:
                fundidas.append(dict(linha))
        ordenadas = fundidas

    return [(y + l["top"] // 2, y + l["bottom"] // 2) for l in ordenadas]


def ocr_numero(gray, y_topo, y_fundo, x_ini, x_fim, pad=4):
    crop = gray[max(0, y_topo - pad):y_fundo + pad, x_ini:x_fim]
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texto = pytesseract.image_to_string(
        otsu, lang="por+eng", config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,."
    ).strip().strip(".")
    # limpa qualquer sujeira que não seja dígito/vírgula/ponto (ex: OCR
    # "inventando" uma vírgula solta onde deveria ter lido "0")
    texto = re.sub(r"[^0-9,.]", "", texto)
    texto = texto.strip(",.")
    # linha em branco (ou só pontuação solta) quase sempre é um "0" que
    # sumiu no threshold
    return texto if texto else "0"


def ler_aba(caminho_imagem):

    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"Imagem '{caminho_imagem}' não encontrada!")

    aba = detectar_aba(img)

    altura_img, largura_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if (largura_img, altura_img) == (1360, 768):
        nome_reg = {"x": 191, "y": 148, "w": 100, "h": 45}
        overall_reg = {"x": 66, "y": 146, "w": 26, "h": 30}
    elif (largura_img, altura_img) == (1920, 1080):
        nome_reg = {"x": 266, "y": 206, "w": 150, "h": 60}
        overall_reg = {"x": 93, "y": 209, "w": 40, "h": 36}
    else:
        raise ValueError("Resolução não suportada!")

    tab = COORDENADAS_TABELA[(largura_img, altura_img)]

    # nome e overall aparecem no painel esquerdo em qualquer aba
    nome_img = preparar_ocr(recortar(img, **nome_reg))
    overall_img = preparar_ocr(recortar(img, **overall_reg))

    nome_texto = ocr_texto(
        nome_img, "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    ).replace("\n", " ")
    nome_texto = limpar_nome(nome_texto)

    overall_texto = ocr_overall(overall_img)

    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]
    cv2.imwrite(f"debug/{nome_arquivo}_nome.png", nome_img)
    cv2.imwrite(f"debug/{nome_arquivo}_overall.png", overall_img)

    resultado = {}

    if aba == "resumo":
        resultado["nome"] = nome_texto
        resultado["overall"] = overall_texto

    rotulos = ABAS[aba]
    w_label = tab["w"] - 115
    linhas = linhas_da_tabela(gray, tab["x"], tab["y"], w_label, tab["h"])

    # aviso quando a quantidade de linhas detectadas não bate com o
    # esperado -- sinal de que a tabela desta imagem desalinhou
    esperado = sum(1 for r in rotulos if r is not None)
    if len(linhas) != esperado:
        print(
            f"  [aviso] {nome_arquivo} [{aba}]: {len(linhas)} linha(s) detectada(s), "
            f"esperava {esperado}. Os valores podem estar desalinhados — "
            f"confira debug/{nome_arquivo}_{aba}_tabela.png"
        )

    # imagem de debug: tabela completa com as linhas detectadas
    # marcadas e o rótulo que foi atribuído a cada uma. Serve pra ver
    # de cara se uma linha ficou "faltando" ou "sobrando" e por isso
    # todo o resto desceu/subiu uma posição.
    debug_tabela = img[tab["y"]:tab["y"] + tab["h"], tab["x"]:tab["x"] + tab["w"]].copy()
    rotulos_nao_none = [r for r in rotulos if r is not None]
    for idx, (yt, yb) in enumerate(linhas):
        y0, y1 = yt - tab["y"], yb - tab["y"]
        cv2.rectangle(debug_tabela, (0, y0), (debug_tabela.shape[1], y1), (0, 0, 255), 1)
        texto_label = rotulos_nao_none[idx] if idx < len(rotulos_nao_none) else "???"
        cv2.putText(
            debug_tabela, texto_label, (2, max(10, y0 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA
        )
    cv2.imwrite(f"debug/{nome_arquivo}_{aba}_tabela.png", debug_tabela)

    if aba == "resumo":
        x_jogador = (tab["x"] + tab["w"] - 160, tab["x"] + tab["w"] - 67)
        x_time = (tab["x"] + tab["w"] - 67, tab["x"] + tab["w"] + 4)

        for rotulo, (yt, yb) in zip(rotulos, linhas):
            resultado[rotulo] = ocr_numero(gray, yt, yb, *x_jogador)
            resultado[f"time_{rotulo}"] = ocr_numero(gray, yt, yb, *x_time)

    else:
        x_num = (tab["x"] + tab["w"] - 115, tab["x"] + tab["w"] + 4)

        for rotulo, (yt, yb) in zip(rotulos, linhas):
            if rotulo is None:
                continue
            resultado[f"{aba}_{rotulo}"] = ocr_numero(gray, yt, yb, *x_num)

    print(f"{nome_arquivo} [{aba}] -> nome={nome_texto!r} overall={overall_texto!r}")

    return resultado, aba



PASTA_PLAYERS = os.path.join("Players", "IMGS")

jogadores = []
# colunas fixas que sempre existem; as colunas de estatística vão
# sendo descobertas conforme aparecem (dependem da aba de cada print)
todos_campos = ["jogador_pasta", "img", "aba", "nome", "overall"]


# percorre automaticamente todas as pastas dos jogadores
pastas_jogadores = sorted(
    pasta for pasta in glob.glob(os.path.join(PASTA_PLAYERS, "*"))
    if os.path.isdir(pasta)
)

print(f"\n{len(pastas_jogadores)} jogador(es) encontrado(s)")

for pasta in pastas_jogadores:

    imagens = sorted(
        f
        for ext in ("*.png", "*.jpg", "*.jpeg")
        for f in glob.glob(os.path.join(pasta, ext))
    )

    if not imagens:
        continue

    nome_pasta = os.path.basename(pasta)

    for imagem in imagens:

        dados, aba = ler_aba(imagem)

        # cada print vira UMA linha própria no CSV -- nada de somar
        # ou acumular com os outros prints do mesmo jogador
        registro = {
            "jogador_pasta": nome_pasta,
            "img": os.path.basename(imagem),
            "aba": aba,
        }

        for chave, valor in dados.items():

            if chave in ("nome", "overall"):
                registro[chave] = valor
                continue

            valor_num = para_float(valor)
            if valor_num is None:
                print(
                    f"  [aviso] valor não numérico em '{chave}': {valor!r} "
                    f"(imagem {os.path.basename(imagem)}) -- salvo em branco"
                )
                registro[chave] = ""
                continue

            registro[chave] = valor_num

        # se o OCR não leu o nome nesta imagem específica, usa o nome
        # da pasta só pra essa linha não ficar sem identificação
        if not registro.get("nome", "").strip():
            registro["nome"] = nome_pasta

        jogadores.append(registro)

        for campo in registro:
            if campo not in todos_campos:
                todos_campos.append(campo)


with open("output/jogadores.csv", "w", newline="", encoding="utf-8") as arquivo:
    writer = csv.DictWriter(arquivo, fieldnames=todos_campos, restval="")
    writer.writeheader()
    writer.writerows(jogadores)

print("\nCSV criado com sucesso!")

fim = time.perf_counter()
tempo = fim - inicio

minutos = int(tempo // 60)
segundos = tempo % 60
print(f"\nTempo de processamento: {minutos} min {segundos:.2f} s")