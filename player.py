import cv2
import pytesseract
import csv
import os
import re
import glob
import time
from pytesseract import Output
from concurrent.futures import ProcessPoolExecutor, as_completed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------
# Configurações e Rótulos
# ---------------------------------------------------------------

CAMPOS_CONTAGEM_PEQUENA = {
    "gols", "assistencias", "finalizacoes", "divididas",
    "impedimentos", "faltas_cometidas",
}

ROTULOS_RESUMO = [
    "gols", "assistencias", "finalizacoes", "precisao_finalizacoes_pct",
    "passes", "precisao_passes_pct", "dribles", "taxa_dribles_certos_pct",
    "divididas", "taxa_divididas_ganhas_pct", "impedimentos", "faltas_cometidas",
    "posses_bola_ganhas", "perdas_posse_bola", "minutos_jogados_media_time",
    "distancia_percorrida_media_time_km", "distancia_corrida_media_time_km",
]

COORDENADAS = {
    (1920, 1080): {
        "nome_crop": (170, 150, 260, 130),
        "overall_crop": (92, 208, 42, 41),
        "tabela": {"x": 1258, "y": 226, "w": 594, "h": 753},
        "linha0_topo": 71,
        "passo_linha": 39,
        "col_jogador": (410, 505), 
        "col_time": (512, 594),    
    },
}

# ---------------------------------------------------------------
# Funções de OCR Isoladas por Coluna
# ---------------------------------------------------------------

def limpar_nome(texto):
    candidatos = re.findall(r"[A-Za-zÀ-ÿ'\-]+", texto)
    validos = [
        p for p in candidatos
        if re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ]*(?:[-'][A-ZÀ-Ýa-zà-ÿ]+)*", p) and len(p) > 1
    ]
    return validos

def ocr_nome_preciso(gray, x, y, w, h, nome_pasta):
    crop = gray[y:y + h, x:x + w]
    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    realce = clahe.apply(big)
    _, otsu = cv2.threshold(realce, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fixo = cv2.threshold(big, 180, 255, cv2.THRESH_BINARY_INV)

    candidatos = []
    for img in (otsu, fixo):
        texto = pytesseract.image_to_string(img, lang="por+eng", config="--oem 3 --psm 6").strip()
        tokens = [t for l in texto.split("\n") for t in limpar_nome(l)]
        if tokens:
            candidatos.append(" ".join(tokens))

    return max(candidatos, key=len) if candidatos else nome_pasta

def ocr_overall_preciso(gray, x, y, w, h):
    crop = gray[max(0, y-2):y + h + 2, max(0, x-2):x + w + 2]
    if crop.size == 0:
        return ""
    
    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, fixo = cv2.threshold(big, 150, 255, cv2.THRESH_BINARY_INV)
    
    texto = pytesseract.image_to_string(
        fixo, config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()
    
    digitos = re.sub(r"\D", "", texto)
    return digitos[:2]

def extrair_coluna_isolada(gray_tabela, faixa_x, linha0_topo, passo_linha):
    x1, x2 = faixa_x
    col_crop = gray_tabela[:, x1:x2]
    
    big = cv2.resize(col_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    realce = clahe.apply(big)
    _, fixo = cv2.threshold(realce, 140, 255, cv2.THRESH_BINARY_INV)

    dados = pytesseract.image_to_data(
        fixo, 
        config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.", 
        output_type=Output.DICT
    )

    resultados_linha = {}
    for i in range(len(dados["text"])):
        texto = dados["text"][i].strip()
        if not re.fullmatch(r"[0-9,.]+", texto):
            continue

        y_topo = dados["top"][i] / 3
        indice = round((y_topo - linha0_topo) / passo_linha)
        
        if 0 <= indice < len(ROTULOS_RESUMO):
            esperado = linha0_topo + indice * passo_linha
            if abs(y_topo - esperado) <= passo_linha / 2:
                resultados_linha[indice] = texto.strip(",.")

    return resultados_linha

def extrair_stats_resumo(gray, coords):
    tabela = coords["tabela"]
    linha0_topo = coords["linha0_topo"]
    passo_linha = coords["passo_linha"]

    crop_tabela = gray[tabela["y"]:tabela["y"] + tabela["h"], tabela["x"]:tabela["x"] + tabela["w"]]

    # Processamento independente de cada coluna
    dados_jogador = extrair_coluna_isolada(crop_tabela, coords["col_jogador"], linha0_topo, passo_linha)
    dados_time = extrair_coluna_isolada(crop_tabela, coords["col_time"], linha0_topo, passo_linha)

    registro = {}
    for idx, rotulo in enumerate(ROTULOS_RESUMO):
        val_jog = dados_jogador.get(idx, "")
        val_time = dados_time.get(idx, "")

        # Trata o zero ausente: se o campo é uma contagem/porcentagem e veio em branco, assume '0'
        if not val_jog:
            val_jog = "0"
        if not val_time:
            val_time = "0"

        # Formatação de campos de distância (_km)
        if rotulo.endswith("_km"):
            if "," not in val_jog and len(val_jog) == 2:
                val_jog = f"{val_jog[0]},{val_jog[1]}"
            if "," not in val_time and len(val_time) == 2:
                val_time = f"{val_time[0]},{val_time[1]}"

        registro[f"{rotulo}_jogador"] = val_jog
        registro[f"{rotulo}_time"] = val_time

    return registro

# ---------------------------------------------------------------
# Processamento de Imagem Individual
# ---------------------------------------------------------------

def processar_imagem(item):
    nome_pasta, imagem_path = item
    img = cv2.imread(imagem_path)
    if img is None:
        return None

    altura, largura = img.shape[:2]
    coords = COORDENADAS.get((largura, altura))
    if coords is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nx, ny, nw, nh = coords["nome_crop"]
    nome = ocr_nome_preciso(gray, nx, ny, nw, nh, nome_pasta)

    ox, oy, ow, oh = coords["overall_crop"]
    overall = ocr_overall_preciso(gray, ox, oy, ow, oh)

    registro = {
        "jogador_pasta": nome_pasta,
        "img": os.path.basename(imagem_path),
        "nome": nome,
        "overall": overall,
    }

    registro.update(extrair_stats_resumo(gray, coords))
    return registro

# ---------------------------------------------------------------
# Execução Principal
# ---------------------------------------------------------------

PASTA_PLAYERS = "Imagens"
EXTENSOES = ("*.png", "*.jpg", "*.jpeg")

def listar_imagens(pasta):
    return sorted(f for ext in EXTENSOES for f in glob.glob(os.path.join(pasta, ext)))

def coletar_imagens():
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

if __name__ == "__main__":
    inicio = time.perf_counter()

    todos_campos = ["jogador_pasta", "img", "nome", "overall"]
    for rotulo in ROTULOS_RESUMO:
        todos_campos.append(f"{rotulo}_jogador")
        todos_campos.append(f"{rotulo}_time")

    itens = coletar_imagens()
    total = len(itens)
    print(f"\n{total} imagem(ns) encontrada(s). Processando...")

    jogadores = []
    max_workers = os.cpu_count() or 4

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(processar_imagem, item): item for item in itens}
        
        concluidos = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                jogadores.append(res)
            
            concluidos += 1
            if concluidos % 10 == 0 or concluidos == total:
                decorrido = time.perf_counter() - inicio
                print(f"  [{concluidos}/{total}] concluídas em {decorrido:.2f}s")

    jogadores = sorted(jogadores, key=lambda x: (x["jogador_pasta"], x["img"]))

    with open("output/jogadores.csv", "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=todos_campos, restval="")
        writer.writeheader()
        writer.writerows(jogadores)

    tempo = time.perf_counter() - inicio
    minutos = int(tempo // 60)
    segundos = tempo % 60
    print(f"\nCSV gerado em 'output/jogadores.csv'!")
    print(f"Tempo total: {minutos} min {segundos:.2f} s")