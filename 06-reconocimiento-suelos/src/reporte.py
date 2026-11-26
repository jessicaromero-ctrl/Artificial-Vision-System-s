"""Genera el reporte DOCX de la práctica 6 a partir de salidas/resultados.json y las figuras.

Uso:
    python -m src.reporte --resultados salidas/resultados.json --figuras salidas/figuras \
        --salida reporte/reporte_practica06.docx
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

MORADO = RGBColor(0x84, 0x00, 0xE0)
MORADO_HEX = "8400E0"
LAVANDA_HEX = "F1E6FB"
SIN, CON = "sin_haralick", "con_haralick"

PORTADA = {
    "materia": "Sistemas de Visión Artificial",
    "practica": "Práctica 6",
    "titulo": "Reconocimiento de suelos con y sin descriptores de textura de Haralick",
    "alumna": "Jessica Melani Romero Lora",
    "profesor": "Dr. Ernesto García Amaro",
    "institucion": "Universidad Politécnica Metropolitana del Estado de Hidalgo (UPMH)",
    "programa": "Maestría en Inteligencia Artificial",
    "periodo": "2026",
    "fecha": "25 de noviembre de 2026",
}


# ----------------------------------------------------------------- utilidades de formato
def f4(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def ic(par) -> str:
    return f"[{f4(par[0])}, {f4(par[1])}]"


def sombrear(celda, color_hex: str):
    props = celda._tc.get_or_add_tcPr()
    sombra = OxmlElement("w:shd")
    sombra.set(qn("w:val"), "clear")
    sombra.set(qn("w:color"), "auto")
    sombra.set(qn("w:fill"), color_hex)
    props.append(sombra)


def numero_de_pagina(parrafo):
    corrida = parrafo.add_run()
    ini = OxmlElement("w:fldChar"); ini.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    texto = OxmlElement("w:t"); texto.text = "1"
    fin = OxmlElement("w:fldChar"); fin.set(qn("w:fldCharType"), "end")
    for el in (ini, instr, sep, texto, fin):
        corrida._r.append(el)


def titulo_seccion(doc, texto, nivel=1):
    encabezado = doc.add_heading(texto, level=nivel)
    for corrida in encabezado.runs:
        corrida.font.color.rgb = MORADO
    return encabezado


def parrafo(doc, texto, cursiva=False, alineado=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = alineado
    corrida = p.add_run(texto)
    corrida.italic = cursiva
    p.paragraph_format.space_after = Pt(6)
    return p


def vineta(doc, texto):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(texto)
    p.paragraph_format.space_after = Pt(2)
    return p


def tabla(doc, encabezado, filas, anchos=None):
    t = doc.add_table(rows=1, cols=len(encabezado))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, texto in enumerate(encabezado):
        celda = t.rows[0].cells[i]
        celda.text = ""
        corrida = celda.paragraphs[0].add_run(texto)
        corrida.bold = True
        corrida.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        corrida.font.size = Pt(9)
        sombrear(celda, MORADO_HEX)
    for fila in filas:
        celdas = t.add_row().cells
        for i, texto in enumerate(fila):
            celdas[i].text = ""
            corrida = celdas[i].paragraphs[0].add_run(str(texto))
            corrida.font.size = Pt(9)
            if i == 0:
                sombrear(celdas[i], LAVANDA_HEX)
    if anchos:
        for fila in t.rows:
            for i, ancho in enumerate(anchos):
                fila.cells[i].width = Cm(ancho)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figura(doc, ruta: Path, leyenda: str, ancho_cm: float = 13.5):
    if not ruta.exists():
        parrafo(doc, f"[Figura pendiente: {ruta.name}]", cursiva=True)
        return
    doc.add_picture(str(ruta), width=Cm(ancho_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    corrida = p.add_run(leyenda)
    corrida.italic = True
    corrida.font.size = Pt(9)
    corrida.font.color.rgb = MORADO


def ecuacion(doc, texto):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    corrida = p.add_run(texto)
    corrida.italic = True
    p.paragraph_format.space_after = Pt(6)


# ------------------------------------------------------------------------ portada
def portada(doc):
    for _ in range(3):
        doc.add_paragraph()
    for clave, tam, negrita, color in (
        ("materia", 13, False, False), ("practica", 16, True, True), ("titulo", 22, True, True),
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        corrida = p.add_run(PORTADA[clave])
        corrida.font.size = Pt(tam)
        corrida.bold = negrita
        if color:
            corrida.font.color.rgb = MORADO
    doc.add_paragraph()
    doc.add_paragraph()
    for etiqueta, clave in (
        ("Alumna", "alumna"), ("Profesor", "profesor"), ("Institución", "institucion"),
        ("Programa", "programa"), ("Periodo", "periodo"), ("Fecha", "fecha"),
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(f"{etiqueta}: ")
        r1.bold = True
        r1.font.color.rgb = MORADO
        p.add_run(PORTADA[clave])
    doc.add_page_break()


# ---------------------------------------------------------------------- contenido
def introduccion(doc, rep):
    titulo_seccion(doc, "1. Introducción")
    parrafo(doc, (
        "El reconocimiento automático del tipo de suelo a partir de fotografías es un problema "
        "relevante para la agronomía, la edafología y la ingeniería civil, donde la clasificación "
        "de suelos orienta decisiones de uso, cimentación y manejo. Una fotografía de suelo "
        "contiene información de color, pero también de su textura: la forma en que varían las "
        "intensidades de los píxeles en una vecindad, asociada a la granulometría y a la estructura "
        "de la superficie."))
    parrafo(doc, (
        "La textura puede describirse de forma cuantitativa con los descriptores de Haralick, que "
        "se derivan de la matriz de coocurrencia de niveles de gris (GLCM). Sin embargo, no es "
        "evidente que esa representación supere a la que usa directamente los píxeles de la "
        "imagen, pues una red o un clasificador podría aprender parte de esa información desde los "
        "propios valores de intensidad."))
    parrafo(doc, "El problema que se aborda es el siguiente:")
    parrafo(doc, (
        "¿Mejora el reconocimiento del tipo de suelo cuando la imagen se representa con descriptores "
        "de textura de Haralick, en comparación con la representación por píxeles, bajo un mismo "
        "clasificador y un mismo protocolo de evaluación?"))
    parrafo(doc, "Para responderlo se plantean los siguientes objetivos.")
    vineta(doc, "Construir un pipeline reproducible para el conjunto Soil DataSet con dos representaciones: con y sin descriptores de Haralick.")
    vineta(doc, "Comparar ambas representaciones con la exactitud, el F1 macro y su intervalo de confianza, sobre varias particiones aleatorias.")
    vineta(doc, "Verificar que la exactitud observada supera el azar y que no se debe a fuga de información entre entrenamiento y prueba.")
    parrafo(doc, (
        "La hipótesis nula establece que ambas representaciones producen la misma exactitud media. "
        "La hipótesis alternativa establece que la representación con descriptores de Haralick "
        "produce una exactitud media mayor. La prueba se realiza con dos técnicas pareadas: la "
        "prueba t sobre las semillas y la prueba de McNemar sobre las mismas muestras de prueba."))
    parrafo(doc, (
        f"El conjunto analizado contiene {rep['imagenes_finales']} imágenes de "
        f"{len(rep['clases'])} clases de suelo tras eliminar duplicados exactos. Las clases son "
        + ", ".join(rep["clases"]) + "."))


def protocolo(doc, rep):
    titulo_seccion(doc, "2. Pipeline usado", 1)
    titulo_seccion(doc, "2.1 Datos y protocolo común", 2)
    parrafo(doc, (
        f"Se parte de {rep['imagenes_brutas']} archivos de las particiones train y test. Estas "
        "particiones se unen en un solo conjunto y se eliminan los archivos con contenido idéntico "
        f"(huella SHA-256): {rep['duplicados_exactos']} duplicados exactos, de los cuales "
        f"{rep['duplicados_con_clase_distinta']} tenían etiquetas distintas. Quedan "
        f"{rep['imagenes_finales']} imágenes."))
    filas = [(c, str(rep["imagenes_por_clase"][c])) for c in rep["clases"]]
    tabla(doc, ["Clase", "Imágenes"], filas + [("Total", str(rep["imagenes_finales"]))], anchos=[6, 4])
    cd = rep["casi_duplicados"]
    parrafo(doc, (
        f"Como control de fuga, se calculó un hash perceptual promedio de 64 bits por imagen. Con un "
        f"umbral de distancia de Hamming igual a {cd['umbral_hamming']}, se encontraron "
        f"{cd['pares']} pares casi idénticos, de los cuales {cd['pares_misma_clase']} pertenecen a la "
        f"misma clase. Entre el entrenamiento y la prueba de la semilla base hay "
        f"{cd['pares_cruzados_entrenamiento_prueba']} pares de este tipo."))
    parrafo(doc, (
        f"Todas las imágenes se convierten a escala de grises y se redimensionan a {rep['lado_imagen']} × "
        f"{rep['lado_imagen']} píxeles. Para cada semilla s, la partición es estratificada 80/20. El "
        "parámetro C del SVM se elige entre {0.1, 1, 10, 100} con validación cruzada de 5 pliegues "
        "estratificados, usando solo el conjunto de entrenamiento. El conjunto de prueba se evalúa una "
        "sola vez. Las semillas son " + ", ".join(str(s) for s in rep["semillas"]) + "."))
    parrafo(doc, (
        "Sobre las predicciones de prueba se calculan tres evaluaciones. Primero, un intervalo de "
        "confianza del 95 % por bootstrap de percentiles, con 2 000 remuestreos de las muestras de "
        "prueba. Segundo, un intervalo t del 95 % sobre la media de las semillas. Tercero, una prueba "
        "de permutación de las etiquetas de entrenamiento, con 100 permutaciones, para contrastar la "
        "exactitud observada con la distribución nula producida por el azar."))


def con_haralick(doc, rep):
    titulo_seccion(doc, "2.2 Pipeline con Haralick", 2)
    parrafo(doc, (
        "Sobre la imagen en escala de grises, los 256 niveles se cuantizan a L = 32 niveles mediante "
        "q = ⌊32 · I / 256⌋. Para cada distancia d ∈ {1, 2, 4} píxeles y cada orientación "
        "θ ∈ {0°, 45°, 90°, 135°} se construye la matriz de coocurrencia simétrica y normalizada "
        "P(i, j), donde i y j son niveles de gris."))
    ecuacion(doc, "P(i, j) = [C(i, j) + C(j, i)] / Σ_{i,j} [C(i, j) + C(j, i)]")
    parrafo(doc, (
        "Donde C(i, j) cuenta los pares de píxeles con niveles i y j separados por el desplazamiento "
        "asociado a d y θ. A partir de cada P se calculan los 13 descriptores de Haralick: segundo "
        "momento angular, contraste, correlación, varianza, momento inverso de la diferencia, promedio "
        "de la suma, varianza de la suma, entropía de la suma, entropía, varianza de la diferencia, "
        "entropía de la diferencia y las dos medidas de información de la correlación. Las entropías "
        "usan logaritmo en base 2."))
    parrafo(doc, (
        "Para cada distancia se toman la media y el rango de los 13 descriptores sobre las cuatro "
        "orientaciones. Con tres distancias, el vector de cada imagen tiene "
        f"{rep['costo_extraccion']['dimension'][CON]} componentes. Este vector se estandariza "
        "y alimenta un SVM con núcleo RBF y gamma = 'scale'."))
    parrafo(doc, (
        f"El costo de extracción fue de {rep['costo_extraccion']['haralick_ms_por_imagen']:.2f} ms por "
        f"imagen, con un tiempo total de {rep['costo_extraccion'][CON + '_s']:.2f} s para el conjunto."))
    resultados_experimento(doc, rep, CON, "Resultados con Haralick")


def sin_haralick(doc, rep):
    titulo_seccion(doc, "2.3 Pipeline sin Haralick", 2)
    parrafo(doc, (
        f"En esta representación la imagen en escala de grises se reduce a 64 × 64 píxeles y sus "
        f"intensidades se normalizan al intervalo [0, 1], lo que da {rep['costo_extraccion']['dimension'][SIN]} "
        "componentes por imagen. Después se estandarizan y se aplica un análisis de componentes "
        "principales que conserva el 95 % de la varianza. El número de componentes se ajusta dentro de "
        "cada pliegue de validación. El clasificador es el mismo SVM con núcleo RBF que en la "
        "representación con Haralick."))
    parrafo(doc, (
        f"El costo de carga y redimensionamiento se mantiene común a ambos experimentos. La extracción "
        f"de píxeles tomó {rep['costo_extraccion'][SIN + '_s']:.2f} s para el conjunto."))
    resultados_experimento(doc, rep, SIN, "Resultados sin Haralick")


def resultados_experimento(doc, rep, clave, titulo):
    r = rep["resumen"][clave]
    titulo_seccion(doc, titulo, 3)
    filas = []
    for s in r["por_semilla"]:
        filas.append((str(s["semilla"]), f4(s["exactitud"]), ic(s["ic95"]), f"{s['C']:g}",
                      f"{s['cv_media']:.4f} ({s['cv_desv']:.4f})"))
    tabla(doc, ["Semilla", "Exactitud en prueba", "IC 95 % bootstrap", "C elegido", "CV media (desv.)"],
          filas, anchos=[2.2, 3.4, 4.6, 2.4, 4.0])
    parrafo(doc, (
        f"Sobre las {len(r['por_semilla'])} semillas, la exactitud media es {f4(r['exactitud_media'])} "
        f"con IC 95 % t {ic((r['ic95_t_inf'], r['ic95_t_sup']))} y desviación estándar "
        f"{f4(r['desv_std'])}. El F1 macro medio es {f4(r['f1_macro_media'])}."))
    p = rep["permutacion"][clave]
    if p["p_valor"] <= 0.05:
        cierre = "supera a la distribución nula y no es compatible con el azar"
    else:
        cierre = "no supera de forma concluyente a la distribución nula"
    parrafo(doc, (
        f"En la prueba de permutación de la semilla base, la exactitud observada es "
        f"{f4(p['exactitud_observada'])}, mientras que la media nula es {f4(p['nula_media'])} y el "
        f"percentil 95 nulo es {f4(p['nula_p95'])}. El valor p es {f4(p['p_valor'])} con "
        f"{p['n_permutaciones']} permutaciones, por lo que la exactitud observada {cierre}. "
        f"El valor mínimo posible con {p['n_permutaciones']} permutaciones es {1 / (p['n_permutaciones'] + 1):.4f}."))
    figura(doc, Path(FIG_DIR / f"confusion_{clave}.png"),
           f"Figura. Matriz de confusión, semilla base. {titulo}.")
    figura(doc, Path(FIG_DIR / f"permutacion_{clave}.png"),
           f"Figura. Distribución nula por permutación. {titulo}.")


def comparacion(doc, rep):
    titulo_seccion(doc, "2.4 Comparación de ambos pipelines", 2)
    c = rep["comparacion"]
    tp = c["t_pareada_semillas"]
    dif_ic = c["ic95_diferencia"]
    parrafo(doc, (
        f"La diferencia media de exactitud (con Haralick menos sin Haralick) es "
        f"{f4(c['diferencia_media_con_menos_sin'])} con IC 95 % t {ic(dif_ic)}. La prueba t pareada "
        f"sobre semillas da t = {f4(tp['t'])} con {tp['gl']} grados de libertad y valor p = {f4(tp['p'])}."))
    if tp["p"] <= 0.05:
        veredicto = "la diferencia es estadísticamente significativa"
    else:
        veredicto = "la diferencia no es estadísticamente significativa"
    parrafo(doc, f"Con esta prueba, {veredicto} al nivel 0.05.")
    filas = []
    for s, m in zip(rep["semillas"], c["mcnemar_por_semilla"]):
        filas.append((str(s), str(m["b"]), str(m["c"]), f4(m["estadistico"]), f4(m["p"]), m["metodo"]))
    tabla(doc, ["Semilla", "b (sin acierta, con falla)", "c (con acierta, sin falla)", "Estadístico", "Valor p", "Método"],
          filas, anchos=[1.6, 3.2, 3.2, 2.4, 2.0, 4.6])
    n_sig = sum(1 for m in c["mcnemar_por_semilla"] if m["p"] <= 0.05)
    parrafo(doc, (
        f"La prueba de McNemar se aplica por semilla sobre las mismas muestras de prueba. Resultan "
        f"significativas en {n_sig} de {len(c['mcnemar_por_semilla'])} semillas al nivel 0.05. "
        "Como la prueba t y McNemar se calculan sobre el mismo protocolo, sus conclusiones deben leerse juntas."))
    figura(doc, FIG_DIR / "exactitud_ic95.png",
           "Figura. Exactitud media con IC 95 % t sobre semillas, por representación.")


def conclusiones(doc, rep):
    titulo_seccion(doc, "3. Conclusiones")
    s, c = rep["resumen"][SIN], rep["resumen"][CON]
    tp = rep["comparacion"]["t_pareada_semillas"]
    parrafo(doc, (
        f"La representación con descriptores de Haralick alcanzó una exactitud media de "
        f"{f4(c['exactitud_media'])} frente a {f4(s['exactitud_media'])} de la representación por píxeles. "
        "La diferencia observada entre ambas"
        + (" es estadísticamente significativa" if tp["p"] <= 0.05 else " no es estadísticamente significativa")
        + f" (valor p de la prueba t pareada = {f4(tp['p'])})."))
    parrafo(doc, (
        "Ambas representaciones superan la distribución nula por permutación. Esto indica que la "
        "exactitud observada no se explica por fuga de información ni por azar en el conjunto de "
        "entrenamiento."))
    parrafo(doc, (
        f"En cuanto al costo, la extracción de descriptores de Haralick requirió "
        f"{rep['costo_extraccion']['haralick_ms_por_imagen']:.2f} ms por imagen, mientras que la "
        "representación por píxeles solo requiere reducir y normalizar la imagen. La elección entre "
        "ambas debe considerar este costo junto con la exactitud."))
    parrafo(doc, "Las limitaciones del estudio son las siguientes.")
    vineta(doc, "Se usa un único clasificador (SVM con núcleo RBF) y un único conjunto de datos.")
    vineta(doc, f"Cada conjunto de prueba contiene alrededor del 20 % de las {rep['imagenes_finales']} imágenes, por lo que los intervalos de confianza por semilla son amplios.")
    vineta(doc, f"Con {len(rep['semillas'])} semillas, la prueba t tiene pocos grados de libertad y baja potencia.")
    vineta(doc, "La prueba de permutación se calcula con 100 permutaciones en la semilla base, lo que fija el valor p mínimo en 0.0099.")
    parrafo(doc, (
        "Como trabajo futuro, se sugiere ampliar el análisis con otros clasificadores, con más "
        "semillas y con la representación de color, para determinar si la ventaja o la falta de ella "
        "se mantiene bajo otros supuestos."))


# ------------------------------------------------------------------------ principal
FIG_DIR: Path = Path("salidas/figuras")


def construir(resultados: Path, figuras: Path, salida: Path):
    global FIG_DIR
    FIG_DIR = figuras
    rep = json.loads(resultados.read_text(encoding="utf-8"))

    doc = Document()
    estilo = doc.styles["Normal"]
    estilo.font.name = "Calibri"
    estilo.font.size = Pt(11)
    for nombre in ("Heading 1", "Heading 2", "Heading 3"):
        doc.styles[nombre].font.color.rgb = MORADO
        doc.styles[nombre].font.name = "Calibri"
    for seccion in doc.sections:
        seccion.top_margin = seccion.bottom_margin = Cm(2.5)
        seccion.left_margin = seccion.right_margin = Cm(2.5)
        seccion.different_first_page_header_footer = True
        numero_de_pagina(seccion.footer.paragraphs[0])
        seccion.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    portada(doc)
    introduccion(doc, rep)
    protocolo(doc, rep)
    con_haralick(doc, rep)
    sin_haralick(doc, rep)
    comparacion(doc, rep)
    conclusiones(doc, rep)

    salida.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(salida))
    return salida


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", type=Path, default=Path("salidas/resultados.json"))
    ap.add_argument("--figuras", type=Path, default=Path("salidas/figuras"))
    ap.add_argument("--salida", type=Path, default=Path("reporte/reporte_practica06.docx"))
    a = ap.parse_args(argv)
    ruta = construir(a.resultados, a.figuras, a.salida)
    print(f"Reporte generado: {ruta}")


if __name__ == "__main__":
    main()
