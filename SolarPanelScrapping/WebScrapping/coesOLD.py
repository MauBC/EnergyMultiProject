# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback
import pandas as pd  # <-- para merge

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://www.coes.org.pe/Portal/mediciones/medidoresgeneracion"

TIPO_EMPRESA_GENERACION = "3"
TIPO_GENERACION_SOLAR = "3"
CENTRAL_COES = "1"
PARAMETRO_POT_ACTIVA = "1"
EMPRESAS_SOLARES = ["12584"]  # Colca, Reparticion Arcus, Tacna
'''
# Empresas solares en COES
## Rep Arcus 16°34'31"S 71°49'05"W
## TACNA 17°59'37"S 70°20'07"W
## Colca 11°02'17"S 77°05'47"W

Empresas seleccionadas actualmente

COLCA SOLAR S.A.C. → value="12584"

REPARTICIÓN ARCUS S.A.C. → value="13966"

TACNA SOLAR SAC. → value="11103"
'''

TIMEOUT_GLOBAL = 180  # segundos

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def build_driver(download_dir="data"):
    download_path = Path(download_dir).resolve()
    download_path.mkdir(parents=True, exist_ok=True)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )
    prefs = {
        "download.default_directory": str(download_path),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_argument("--log-level=3")
    options.add_argument("--remote-debugging-port=0")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver, download_path

def wait_js_ready(driver):
    WebDriverWait(driver, 30).until(lambda d: d.execute_script("return document.readyState") == "complete")

def safe_click(driver, locator):
    try:
        el = WebDriverWait(driver, 20).until(EC.presence_of_element_located(locator))
        driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")
        dismiss_overlays(driver)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(locator)).click()
    except Exception:
        dismiss_overlays(driver)
        try:
            driver.execute_script("arguments[0].click();", el)
        except Exception as e:
            raise RuntimeError(f"No se pudo hacer click en {locator}: {e}")

def dismiss_overlays(driver):
    try:
        close_btn = driver.find_elements(By.CSS_SELECTOR, ".coes-modal-close--button.b-close")
        for btn in close_btn:
            if btn.is_displayed():
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.3)
    except:
        pass
    driver.execute_script("""
        document.querySelectorAll('div.b-modal').forEach(m => { 
            m.style.display='none'; 
            m.style.opacity=0; 
            m.style.pointerEvents='none'; 
        });
    """)
    time.sleep(0.1)

def set_date_input(driver, element_id, yyyymmdd):
    try:
        el = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, element_id)))
        driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change'));", el, yyyymmdd)
    except:
        log(f"⚠ No se pudo establecer la fecha en {element_id}")

def ms_set_selects(driver, select_id, values):
    js = """
        var id = arguments[0], vals = arguments[1];
        if (!window.jQuery) return "no-jquery";
        var $el = jQuery("#"+id);
        if (!$el.length) return "no-el";
        if (!$el.data("multipleSelect")) { $el.multipleSelect({ filter: true, width: '250px' }); }
        $el.multipleSelect('uncheckAll');
        $el.multipleSelect('setSelects', vals);
        $el.trigger('change');
        return "ok";
    """
    res = driver.execute_script(js, select_id, values if isinstance(values, list) else [values])
    if res != "ok":
        log(f"⚠ Problema al aplicar select {select_id}: {res}")

def configure_filters(driver, fecha_ini, fecha_fin):
    log("Aplicando filtros...")
    set_date_input(driver, "txtFechaInicial", fecha_ini)
    set_date_input(driver, "txtFechaFinal", fecha_fin)
    ms_set_selects(driver, "cbTipoEmpresa", [TIPO_EMPRESA_GENERACION])
    ms_set_selects(driver, "cbEmpresas", EMPRESAS_SOLARES)
    ms_set_selects(driver, "cbTipoGeneracion", [TIPO_GENERACION_SOLAR])
    driver.execute_script("var s=document.getElementById('cbParametro'); if(s){s.value=arguments[0]; s.dispatchEvent(new Event('change'));}", PARAMETRO_POT_ACTIVA)
    driver.execute_script("var s=document.getElementById('cbCentral'); if(s){s.value=arguments[0]; s.dispatchEvent(new Event('change'));}", CENTRAL_COES)
    time.sleep(0.2)

def click_consultar(driver):
    log("Consultando datos en COES...")
    safe_click(driver, (By.ID, "btnBuscar"))
    WebDriverWait(driver, 60).until(
        lambda d: d.execute_script("var t=document.querySelectorAll('table tbody tr'); return t && t.length>0;")
    )

def open_export_modal(driver):
    log("Abriendo modal de exportación...")
    safe_click(driver, (By.ID, "btnExportar"))
    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.ID, "popupDisclaimerLabel")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "txtExportarDesde")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".modal-dialog #btnProcesarFile")))
    dismiss_overlays(driver)
    driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")

def configure_export_modal(driver, export_ini, export_fin, formato="csv"):
    log("Configurando opciones de exportación...")
    set_date_input(driver, "txtExportarDesde", export_ini)
    set_date_input(driver, "txtExportarHasta", export_fin)
    ms_set_selects(driver, "cbParametroExportar", [PARAMETRO_POT_ACTIVA])
    if formato.lower() == "csv":
        safe_click(driver, (By.ID, "FormatoCSV"))
    elif formato.lower() == "vertical":
        safe_click(driver, (By.ID, "FormatoVertical"))
    else:
        safe_click(driver, (By.ID, "FormatoHorizontal"))

def click_export_aceptar(driver):
    driver.execute_script("document.querySelectorAll('#hfParametro').forEach(hf => hf.value='1');")
    driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")
    dismiss_overlays(driver)
    btn = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".modal-dialog #btnProcesarFile"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    try:
        btn.click()
    except:
        driver.execute_script("arguments[0].click();", btn)

def wait_for_download(download_dir: Path, timeout=TIMEOUT_GLOBAL):
    log("Esperando descarga del archivo...")
    end = time.time() + timeout
    before = set(download_dir.glob("*"))
    while time.time() < end:
        current = set(download_dir.glob("*"))
        newf = [p for p in (current - before) if p.suffix != ".crdownload"]
        if newf:
            file = max(newf, key=lambda p: p.stat().st_mtime)
            if file.stat().st_size > 0:
                time.sleep(1)
                return file
        time.sleep(0.5)
    raise TimeoutError("⛔ No se detectó el archivo descargado")

def clean_incomplete_files(download_dir: Path):
    for f in download_dir.glob("*.crdownload"):
        f.unlink(missing_ok=True)

def run(fecha_ini="2025-07-01", fecha_fin="2025-07-31",
        export_ini=None, export_fin=None, formato="csv", out_dir=".", out_name="COES.csv"):
    driver, dl = build_driver(out_dir)
    try:
        clean_incomplete_files(dl)
        log("Abriendo página COES...")
        driver.get(URL)
        wait_js_ready(driver)

        configure_filters(driver, fecha_ini, fecha_fin)
        click_consultar(driver)

        open_export_modal(driver)
        export_ini = export_ini or fecha_ini
        export_fin = export_fin or fecha_fin
        configure_export_modal(driver, export_ini, export_fin, formato=formato)

        click_export_aceptar(driver)
        fpath = wait_for_download(dl)
        dest = Path(out_dir) / out_name
        if dest.exists():
            dest.unlink()
        fpath.rename(dest)
        log(f"✅ Archivo COES guardado en: {dest}")
        return str(dest)
    except Exception as e:
        log(f"❌ Error: {e}")
        traceback.print_exc()
        return None
    finally:
        driver.quit()

# ========= NUEVO: helpers de rango y merge =========
def generar_rangos_mensuales(fecha_ini: str, fecha_fin: str):
    """Devuelve lista de (YYYY-MM-DD inicio, YYYY-MM-DD fin) por mes."""
    rangos = []
    inicio = datetime.strptime(fecha_ini, "%Y-%m-%d")
    fin    = datetime.strptime(fecha_fin,   "%Y-%m-%d")
    while inicio <= fin:
        ultimo_dia = (inicio.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        fecha_final = min(ultimo_dia, fin)
        rangos.append((inicio.strftime("%Y-%m-%d"), fecha_final.strftime("%Y-%m-%d")))
        inicio = fecha_final + timedelta(days=1)
    return rangos

def run_coes_rango(fecha_ini, fecha_fin,
                   out_temp_dir="data/temp",
                   out_final_dir="data",
                   out_final="COES.csv"):
    # asegurar carpetas
    Path(out_temp_dir).mkdir(parents=True, exist_ok=True)
    Path(out_final_dir).mkdir(parents=True, exist_ok=True)

    rangos = generar_rangos_mensuales(fecha_ini, fecha_fin)
    log(f"Se generaran {len(rangos)} descargas")

    csv_paths = []
    for i, (ini, fin) in enumerate(rangos, 1):
        log(f"[{i}/{len(rangos)}] {ini} -> {fin}")
        nombre = f"COES_{ini[:7]}.csv"  # ej: COES_2025-07.csv
        path = run(
            fecha_ini=ini, fecha_fin=fin,
            export_ini=ini, export_fin=fin,
            formato="csv",
            out_dir=out_temp_dir,          # <-- descarga y deja los CSV chicos en data/temp
            out_name=nombre
        )
        if path:
            csv_paths.append(path)
        time.sleep(1.0)

    if not csv_paths:
        log("No se genero ningun archivo")
        return None

    log("Uniendo CSVs...")
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            # opcional: agrega la columna fuente para rastrear de donde vino
            df["fuente_csv"] = Path(p).name
            dfs.append(df)
        except Exception as e:
            log(f"⚠ No pude leer {p}: {e}")

    if not dfs:
        log("No hay CSVs validos para unir")
        return None

    df_final = pd.concat(dfs, ignore_index=True)

    # opcional: quita duplicados si los hubiera
    try:
        df_final = df_final.drop_duplicates()
    except Exception:
        pass

    final_path = Path(out_final_dir) / out_final  # <-- deja el CSV final en data/COES.csv
    df_final.to_csv(final_path, index=False, encoding="utf-8")
    log(f"Archivo final: {final_path}")
    return str(final_path)


if __name__ == "__main__":
    ruta = run_coes_rango(
        fecha_ini="2023-01-01",
        fecha_fin="2025-03-31",
        out_temp_dir="data/temp",   # CSV mensuales aqui
        out_final_dir="data",       # CSV unido aqui
        out_final="COES.csv"
    )
    if ruta:
        log(f"Fin: {ruta}")
