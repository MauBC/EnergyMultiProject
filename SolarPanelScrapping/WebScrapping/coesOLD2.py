# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://www.coes.org.pe/Portal/mediciones/medidoresgeneracion"

TIPO_EMPRESA_GENERACION = "3"
TIPO_GENERACION_SOLAR = "3"
CENTRAL_COES = "1"
PARAMETRO_POT_ACTIVA = "1"
EMPRESAS_SOLARES = ["12584"]  # Colca

TIMEOUT_GLOBAL = 180  # segundos

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def build_driver(download_dir="data"):
    download_path = Path(download_dir).resolve()
    download_path.mkdir(parents=True, exist_ok=True)

    options = Options()
    options.add_argument("--headless=new")  # comenta esta linea si quieres ver la UI
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

    # permitir descargas en headless (CDP)
    try:
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": str(download_path)
        })
    except Exception:
        try:
            driver.execute_cdp_cmd("Browser.setDownloadBehavior", {
                "behavior": "allow",
                "downloadPath": str(download_path)
            })
        except Exception:
            pass

    return driver, download_path

def wait_js_ready(driver):
    WebDriverWait(driver, 30).until(lambda d: d.execute_script("return document.readyState") == "complete")

def safe_click(driver, locator):
    try:
        el = WebDriverWait(driver, 20).until(EC.presence_of_element_located(locator))
        # oculta listas desplegables flotantes
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
    # cierra botones de cierre si existen
    try:
        close_btn = driver.find_elements(By.CSS_SELECTOR, ".coes-modal-close--button.b-close")
        for btn in close_btn:
            if btn.is_displayed():
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.2)
    except:
        pass
    # oculta overlays oscuros sin tocar el modal activo
    driver.execute_script("""
        document.querySelectorAll('div.b-modal').forEach(m => {
            // evita tocar modales bootstrap activos
            if (!m.closest('.modal-dialog')) {
                m.style.display='none'; 
                m.style.opacity=0; 
                m.style.pointerEvents='none';
            }
        });
    """)
    time.sleep(0.1)

def set_date_input(driver, element_id, yyyymmdd):
    try:
        el = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, element_id)))
        driver.execute_script(
            "arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change'));",
            el, yyyymmdd
        )
    except:
        log(f"warn: no se pudo establecer la fecha en {element_id}")

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
        return $el.multipleSelect('getSelects');
    """
    res = driver.execute_script(js, select_id, values if isinstance(values, list) else [values])
    log(f"{select_id} -> {res}")

def configure_filters(driver, fecha_ini, fecha_fin):
    log("Aplicando filtros...")
    set_date_input(driver, "txtFechaInicial", fecha_ini)
    set_date_input(driver, "txtFechaFinal", fecha_fin)
    ms_set_selects(driver, "cbTipoEmpresa", [TIPO_EMPRESA_GENERACION])
    ms_set_selects(driver, "cbEmpresas", EMPRESAS_SOLARES)
    ms_set_selects(driver, "cbTipoGeneracion", [TIPO_GENERACION_SOLAR])
    driver.execute_script(
        "var s=document.getElementById('cbParametro'); if(s){s.value=arguments[0]; s.dispatchEvent(new Event('change'));}",
        PARAMETRO_POT_ACTIVA
    )
    driver.execute_script(
        "var s=document.getElementById('cbCentral'); if(s){s.value=arguments[0]; s.dispatchEvent(new Event('change'));}",
        CENTRAL_COES
    )
    time.sleep(0.2)
    # verificacion de filtros
    vals = driver.execute_script("""
        function selVals(id){ 
            var $e = window.jQuery && jQuery('#'+id);
            if ($e && $e.length && $e.data('multipleSelect')) return $e.multipleSelect('getSelects');
            var el = document.getElementById(id);
            if (!el) return [];
            return Array.from(el.options).filter(o=>o.selected).map(o=>o.value);
        }
        return {
            tipoEmpresa: selVals('cbTipoEmpresa'),
            empresas: selVals('cbEmpresas'),
            tipoGen: selVals('cbTipoGeneracion')
        };
    """)
    log(f"Filtros -> tipoEmpresa={vals.get('tipoEmpresa')} empresas={vals.get('empresas')} tipoGen={vals.get('tipoGen')}")

def click_consultar(driver):
    log("Consultando datos en COES...")
    safe_click(driver, (By.ID, "btnBuscar"))
    WebDriverWait(driver, 60).until(
        lambda d: d.execute_script("var t=document.querySelectorAll('table tbody tr'); return t && t.length>0;")
    )

def open_export_modal(driver):
    log("Abriendo modal de exportacion...")
    safe_click(driver, (By.ID, "btnExportar"))
    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".modal-dialog")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "txtExportarDesde")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "txtExportarHasta")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".modal-dialog #btnProcesarFile")))
    # oculta solo dropdowns flotantes
    driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")

def configure_export_modal(driver, export_ini, export_fin, formato="csv"):
    log("Configurando opciones de exportacion...")
    set_date_input(driver, "txtExportarDesde", export_ini)
    set_date_input(driver, "txtExportarHasta", export_fin)

    # multipleSelect del modal: setear Potencia Activa (1)
    js = """
        var $s = window.jQuery && jQuery('#cbParametroExportar');
        if ($s && $s.length) {
            if (!$s.data('multipleSelect')) { $s.multipleSelect({ filter: true, width: '250px' }); }
            $s.multipleSelect('uncheckAll');
            $s.multipleSelect('setSelects', [arguments[0]]);
            $s.trigger('change');
            return $s.val() || [];
        }
        return [];
    """
    seleccion = driver.execute_script(js, PARAMETRO_POT_ACTIVA)
    log(f"Parametro export seleccionado: {seleccion}")

    # formato
    if formato.lower() == "csv":
        safe_click(driver, (By.ID, "FormatoCSV"))
    elif formato.lower() == "vertical":
        safe_click(driver, (By.ID, "FormatoVertical"))
    else:
        safe_click(driver, (By.ID, "FormatoHorizontal"))

    time.sleep(0.2)

def click_export_aceptar(driver):
    # por si la pagina lo usa
    driver.execute_script("document.querySelectorAll('#hfParametro').forEach(hf => hf.value='1');")
    # oculta dropdowns flotantes pero no el modal
    driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")

    btn = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, ".modal-dialog #btnProcesarFile"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    try:
        btn.click()
    except Exception:
        try:
            btn.send_keys(Keys.ENTER)
        except Exception:
            driver.execute_script("arguments[0].click();", btn)

def wait_for_download(download_dir: Path, timeout=TIMEOUT_GLOBAL):
    log(f"Esperando descarga en: {download_dir}")
    end = time.time() + timeout
    before = {p.name for p in download_dir.glob("*")}
    while time.time() < end:
        files = list(download_dir.glob("*"))
        # si hay .crdownload, aun esta bajando
        if any(p.suffix.lower() == ".crdownload" for p in files):
            time.sleep(0.5)
            continue

        # nuevos archivos
        new_files = [p for p in files if p.name not in before and p.is_file()]
        if not new_files:
            # toma muy recientes por si el nombre coincide
            new_files = [p for p in files if (time.time() - p.stat().st_mtime) < 5]

        candidates = [p for p in new_files if p.suffix.lower() in (".csv", ".xlsx", ".zip")]
        if candidates:
            f = max(candidates, key=lambda p: p.stat().st_mtime)
            if f.stat().st_size > 0:
                time.sleep(1)
                log(f"Descarga detectada: {f.name} ({f.stat().st_size} bytes)")
                return f

        time.sleep(0.5)

    existentes = ", ".join(p.name for p in download_dir.glob("*"))
    raise TimeoutError(f"No se detecto archivo descargado. Contenido: {existentes}")

def clean_incomplete_files(download_dir: Path):
    for f in download_dir.glob("*.crdownload"):
        f.unlink(missing_ok=True)

def intentar_exportar(driver, export_ini, export_fin, formato, download_dir):
    open_export_modal(driver)
    configure_export_modal(driver, export_ini, export_fin, formato=formato)

    # sanity check: verifica fechas seteadas
    ok_ini, ok_fin = driver.execute_script("""
        var a=document.getElementById('txtExportarDesde')?.value||'';
        var b=document.getElementById('txtExportarHasta')?.value||'';
        return [a,b];
    """)
    if ok_ini != export_ini or ok_fin != export_fin:
        set_date_input(driver, "txtExportarDesde", export_ini)
        set_date_input(driver, "txtExportarHasta", export_fin)
        time.sleep(0.2)

    click_export_aceptar(driver)
    return wait_for_download(download_dir)

def run(fecha_ini="2025-07-01", fecha_fin="2025-07-31",
        export_ini=None, export_fin=None, formato="csv", out_dir=".", out_name="COES.csv"):
    driver, dl = build_driver(out_dir)
    try:
        clean_incomplete_files(dl)
        log("Abriendo pagina COES...")
        driver.get(URL)
        wait_js_ready(driver)

        configure_filters(driver, fecha_ini, fecha_fin)
        click_consultar(driver)

        export_ini = export_ini or fecha_ini
        export_fin = export_fin or fecha_fin

        last_exc = None
        for intento in range(1, 4):  # hasta 3 intentos
            try:
                fpath = intentar_exportar(driver, export_ini, export_fin, formato, dl)
                break
            except Exception as e:
                last_exc = e
                log(f"warn: export intento {intento} fallo: {e}. reintentando...")
                time.sleep(1.0)
        else:
            # todos los intentos fallaron
            raise last_exc

        dest = Path(out_dir) / out_name
        if dest.exists():
            dest.unlink()
        fpath.rename(dest)
        log(f"Archivo COES guardado en: {dest}")
        return str(dest)
    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()
        return None
    finally:
        driver.quit()

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
    log(f"Se generaran {len(rangos)} descargas mensuales")

    csv_paths = []
    for i, (ini, fin) in enumerate(rangos, 1):
        log(f"[{i}/{len(rangos)}] {ini} -> {fin}")
        nombre = f"COES_{ini[:7]}.csv"  # ej: COES_2025-07.csv
        path = run(
            fecha_ini=ini, fecha_fin=fin,
            export_ini=ini, export_fin=fin,
            formato="csv",
            out_dir=out_temp_dir,  # deja los CSV chicos en data/temp
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
            df["fuente_csv"] = Path(p).name
            dfs.append(df)
        except Exception as e:
            log(f"warn: no pude leer {p}: {e}")

    if not dfs:
        log("No hay CSVs validos para unir")
        return None

    df_final = pd.concat(dfs, ignore_index=True)
    try:
        df_final = df_final.drop_duplicates()
    except Exception:
        pass

    final_path = Path(out_final_dir) / out_final  # CSV final en data/COES.csv
    df_final.to_csv(final_path, index=False, encoding="utf-8")
    log(f"Archivo final: {final_path}")
    return str(final_path)

if __name__ == "__main__":
    ruta = run_coes_rango(
        fecha_ini="2023-01-01",
        fecha_fin="2025-07-31",
        out_temp_dir="data/temp",   # CSV mensuales
        out_final_dir="data",       # CSV unido
        out_final="COES.csv"
    )
    if ruta:
        log(f"Fin: {ruta}")
