# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta
from pathlib import Path
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
TIPO_GENERACION_SOLAR   = "3"
CENTRAL_COES            = "1"
PARAMETRO_POT_ACTIVA    = "1"   # "Potencia Activa (MW)"
EMPRESAS_SOLARES        = ["12584","13966","11103"] 

'''
# Empresas solares en COES
## Rep Arcus 16°34'31"S 71°49'05"W 
## TACNA 17°59'37"S 70°20'07"W
## Colca 11°02'17"S 77°05'47"W lat -11.038056 long -77.096389

Empresas seleccionadas actualmente

COLCA SOLAR S.A.C. → value="12584"

REPARTICIÓN ARCUS S.A.C. → value="13966"

TACNA SOLAR SAC. → value="11103"
'''

TIMEOUT_GLOBAL = 180  # segundos

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# --------------------------- Driver ---------------------------

def build_driver(download_dir="data"):
    download_path = Path(download_dir).resolve()
    download_path.mkdir(parents=True, exist_ok=True)

    options = Options()
    options.add_argument("--headless=new")  # comenta para ver la UI
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

    # Permitir descargas en headless
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

# --------------------------- Helpers UI ---------------------------

def safe_click(driver, locator):
    try:
        el = WebDriverWait(driver, 20).until(EC.presence_of_element_located(locator))
        driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(locator)).click()
    except Exception:
        try:
            driver.execute_script("arguments[0].click();", el)
        except Exception as e:
            raise RuntimeError(f"No se pudo hacer click en {locator}: {e}")

def set_date_input_soft(driver, element_id, yyyymmdd):
    """Setea fecha con JS y dispara eventos."""
    el = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, element_id)))
    driver.execute_script("""
        arguments[0].value = arguments[1];
        arguments[0].dispatchEvent(new Event('input', {bubbles:true}));
        arguments[0].dispatchEvent(new Event('change', {bubbles:true}));
        arguments[0].dispatchEvent(new Event('blur',   {bubbles:true}));
    """, el, yyyymmdd)
    return driver.execute_script("return arguments[0].value;", el) == yyyymmdd

def set_date_input_hard(driver, element_id, yyyymmdd):
    """Reintenta 3 veces: JS + tipeo real si hace falta."""
    for _ in range(3):
        if set_date_input_soft(driver, element_id, yyyymmdd):
            return True
        # fallback tipeando
        el = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, element_id)))
        try:
            el.click()
            el.send_keys(Keys.CONTROL, "a")
            el.send_keys(Keys.DELETE)
            el.send_keys(yyyymmdd)
            time.sleep(0.1)
        except Exception:
            pass
        if driver.execute_script("return arguments[0].value;", el) == yyyymmdd:
            return True
        time.sleep(0.1)
    return False

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
    return res

def ensure_modal_clickable_surface(driver):
    # Quita backdrops que bloqueen
    driver.execute_script("""
        document.querySelectorAll('.modal-backdrop').forEach(b=>{
            b.style.pointerEvents='none';
            b.style.opacity='0.001';
        });
        const body = document.querySelector('body');
        if (body) { body.classList.remove('modal-open'); body.style.paddingRight='0px'; }
    """)
    # Habilita boton aceptar por si estuviera deshabilitado
    driver.execute_script("""
        var $ = window.jQuery;
        var btn = document.getElementById('btnProcesarFile');
        if (!btn) return;
        btn.removeAttribute('disabled');
        btn.setAttribute('aria-disabled','false');
        btn.style.pointerEvents='auto';
        btn.style.opacity='1';
        if ($ && $(btn).prop) { $(btn).prop('disabled', false); }
    """)

# --------------------------- Filtros principales ---------------------------

def configure_filters(driver, fecha_ini, fecha_fin):
    log("Aplicando filtros...")
    set_date_input_hard(driver, "txtFechaInicial", fecha_ini)
    set_date_input_hard(driver, "txtFechaFinal",   fecha_fin)
    ms_set_selects(driver, "cbTipoEmpresa", [TIPO_EMPRESA_GENERACION])
    ms_set_selects(driver, "cbEmpresas",    EMPRESAS_SOLARES)
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

# --------------------------- Modal Exportar ---------------------------

def open_export_modal(driver):
    log("Abriendo modal de exportacion...")
    safe_click(driver, (By.ID, "btnExportar"))
    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".modal-dialog")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "txtExportarDesde")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "txtExportarHasta")))
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "btnProcesarFile")))
    driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")

def set_parametro_exportar(driver, valor=PARAMETRO_POT_ACTIVA):
    js = """
        var $s = window.jQuery && jQuery('#cbParametroExportar');
        if ($s && $s.length) {
            if (!$s.data('multipleSelect')) { $s.multipleSelect({ filter: true, width: '250px' }); }
            $s.multipleSelect('uncheckAll');
            $s.multipleSelect('setSelects', [arguments[0]]);
            $s.trigger('change');
            return $s.multipleSelect('getSelects');
        }
        var el = document.getElementById('cbParametroExportar');
        if (el) { el.value = arguments[0]; el.dispatchEvent(new Event('change')); return [el.value]; }
        return [];
    """
    sel = driver.execute_script(js, valor)
    log(f"Parametro export seleccionado: {sel}")
    return (sel and str(sel[0]) == str(valor))

def ensure_csv_format(driver):
    # marca CSV si no está marcado
    is_csv = driver.execute_script("""
        var r=document.getElementById('FormatoCSV');
        return !!(r && r.checked);
    """)
    if not is_csv:
        try:
            safe_click(driver, (By.ID, "FormatoCSV"))
        except Exception:
            driver.execute_script("""
                var r=document.getElementById('FormatoCSV');
                if(r){ r.checked=true; r.dispatchEvent(new Event('change',{bubbles:true})); }
            """)
    time.sleep(0.1)
    return driver.execute_script("var r=document.getElementById('FormatoCSV'); return !!(r && r.checked);")

def set_and_verify_export_fields(driver, export_ini, export_fin, max_tries=3):
    """Recoloca y verifica Desde/Hasta + Parámetro + CSV. Vuelve a intentar si la página 'borra' Hasta."""
    ok_all = False
    for t in range(1, max_tries+1):
        ok_ini = set_date_input_hard(driver, "txtExportarDesde", export_ini)
        ok_fin = set_date_input_hard(driver, "txtExportarHasta", export_fin)
        ok_par = set_parametro_exportar(driver, PARAMETRO_POT_ACTIVA)
        ok_csv = ensure_csv_format(driver)

        # lectura efectiva
        rd_ini, rd_fin, rd_par, rd_csv = driver.execute_script("""
            var a=document.getElementById('txtExportarDesde')?.value||'';
            var b=document.getElementById('txtExportarHasta')?.value||'';
            var par = (function(){
                var $s = window.jQuery && jQuery('#cbParametroExportar');
                if ($s && $s.length && $s.data('multipleSelect')) {
                    var v=$s.multipleSelect('getSelects'); return v && v[0] ? v[0] : '';
                }
                var el=document.getElementById('cbParametroExportar');
                return el ? el.value : '';
            })();
            var csv = !!(document.getElementById('FormatoCSV') && document.getElementById('FormatoCSV').checked);
            return [a,b,par,csv];
        """)

        log(f"Verif modal (try {t}) -> desde={rd_ini} hasta={rd_fin} par={rd_par} csv={rd_csv}")

        ok_all = (rd_ini == export_ini and rd_fin == export_fin and str(rd_par) == str(PARAMETRO_POT_ACTIVA) and rd_csv)
        if ok_all:
            return True

        # si la pagina borra 'Hasta', vuelve a escribir y a disparar eventos
        time.sleep(0.2)
    return ok_all

def click_export_aceptar(driver):
    driver.execute_script("document.querySelectorAll('#hfParametro').forEach(hf => hf.value='1');")
    btn = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "btnProcesarFile")))
    ensure_modal_clickable_surface(driver)
    driver.execute_script("document.querySelectorAll('.ms-drop').forEach(e=>e.style.display='none');")
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    clicked = driver.execute_script("""
        var $ = window.jQuery, b = document.getElementById('btnProcesarFile');
        if (!b) return 'no-btn';
        try {
            if ($ && $(b).trigger) { $(b).trigger('click'); return 'triggered'; }
            b.click(); return 'clicked';
        } catch(e) {
            return 'error:' + (e && e.message ? e.message : e);
        }
    """)
    log(f"Click Aceptar -> {clicked}")

def wait_for_download(download_dir: Path, timeout=TIMEOUT_GLOBAL):
    log(f"Esperando descarga en: {download_dir}")
    end = time.time() + timeout
    before = {p.name for p in download_dir.glob("*")}
    while time.time() < end:
        files = list(download_dir.glob("*"))
        if any(p.suffix.lower() == ".crdownload" for p in files):
            time.sleep(0.5); continue
        new_files = [p for p in files if p.name not in before and p.is_file()]
        if not new_files:
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

# --------------------------- Flujo Export ---------------------------

def intentar_exportar(driver, export_ini, export_fin, formato, download_dir):
    # Abre modal limpio
    open_export_modal(driver)
    # Recoloca y verifica todo (desde/hasta/parametro/csv)
    if not set_and_verify_export_fields(driver, export_ini, export_fin, max_tries=3):
        raise RuntimeError("No se pudieron fijar correctamente los campos del modal (Desde/Hasta/Parámetro/CSV).")

    # Click aceptar y esperar
    click_export_aceptar(driver)
    return wait_for_download(download_dir)

# --------------------------- Run mes a mes ---------------------------

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
        for intento in range(1, 4):
            try:
                fpath = intentar_exportar(driver, export_ini, export_fin, formato, dl)
                break
            except Exception as e:
                last_exc = e
                log(f"warn: export intento {intento} fallo: {e}. reintentando...")
                # Cierra modal zombie y limpia backdrops antes del siguiente intento
                driver.execute_script("""
                    try { document.querySelectorAll('.modal.show .close, .modal .btn-close').forEach(b=>b.click()); } catch(e){}
                    document.querySelectorAll('.modal').forEach(m=>{ m.classList.remove('show'); m.style.display='none'; });
                    document.querySelectorAll('.modal-backdrop').forEach(b=>{ b.remove(); });
                    document.body.classList.remove('modal-open');
                    document.body.style.paddingRight='0px';
                """)
                time.sleep(1.0)
        else:
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

# --------------------------- Rango mensual ---------------------------

def generar_rangos_mensuales(fecha_ini: str, fecha_fin: str):
    """Devuelve lista de (YYYY-MM-DD inicio, YYYY-MM-DD fin) por mes."""
    rangos = []
    inicio = datetime.strptime(fecha_ini, "%Y-%m-%d")
    fin    = datetime.strptime(fecha_fin,   "%Y-%m-%d")
    while inicio <= fin:
        ultimo_dia  = (inicio.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        fecha_final = min(ultimo_dia, fin)
        rangos.append((inicio.strftime("%Y-%m-%d"), fecha_final.strftime("%Y-%m-%d")))
        inicio = fecha_final + timedelta(days=1)
    return rangos

def run_coes_rango(fecha_ini, fecha_fin,
                   out_temp_dir="data/temp",
                   out_final_dir="data",
                   out_final="COES.csv"):
    Path(out_temp_dir).mkdir(parents=True, exist_ok=True)
    Path(out_final_dir).mkdir(parents=True, exist_ok=True)

    rangos = generar_rangos_mensuales(fecha_ini, fecha_fin)
    log(f"Se generaran {len(rangos)} descargas mensuales")

    csv_paths = []
    for i, (ini, fin) in enumerate(rangos, 1):
        log(f"[{i}/{len(rangos)}] {ini} -> {fin}")
        nombre = f"COES_{ini[:7]}.csv"
        path = run(
            fecha_ini=ini, fecha_fin=fin,
            export_ini=ini, export_fin=fin,
            formato="csv",
            out_dir=out_temp_dir,
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

    final_path = Path(out_final_dir) / out_final
    df_final.to_csv(final_path, index=False, encoding="utf-8")
    log(f"Archivo final: {final_path}")
    return str(final_path)

# --------------------------- Main local ---------------------------

if __name__ == "__main__":
    ruta = run_coes_rango(
        fecha_ini="2023-01-01",
        fecha_fin="2025-07-31",
        out_temp_dir="data/temp",
        out_final_dir="data",
        out_final="COES.csv"
    )
    if ruta:
        log(f"Fin: {ruta}")
