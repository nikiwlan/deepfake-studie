import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import uuid

# Google Sheets Library
from streamlit_gsheets import GSheetsConnection

# ==========================================================
# ⚙️ KONFIGURATION
# ==========================================================
ANZAHL_VIDEOS = 5 
DATA_FOLDER = "studien_daten"
VIDEO_ROOT = "videos"  # Der Hauptordner

# MAPPING: Ordnername auf Festplatte -> Gruppenname im Code
# (Basiert auf deinen Screenshots)
FOLDER_TO_GROUP_MAPPING = {
    "normalisiert_720p_40fps": "720p_mit_ton",
    "normalisiert_1080p_60fps": "1080p_mit_ton",
    "normalisiert_ohne_Ton": "720p_ohne_ton"
}

# MAPPING: ID -> Gruppenname (für die Zuteilung)
GRUPPEN_MAPPING = {
    0: "720p_mit_ton",  # ID 3, 6, 9...
    1: "1080p_mit_ton", # ID 1, 4, 7...
    2: "720p_ohne_ton"  # ID 2, 5, 8...
}

# ==========================================================
# 📂 ORDNER MANAGEMENT & SCAN-LOGIK
# ==========================================================
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def scan_video_folders():
    """
    Ersetzt die metadata.csv.
    Scannt die Ordner 'videos/GRUPPE/Real' und 'videos/GRUPPE/Fake'.
    """
    video_list = []
    
    if not os.path.exists(VIDEO_ROOT):
        st.error(f"Kritischer Fehler: Der Ordner '{VIDEO_ROOT}' wurde nicht gefunden!")
        return pd.DataFrame()

    # Wir gehen durch die definierten Gruppen-Ordner
    for folder_name, group_label in FOLDER_TO_GROUP_MAPPING.items():
        group_path = os.path.join(VIDEO_ROOT, folder_name)
        
        # Prüfen, ob der Gruppenordner existiert
        if os.path.exists(group_path):
            # Wir suchen nach "Real" und "Fake" Unterordnern (Großschreibung beachten!)
            for label in ["Real", "Fake"]:
                label_path = os.path.join(group_path, label)
                
                if os.path.exists(label_path):
                    # Alle Dateien im Ordner auflisten
                    for file in os.listdir(label_path):
                        # Nur Videodateien beachten
                        if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                            video_list.append({
                                "filename": file,
                                # Wir speichern den Pfad relativ zum videos-Ordner
                                "full_path": os.path.join(folder_name, label, file),
                                "label": label.lower(), # 'real' oder 'fake'
                                "gruppe": group_label
                            })
    
    return pd.DataFrame(video_list)

# ==========================================================
# 🔢 NEUE ID LOGIK (Cloud-Safe)
# ==========================================================
def get_next_id_from_cloud(conn):
    """Ermittelt die nächste ID basierend auf der Anzahl der Teilnehmer in der DB"""
    try:
        # ttl=0 ist wichtig, damit er wirklich die aktuellen Daten holt!
        df = conn.read(worksheet="Tabellenblatt1", ttl=0)
        
        # Wenn Tabelle leer ist oder Spalte fehlt -> ID 1
        if df.empty or "Testperson" not in df.columns:
            return 1
        
        # Wir zählen, wie viele eindeutige Testpersonen es schon gibt
        unique_ids = df["Testperson"].nunique()
        return unique_ids + 1
    except:
        # Falls gar keine Verbindung klappt, Fallback auf 1
        return 1

# ==========================================================
# Helpers: Query Params
# ==========================================================
def _get_qp() -> dict:
    try: return dict(st.query_params)
    except: return {k: v[0] for k, v in st.experimental_get_query_params().items()}

def _set_qp(**kwargs):
    clean = {k: str(v) for k, v in kwargs.items() if v is not None}
    try:
        st.query_params.clear()
        st.query_params.update(clean)
    except: st.experimental_set_query_params(**clean)

def _clear_qp():
    try: st.query_params.clear()
    except: st.experimental_set_query_params()

def _parse_int(val, default):
    try: return int(val)
    except: return default

# ==========================================================
# ✅ NO SCROLL (global)
# ==========================================================
no_scroll_css = """
<style>
html, body { overflow: hidden !important; height: 100% !important; }
[data-testid="stAppViewContainer"] { overflow: hidden !important; height: 100% !important; }
[data-testid="stApp"] { overflow: hidden !important; height: 100% !important; }
section.main { overflow: hidden !important; }
::-webkit-scrollbar { width: 0px; height: 0px; }
</style>
"""
st.markdown(no_scroll_css, unsafe_allow_html=True)

# ==========================================================
# --- INITIALISIERUNG ---
# ==========================================================
qp = _get_qp()

if 'user_name' not in st.session_state: st.session_state.user_name = None
if 'group_name' not in st.session_state: st.session_state.group_name = None
if 'video_index' not in st.session_state: st.session_state.video_index = 0
if 'phase' not in st.session_state: st.session_state.phase = "viewing"
if 'session_data' not in st.session_state: st.session_state.session_data = []
if 'session_id' not in st.session_state: st.session_state.session_id = None
if 'seed' not in st.session_state: st.session_state.seed = None
if 'active_df' not in st.session_state: st.session_state.active_df = None
if 'db_saved' not in st.session_state: st.session_state.db_saved = False

# --- Restore Logic ---
if st.session_state.user_name is None and qp.get("user") is not None:
    st.session_state.user_name = qp.get("user")
    st.session_state.group_name = qp.get("grp")
    st.session_state.video_index = _parse_int(qp.get("i"), 0)
    st.session_state.phase = qp.get("phase", "viewing")
    st.session_state.seed = _parse_int(qp.get("seed"), None)
    st.session_state.session_id = qp.get("sid")

# --- RELOAD LOGIC (Scannt Ordner statt CSV) ---
if st.session_state.user_name is not None and st.session_state.active_df is None and st.session_state.seed is not None:
    try:
        # HIER WIRD GESCANNT STATT GELESEN
        full_df = scan_video_folders()
        
        if not full_df.empty and st.session_state.group_name:
            filtered_df = full_df[full_df["gruppe"] == st.session_state.group_name]
            real_anzahl = min(ANZAHL_VIDEOS, len(filtered_df))
            st.session_state.active_df = filtered_df.sample(n=real_anzahl, random_state=st.session_state.seed).reset_index(drop=True)
    except Exception as e:
        st.error(f"Fehler beim Scannen der Videos: {e}")

if st.session_state.user_name is not None:
    if st.session_state.seed is None: st.session_state.seed = int(time.time())
    if st.session_state.session_id is None: st.session_state.session_id = uuid.uuid4().hex

# ==========================================================
# --- RESTORE SESSION DATA ---
# ==========================================================
def _rehydrate_session_data_from_csv():
    file = os.path.join(DATA_FOLDER, "ergebnisse.csv")
    if not os.path.isfile(file): return
    try:
        df_all = pd.read_csv(file, sep=';')
    except: return
    if "SessionID" not in df_all.columns: return
    sid = st.session_state.session_id
    if not sid: return
    df_sid = df_all[df_all["SessionID"] == sid].copy()
    if df_sid.empty: return
    if "Zeitstempel" in df_sid.columns:
        try:
            df_sid["_ts"] = pd.to_datetime(df_sid["Zeitstempel"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            df_sid = df_sid.sort_values("_ts")
        except: pass
    st.session_state.session_data = df_sid.to_dict(orient="records")

if st.session_state.user_name is not None and not st.session_state.session_data:
    _rehydrate_session_data_from_csv()

# ==========================================================
# --- SPEICHERN ---
# ==========================================================
def save_result(video_name, wahl, korrektes_label):
    file = os.path.join(DATA_FOLDER, 'ergebnisse.csv')
    wahl_mapped = "fake" if wahl == "Deepfake" else "real"
    erfolg_wert = 1 if wahl_mapped.lower() == korrektes_label.lower() else 0

    daten_zeile = {
        "Zeitstempel": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "Testperson": st.session_state.user_name,
        "Gruppe": st.session_state.group_name,
        "SessionID": st.session_state.session_id,
        "Video": video_name,
        "Antwort_User": wahl,
        "Korrektes_Label": korrektes_label,
        "Erfolg": erfolg_wert,
        "Wahl_Mapped": wahl_mapped
    }
    st.session_state.session_data.append(daten_zeile)
    df_new = pd.DataFrame([daten_zeile])
    
    if not os.path.isfile(file):
        df_new.to_csv(file, index=False, sep=';')
    else:
        df_new.to_csv(file, mode='a', index=False, sep=';', header=False)

# ==========================================================
# --- CSS ---
# ==========================================================
viewing_css = """
<style>
    .block-container { max-width: 1200px !important; padding-top: 2rem; text-align: left !important; }
    video { width: 100% !important; max-width: 1600px; height: auto; border-radius: 10px; display: block; margin-left: 0; }
    h1 { margin-top: -10px; }
</style>
"""
voting_css = """
<style>
    .block-container { max-width: 800px !important; padding-top: 5rem; text-align: center !important; }
    div[data-testid="stRadio"] > div { justify-content: center !important; }
    div[data-testid="stRadio"] label { display: flex !important; align-items: center !important; justify-content: flex-start !important; gap: 15px !important; margin-bottom: 10px !important; }
    div[data-testid="stRadio"] label p { font-size: 30px !important; font-weight: bold !important; margin: 0 !important; line-height: 1.2 !important; }
    .stAlert { font-size: 20px !important; text-align: center !important; }
</style>
"""
results_css = """
<style>
.block-container { max-width: 1600px !important; padding-top: 0.8rem !important; padding-bottom: 0.8rem !important; padding-left: 2.2rem !important; padding-right: 2.2rem !important; text-align: left !important; }
div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
</style>
"""
start_css = """
<style>
  .start-desc  { font-size: 20px; line-height: 1.6; max-width: 950px; }
  .start-gap   { height: 12px; }
  .start-label { font-size: 20px; font-weight: 600; margin-top: 14px; margin-bottom: 6px; }
  .start-info  { font-size: 20px; line-height: 1.55; }
</style>
"""

# ==========================================================
# URL SYNC
# ==========================================================
def _sync_state_to_url():
    if st.session_state.user_name is None:
        _clear_qp(); return
    _set_qp(user=st.session_state.user_name, grp=st.session_state.group_name, i=st.session_state.video_index, phase=st.session_state.phase, seed=st.session_state.seed, sid=st.session_state.session_id)

# ==========================================================
# 1. STARTSCREEN
# ==========================================================
start_slot = st.empty()

def render_start():
    st.markdown(start_css, unsafe_allow_html=True)
    st.title("Willkommen zur Deepfake-Studie")
    st.markdown("""<div class="start-desc">In dieser Studie siehst du kurze Videos. Einige sind <b>echt</b>, andere sind <b>Deepfakes</b>.</div><div class="start-gap"></div>""", unsafe_allow_html=True)
    st.markdown('<div class="start-info">', unsafe_allow_html=True)
    st.info(f"""
**Deine Aufgabe:** Entscheide nach jedem Clip, ob das Video **echt** oder ein **Deepfake** ist.
⏱️ **Zeitlimit:** Du hast pro Video **20 Sekunden**.
ℹ️ **Umfang:** Es werden **{ANZAHL_VIDEOS} Videos** gezeigt.
""")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Studie starten", key="__start_btn", type="primary"):
        # 1. Verbindung herstellen und ID aus der Cloud holen
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            new_id = get_next_id_from_cloud(conn)
        except Exception as e:
            # Notfall-Fallback, falls Internet weg ist
            new_id = 999 
        
        st.session_state.user_name = str(new_id)
        
        # 2. Gruppe berechnen
        gruppen_rest = new_id % 3
        zugewiesene_gruppe = GRUPPEN_MAPPING[gruppen_rest]
        st.session_state.group_name = zugewiesene_gruppe

        # 3. Session Init
        st.session_state.video_index = 0
        st.session_state.phase = "viewing"
        st.session_state.session_data = []
        st.session_state.seed = int(time.time())
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.db_saved = False

        # 4. VIDEOS SCANNEN STATT CSV LADEN
        try:
            full_df = scan_video_folders()
            
            if full_df.empty:
                st.error("Fehler: Keine Videos gefunden! Bitte Ordnerstruktur prüfen.")
                st.stop()
            
            # Filtern nach Gruppe
            filtered_df = full_df[full_df["gruppe"] == zugewiesene_gruppe]
            
            if len(filtered_df) == 0:
                st.error(f"Fehler: Keine Videos für Gruppe '{zugewiesene_gruppe}' gefunden. Ordnernamen prüfen!")
                st.stop()

            real_anzahl = min(ANZAHL_VIDEOS, len(filtered_df))
            st.session_state.active_df = filtered_df.sample(
                n=real_anzahl, 
                random_state=st.session_state.seed
            ).reset_index(drop=True)

            _sync_state_to_url()
            start_slot.empty()
            st.rerun()
            
        except Exception as e:
            st.error(f"Kritischer Fehler: {e}")

if st.session_state.user_name is None:
    with start_slot.container(): render_start()
    st.stop()
else:
    start_slot.empty()
    _sync_state_to_url()

# ==========================================================
# HAUPT-LOGIK
# ==========================================================
if st.session_state.active_df is None:
    st.error("Fehler: Videoliste nicht geladen. Bitte Seite neu laden (F5).")
    st.stop()

df = st.session_state.active_df

if st.session_state.video_index < len(df):
    video_info = df.iloc[st.session_state.video_index]
    # WICHTIG: Hier nutzen wir den relativen Pfad aus dem Scan
    video_path = os.path.join(VIDEO_ROOT, video_info['full_path'])

    content_placeholder = st.empty()
    footer_placeholder = st.empty()

    if st.session_state.phase == "viewing":
        footer_placeholder.empty()
        with content_placeholder.container():
            st.markdown(viewing_css, unsafe_allow_html=True)
            col_titel, col_hinweis = st.columns([1, 1])
            with col_titel:
                st.title(f"Video {st.session_state.video_index + 1} von {len(df)}")
                st.caption(f"Teilnehmer ID: {st.session_state.user_name} | Gruppe: {st.session_state.group_name}")
            with col_hinweis:
                st.markdown("<div style='padding-top: 25px;'></div>", unsafe_allow_html=True)
                st.info("Das Video verschwindet automatisch nach 20 Sekunden.")

            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.error(f"Video nicht gefunden: {video_path}")

            col_links, col_rechts = st.columns([1, 1])
            with col_links: timer_anzeige = st.empty()
            with col_rechts:
                st.markdown("<div style='padding-top: 0px;'></div>", unsafe_allow_html=True)
                if st.button("Video fertig geschaut - zur Bewertung", use_container_width=True):
                    st.session_state.phase = "voting"; _sync_state_to_url(); st.rerun()

            for sekunde in range(20, -1, -1):
                timer_anzeige.subheader(f"⏳ Noch {sekunde} Sekunden")
                time.sleep(1)
            st.session_state.phase = "voting"; _sync_state_to_url(); st.rerun()

    elif st.session_state.phase == "voting":
        with content_placeholder.container():
            st.markdown(voting_css, unsafe_allow_html=True)
            st.warning("Das Video ist nun weg. Was denkst du?")
            wahl = st.radio("Ist dieses Video echt oder ein Deepfake?", ["Echt", "Deepfake"], index=None, key=f"entscheidung_{st.session_state.video_index}")

        if wahl:
            with footer_placeholder.container():
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Nächstes Video →", use_container_width=True):
                    save_result(video_info['filename'], wahl, video_info['label'])
                    st.session_state.video_index += 1
                    st.session_state.phase = "viewing"
                    _sync_state_to_url()
                    footer_placeholder.empty()
                    st.rerun()

# ==========================================================
# AUSWERTUNG
# ==========================================================
else:
    try:
        content_placeholder.empty()
        footer_placeholder.empty()
    except: pass

    st.markdown(results_css, unsafe_allow_html=True)
    st.balloons()
    st.markdown("<div style='padding-top: 25px;'></div>", unsafe_allow_html=True)
    st.title("Vielen Dank für deine Teilnahme!")

    results_df = pd.DataFrame(st.session_state.session_data)
    if results_df.empty:
        st.warning("Keine Ergebnisse vorhanden.")
        if st.button("Zurück zum Start"):
            _clear_qp(); st.session_state.clear(); st.rerun()
        st.stop()

    # Statistiken
    y_true = results_df['Korrektes_Label'].map({'real': 0, 'fake': 1})
    y_pred = results_df['Wahl_Mapped'].map({'real': 0, 'fake': 1})
    if y_true.isnull().any() or y_pred.isnull().any():
         st.error("Fehler bei der Datenauswertung (Label-Mismatch)."); st.stop()

    acc = accuracy_score(y_true, y_pred)
    roc_auc = None
    fpr, tpr = None, None
    if y_true.nunique() == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    left, right = st.columns([1.0, 1.35], gap="large")

    with left:
        st.subheader("Deine Statistik")
        acc_percent = acc * 100
        if acc_percent >= 70: st.success(f"✅ Genauigkeit: {acc_percent:.1f}%")
        elif acc_percent >= 60: st.warning(f"🟠 Genauigkeit: {acc_percent:.1f}%")
        else: st.error(f"🔴 Genauigkeit: {acc_percent:.1f}%")
        fig_cm, ax_cm = plt.subplots(figsize=(3.3, 3.0), dpi=150)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Echt", "Fake"])
        disp.plot(ax=ax_cm, values_format="d", colorbar=False)
        st.pyplot(fig_cm)

    with right:
        st.subheader("ROC")
        if fpr is not None:
            fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
            ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], lw=2, linestyle='--')
            st.pyplot(fig)
        else: st.info("ROC benötigt beide Klassen.")

# ☁️ GOOGLE SHEETS UPLOAD (MIT DUPLIKAT-CHECK)
    if not st.session_state.db_saved:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            # 1. Daten lesen (frisch, ohne Cache!)
            existing_data = conn.read(worksheet="Tabellenblatt1", ttl=0)
            
            # 2. Prüfen, ob SessionID schon existiert
            current_sid = st.session_state.session_id
            
            if not existing_data.empty and "SessionID" in existing_data.columns and current_sid in existing_data["SessionID"].values:
                # Fall A: Schon da -> Nix machen
                st.warning("⚠️ Daten für diese Session sind bereits in der Cloud! (Upload übersprungen)")
                st.session_state.db_saved = True
            else:
                # Fall B: Neu -> Uploaden
                if existing_data.empty: 
                    updated_data = results_df
                else: 
                    updated_data = pd.concat([existing_data, results_df], ignore_index=True)
                
                conn.update(worksheet="Tabellenblatt1", data=updated_data)
                st.session_state.db_saved = True
                st.success("✅ Ergebnisse wurden erfolgreich in der Cloud-Datenbank gespeichert.")
                
        except Exception as e:
            st.error(f"Fehler beim Cloud-Upload: {e}")
            st.info("Keine Sorge, die Daten sind lokal gesichert.")   

    # Forscher-Zusammenfassung
    summary_file = os.path.join(DATA_FOLDER, f"summary_{st.session_state.user_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    summary_data = pd.DataFrame([{"ID": st.session_state.user_name, "Gruppe": st.session_state.group_name, "Accuracy": acc, "AUC": (roc_auc if roc_auc is not None else "")}])
    summary_data.to_csv(summary_file, index=False, sep=';')
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Nächster Teilnehmer (Neue ID)", use_container_width=True):
        st.session_state.clear(); _clear_qp(); st.rerun()