# Dateien ohne Nutzen für Projekt und Bewertung

## ❌ LÖSCHEN - Komplett unnötig

### 1. Temporäre/Fehlerhafte Dateien
```
./nul                                    # Fehlerhaft erstellte Datei (Git kann sie nicht adden)
```

### 2. IDE/Editor Konfigurationen (optional)
```
./.claude/settings.local.json            # Claude Code lokale Einstellungen
./.devcontainer/servers.json             # VSCode DevContainer Config
```
**Empfehlung:** Behalten, wenn du mit Claude Code/VSCode arbeitest. Sonst löschen.

### 3. Leere Verzeichnisse
```
./data/                                  # Komplett leer
./tests/                                 # Komplett leer
```
**Empfehlung:** Löschen, bringen keinen Nutzen für Bewertung.

### 4. Alte/Redundante Dokumentation
```
./WAS_NOCH_ZU_TUN_IST.md                 # Alte ToDo-Liste (veraltet)
./QUICKSTART.md                          # Redundant zu README.md
./PRESENTATION_OUTLINE.md                # Alte Präsentations-Outline (veraltet)
```

### 5. Temporäre Analyse-Scripts (nach Nutzung)
```
./extract_plots.py                       # Plots bereits extrahiert
./rename_plots.py                        # Plots bereits umbenannt
./run_notebooks.py                       # Notebooks bereits ausgeführt
./run_analysis_summary.py                # Einmalige Analyse (bereits gemacht)
./setup_project.py                       # Setup bereits durchgeführt
```
**Empfehlung:** Können gelöscht werden, da Aufgabe erledigt. Oder behalten für Reproduzierbarkeit.

### 6. Alte Markdown Dokumentationen
```
./INTERPRETATIONS_EDA.md                 # EDA Interpretationen (veraltet)
./presentation/ZAHLEN_SAMMELN.md         # Temporäre Notizen
./presentation/SCREENSHOT_CHECKLIST.md   # Checkliste (bereits erledigt)
./presentation/SCREENSHOT_GUIDE.md       # Guide (bereits erledigt)
./presentation/SCREENSHOTS_READY.md      # Status Dokument (veraltet)
```

### 7. Doppelte/Ungenutzte Plots
```
# Original Plots (bereits in renamed/ vorhanden):
./presentation/screenshots/plots/01_exploratory_analysis/*.png  (12 Dateien)
./presentation/screenshots/plots/02_model_training/*.png        (7 Dateien)
./presentation/screenshots/plots/03_clustering_analysis/*.png   (10 Dateien)
./presentation/screenshots/plots/04_geo_visualization/*.png     (1 Datei)

# Alte evaluation_results (redundant zu notebooks):
./evaluation_results/confusion_matrix_sentiment.png
./evaluation_results/regression_evaluation.png
```
**Empfehlung:** Originale in `plots/` löschen, nur `renamed/used/` behalten.

---

## ⚠️ OPTIONAL BEHALTEN

### Für Reproduzierbarkeit
```
./extract_plots.py                       # Falls Plots neu generiert werden müssen
./rename_plots.py                        # Falls Plots neu benannt werden müssen
./run_notebooks.py                       # Falls Notebooks neu ausgeführt werden müssen
./setup_project.py                       # Falls Projekt neu aufgesetzt wird
```

### Für Entwicklung
```
./.claude/                               # Wenn du Claude Code verwendest
./.devcontainer/                         # Wenn du VSCode DevContainer verwendest
./docker-compose.yml                     # Wenn du Docker verwendest
```

---

## ✅ BEHALTEN - Wichtig für Bewertung

### Core Projekt Dateien
```
./README.md                              # Projekt Übersicht
./requirements.txt                       # Python Dependencies
./.env.sample                            # Env Beispiel
./.gitignore                             # Git Konfiguration
```

### Datenbank
```
./init-db.sql                            # Initiales DB Setup
./sql/schema.sql                         # Datenbank Schema
```

### Source Code
```
./src/**/*.py                            # Alle Python Module (WICHTIG!)
./dashboards/app.py                      # Dash Dashboard
```

### Notebooks (WICHTIG!)
```
./notebooks/01_exploratory_analysis.ipynb
./notebooks/02_model_training.ipynb
./notebooks/03_clustering_analysis.ipynb
./notebooks/04_geo_visualization.ipynb
```

### Models (WICHTIG!)
```
./models/*.pkl                           # Trainierte ML Modelle
```

### Präsentation Materialien
```
./presentation/MovieMind_Speaker_Script.md        # Speaker Notizen
./presentation/PRESENTATION_SLIDES.md             # Präsentation Content
./presentation/METRICS_SUMMARY.md                 # Metriken Zusammenfassung
./presentation/screenshots/renamed/used/*.png     # Verwendete Screenshots (10 Dateien)
./presentation/screenshots/renamed/INDEX.md       # Plot Index
```

### Dokumentation
```
./DECIMAL_FIX_IMPACT.md                  # Wichtige technische Dokumentation

---

## 📋 CLEANUP KOMMANDOS

### Sichere Variante (nur offensichtlich unnötige Dateien):
```bash
# 1. Fehlerhafte Datei
rm -f nul

# 2. Leere Verzeichnisse
rmdir data tests 2>/dev/null

# 3. Doppelte Plots (Original-Plots)
rm -rf presentation/screenshots/plots/

# 4. Alte evaluation_results
rm -rf evaluation_results/
```

### Aggressive Variante (mehr aufräumen):
```bash
# Zusätzlich zur sicheren Variante:

# Alte Dokumentation
rm -f WAS_NOCH_ZU_TUN_IST.md
rm -f QUICKSTART.md
rm -f PRESENTATION_OUTLINE.md
rm -f INTERPRETATIONS_EDA.md

# Temporäre Analyse Scripts
rm -f extract_plots.py
rm -f rename_plots.py
rm -f run_notebooks.py
rm -f run_analysis_summary.py
rm -f setup_project.py

# Alte Präsentations-Docs
rm -f presentation/ZAHLEN_SAMMELN.md
rm -f presentation/SCREENSHOT_CHECKLIST.md
rm -f presentation/SCREENSHOT_GUIDE.md
rm -f presentation/SCREENSHOTS_READY.md

# Ungenutzte Plots
rm -rf presentation/screenshots/renamed/unused/
```

---

## 📊 ZUSAMMENFASSUNG

### Empfohlene Löschungen:
- **nul** - fehlerhaft ❌
- **data/, tests/** - leer ❌
- **presentation/screenshots/plots/** - Duplikate (30 Dateien) ❌
- **evaluation_results/** - redundant ❌
- **WAS_NOCH_ZU_TUN_IST.md, QUICKSTART.md, etc.** - veraltet ❌

### Geschätzte Größenersparnis:
- Original Plots: ~2-3 MB
- Evaluation Results: ~200 KB
- Dokumentation: ~50 KB
- **TOTAL: ~2.5-3.5 MB**

### Nach Cleanup:
- **Behalten:** ~5-6 MB (Code, Notebooks, Models, verwendete Plots)
- **Projekt bleibt vollständig funktionsfähig für Bewertung**

---

**Empfehlung:** Starte mit der "Sicheren Variante" und entferne dann nach Bedarf weitere Dateien.
