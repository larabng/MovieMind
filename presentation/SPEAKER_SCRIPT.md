# MovieMind - Speaker Script
## 15-Minuten Praesentation (3 Personen)

---

# Rollenverteilung

| Person | Themen | Zeit |
|--------|--------|------|
| **Speaker 1** | Title, Introduction, Materials | ~5 Min |
| **Speaker 2** | Methods (Preprocessing, ML, EDA) | ~5 Min |
| **Speaker 3** | Results, Demo, Conclusions | ~5 Min |

---

# SPEAKER 1 (ca. 5 Minuten)

## Slide 1: Title (30 Sek)

> "Willkommen zu unserer Praesentation ueber MovieMind - ein End-to-End Analytics System fuer Film-Reviews.
>
> Unser Team besteht aus [Name 1], [Name 2] und [Name 3]. Ich werde euch durch die Einfuehrung und Daten fuehren, [Name 2] erklaert die Methoden, und [Name 3] praesentiert die Ergebnisse."

---

## Slide 2: Introduction - Background (1 Min)

> "Warum analysieren wir Film-Reviews?
>
> Streaming-Plattformen wie Netflix und Disney+ erhalten taeglich tausende von Nutzer-Reviews. Eine manuelle Analyse ist zeitaufwendig und inkonsistent.
>
> Der globale Streaming-Markt ist ueber 300 Milliarden Dollar wert. Fruehzeitige Erkennung negativer Trends kann Millionen an Marketing-Kosten sparen.
>
> Studios brauchen automatisierte Sentiment-Analyse, um schnell auf Nutzerfeedback reagieren zu koennen."

---

## Slide 3: Objective & Research Questions (1 Min)

> "Unser Ziel ist es, eine komplette Analytics-Pipeline zu bauen, die unstrukturierte Reviews in verwertbare Insights transformiert.
>
> Wir haben vier zentrale Forschungsfragen:
>
> **Erstens:** Kann Sentiment in Reviews akkurat klassifiziert werden - also positiv, neutral oder negativ?
>
> **Zweitens:** Koennen wir Film-Bewertungen von 0 bis 10 aus dem Review-Text vorhersagen?
>
> **Drittens:** Welche Muster existieren ueber Genres und Zeit hinweg?
>
> **Und viertens:** Wie gruppieren sich Filme basierend auf Publikumsreaktionen?"

---

## Slide 4: Data Source (1 Min)

> "Fuer unsere Daten nutzen wir die TMDb API - The Movie Database - den Industriestandard fuer Film-Metadaten.
>
> Wir haben 500 bis 1000 Filme gesammelt mit jeweils mindestens 30 Reviews pro Film. Das ergibt tausende von Reviews insgesamt.
>
> Die Daten umfassen Film-Metadaten wie Titel, Genre, Laufzeit, Budget und Revenue. Dazu die Reviews mit Inhalt, Autor, Rating und Datum.
>
> Unsere Sammelstrategie kombiniert populaere Filme mit Top-bewerteten Filmen, um eine gute Mischung zu bekommen."

---

## Slide 5: Database Schema (1.5 Min)

> "Die Daten speichern wir in einer PostgreSQL-Datenbank.
>
> Wir haben drei Haupt-Tabellen: Movies, Reviews und Countries fuer geo-Visualisierungen.
>
> Besonders wichtig sind unsere Optimierungen:
>
> Wir nutzen GIN-Indexes fuer die Genre-Arrays - das beschleunigt Abfragen erheblich.
>
> Ausserdem haben wir drei SQL-Views erstellt:
> - **movie_review_stats** aggregiert Statistiken pro Film
> - **genre_sentiment_analysis** zeigt Sentiment-Verteilung nach Genre
> - **temporal_sentiment_trends** analysiert Trends ueber Zeit
>
> Das zeigt, dass wir nicht nur die Daten speichern, sondern auch direkt in der Datenbank analysieren."
>
> *[UEBERGABE]* "Jetzt uebergebe ich an [Name 2], der unsere Methoden erklaert."

---

# SPEAKER 2 (ca. 5 Minuten)

## Slide 6: Text Preprocessing (1.5 Min)

> "Danke [Name 1]. Ich erklaere jetzt unsere NLP-Pipeline.
>
> Text-Daten sind unstrukturiert und muessen aufbereitet werden. Unser TextProcessor durchlaeuft sieben Schritte:
>
> **Erstens:** HTML-Tags entfernen mit BeautifulSoup
> **Zweitens:** URLs entfernen mit Regex
> **Drittens:** Alles in Kleinbuchstaben
> **Viertens:** Sonderzeichen entfernen
> **Fuenftens:** Tokenisierung mit NLTK
> **Sechstens:** Stopwoerter entfernen
> **Und siebtens:** Lemmatisierung - also Woerter auf ihre Grundform bringen
>
> Zusaetzlich extrahieren wir Features wie Textlaenge, Wortanzahl, Satzanzahl und Ausrufezeichen-Haeufigkeit. Diese koennen spaeter als Metadaten-Features verwendet werden."

---

## Slide 7: Machine Learning Models (1.5 Min)

> "Wir setzen drei Machine-Learning-Modelle ein.
>
> **Fuer die Sentiment-Klassifikation** verwenden wir Logistic Regression mit TF-IDF-Vektorisierung. Wir nutzen Uni- und Bigrams mit bis zu 5000 Features. Die Klassen-Gewichtung ist 'balanced', um mit ungleich verteilten Daten umzugehen.
>
> **Fuer die Score-Prediction** nutzen wir Ridge Regression - also L2-regularisierte lineare Regression. Wir kombinieren TF-IDF-Features mit Metadaten wie Textlaenge und Wortanzahl. Die Vorhersage wird auf den Bereich 0-10 geclipt.
>
> **Fuer das Clustering** setzen wir K-Means ein. Die optimale Cluster-Anzahl bestimmen wir mit der Elbow-Methode und Silhouette-Score."

---

## Slide 8: Exploratory Data Analysis (2 Min)

> "Unsere EDA folgt einem strukturierten Ansatz.
>
> **Univariate Analyse:** Wir schauen uns Verteilungen einzelner Variablen an - Rating-Histogramme, Laufzeit-Verteilung, Review-Laengen.
>
> **Bivariate Analyse:** Hier analysieren wir Zusammenhaenge - zum Beispiel die Korrelationsmatrix zwischen Laufzeit, Budget, Revenue und Ratings. Oder Genre-spezifische Rating-Muster.
>
> **Und ganz wichtig - statistische Tests mit P-Werten:**
>
> Der Chi-Squared-Test zeigt, ob Genre und Rating-Kategorie zusammenhaengen. Ergebnis: p unter 0.05 - also signifikant.
>
> ANOVA testet, ob sich Ratings zwischen Genres unterscheiden. Ergebnis: p unter 0.01 - Drama und Thriller haben signifikant hoehere Ratings.
>
> Pearson-Korrelation zwischen Laufzeit und Rating ist ebenfalls signifikant.
>
> Diese statistischen Tests mit expliziten P-Werten sind wichtig fuer die wissenschaftliche Rigorositaet."
>
> *[UEBERGABE]* "Jetzt uebergebe ich an [Name 3] fuer die Ergebnisse."

---

# SPEAKER 3 (ca. 5 Minuten)

## Slide 9: Classification Results (1 Min)

> "Danke [Name 2]. Ich praesentiere jetzt unsere Ergebnisse.
>
> Die Confusion Matrix zeigt die Performance unseres Sentiment-Classifiers.
>
> Was faellt auf? Die **positive Klasse wird sehr gut erkannt** - 214 von rund 235 korrekt klassifiziert. Das ist eine Accuracy von ueber 90% fuer diese Klasse.
>
> Die Unterscheidung zwischen neutral und negativ ist schwieriger. Das ist typisch, weil neutrale Reviews oft gemischte Signale haben.
>
> Insgesamt erreichen wir etwa 80% Accuracy. Die Class-Weighting hilft bei der Imbalance, aber es gibt noch Verbesserungspotential."

---

## Slide 10: Regression Results (1 Min)

> "Bei der Score-Prediction sehen wir die Residual-Plots.
>
> Der linke Plot zeigt Residuals gegen vorhergesagte Werte. Die Residuals sind um Null zentriert - das ist gut, es zeigt keinen systematischen Bias.
>
> Der rechte Plot zeigt Predicted vs. Actual. Die Punkte liegen nahe der Diagonale - je naeher, desto besser die Vorhersage.
>
> Unser R-Squared und RMSE zeigen, dass das Modell einen signifikanten Teil der Varianz erklaert.
>
> Die wichtigsten Features fuer gute Scores sind Woerter wie 'brilliant', 'masterpiece', 'excellent'. Fuer schlechte Scores: 'boring', 'disappointing', 'waste'."

---

## Slide 11: Clustering Results (1 Min)

> "Das K-Means-Clustering mit k gleich 5 zeigt interessante Muster.
>
> Wir sehen fuenf distinkte Cluster:
> - Cluster 1: Blockbuster-Action mit moderaten Ratings
> - Cluster 2: Indie-Dramen mit hoher kritischer Anerkennung
> - Cluster 3: Familien-Komoedien
> - Cluster 4: Horror und Thriller mit polarisierten Meinungen
> - Cluster 5: Underperformer mit niedrigen Ratings
>
> Der Silhouette-Score bestaetigt gute Cluster-Trennung.
>
> Der Business-Nutzen: Studios koennen Marketingstrategien nach Cluster differenzieren und Fruehwarnsignale bei Cluster 5 erkennen."

---

## Slide 12: Statistical Insights (30 Sek)

> "Zusammenfassend die wichtigsten statistischen Erkenntnisse:
>
> Genre ist stark mit Rating assoziiert. Drama und Thriller performen signifikant besser.
>
> Laengere Filme tendieren zu hoeheren, aber auch polarisierten Ratings.
>
> Alle unsere Hypothesentests haben signifikante P-Werte unter 0.05."

---

## Slide 13: Live Demo (1 Min) - OPTIONAL

> "Falls Zeit bleibt, zeige ich kurz unser Dashboard.
>
> *[Wenn Demo gezeigt wird:]*
> Hier kann man einen Review-Text eingeben. Das System preprocessing den Text, wendet unsere trainierten Modelle an, und zeigt Sentiment plus vorhergesagten Score.
>
> Dazu Text-Statistiken und die Database-Uebersicht.
>
> *[Wenn keine Demo:]*
> Das Dashboard ist eine Flask-Dash-Applikation mit Live-Prediction, Database-Statistiken und Visualisierungen."

---

## Slide 14: Conclusions (30 Sek)

> "Zusammenfassend:
>
> Wir haben eine vollstaendige End-to-End-Pipeline gebaut - von der API bis zur Vorhersage.
>
> Unsere Modelle erreichen solide Performance mit statistisch validierten Ergebnissen.
>
> Als Limitationen: Wir arbeiten nur mit englischen Reviews, und die TMDb-Ratings koennen sich von anderen Plattformen unterscheiden.
>
> Fuer die Zukunft waere Deep Learning mit BERT interessant, sowie Multi-Language-Support."
>
> *[ABSCHLUSS]* "Vielen Dank fuer eure Aufmerksamkeit. Der Appendix zeigt unsere detaillierten Punktenachweise mit Screenshots."

---

# Timing-Checkliste

| Abschnitt | Ziel-Zeit | Checkpoint |
|-----------|-----------|------------|
| Speaker 1 Start | 0:00 | |
| Slide 3 fertig | 2:30 | |
| Speaker 1 Ende | 5:00 | |
| Speaker 2 Start | 5:00 | |
| Slide 7 fertig | 8:00 | |
| Speaker 2 Ende | 10:00 | |
| Speaker 3 Start | 10:00 | |
| Slide 12 fertig | 13:00 | |
| Speaker 3 Ende | 15:00 | |

---

# Tipps fuer die Aufnahme

1. **Vorbereitung:**
   - Skript mehrmals durchlesen
   - Wichtige Woerter markieren
   - Timing ueben

2. **Technisches:**
   - Gutes Mikrofon verwenden
   - Ruhige Umgebung
   - Bildschirmaufnahme testen

3. **Praesentation:**
   - Frei sprechen, nicht ablesen
   - Auf Slides zeigen/highlighten
   - Pausen bei Uebergaben

4. **Uebergaben:**
   - Kurz + klar: "Jetzt uebergebe ich an [Name]..."
   - Blickkontakt (virtuell) halten

5. **Appendix:**
   - Nicht live praesentieren
   - Nur erwaehnen: "Details im Appendix"

---

# Notfall-Kuerzungen

Falls Zeit knapp wird, kuerze hier:

1. **Slide 4 (Data Source):** Nur API erwaehnen, Details skippen
2. **Slide 8 (EDA):** Fokus auf statistische Tests, univariate skippen
3. **Slide 13 (Demo):** Komplett skippen, nur erwaehnen
4. **Slide 12 (Stats Insights):** In Conclusions integrieren

---

# Appendix-Verweis

Am Ende von Slide 14:

> "Im Appendix findet ihr detaillierte Dokumentation unserer Punktenachweise:
> - PostgreSQL Schema mit Views und Indexes
> - Alle statistischen Tests mit P-Werten
> - K-Means Clustering mit Elbow-Plot und Silhouette
> - Regression-Diagnostics mit Residual-Plots
> - Confusion Matrix und Classification Report
> - Code-Snippets aus unseren Implementierungen"

---

# Ende des Skripts
