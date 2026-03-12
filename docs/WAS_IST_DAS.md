# Woran arbeitet Felix? 🌟

## Die Kurzfassung

Ich baue eine **Landkarte von allem, was Wissenschaftler über Kernfusion wissen** — die gleiche Energie, die die Sonne antreibt — und nutze Mathematik, um herauszufinden, **was sie noch nicht herausgefunden haben**.

## Etwas ausführlicher

Wissenschaftler haben **über 8.000 Paper** über Kernfusion geschrieben (die Technologie, die uns unbegrenzte, saubere Energie liefern könnte). Ein Team in Italien hat all diese Paper mithilfe von KI ausgelesen und die **50.000 wichtigsten Konzepte** herausgezogen — Dinge wie „Tokamak" (ein donutförmiger Reaktor), „Plasma" (superheißes Gas), „Deuterium" (eine Art Wasserstoff-Brennstoff) usw.

Ich habe all diese Konzepte genommen und einen riesigen **Wissensgraphen** gebaut — stell dir das vor wie ein Netz, in dem jedes Konzept ein Punkt ist, und immer wenn zwei Konzepte zusammen in einem Paper vorkommen, werden sie durch eine Linie verbunden. Je mehr Paper sie zusammen erwähnen, desto dicker die Linie.

Dann habe ich eine ganze Reihe mathematischer Werkzeuge benutzt, um dieses Netz zu analysieren:

- **Welche Konzepte sind am wichtigsten?** → „Tokamak" ist mit Abstand auf Platz 1
- **Gibt es Cluster verwandter Themen?** → Ja, 1.402 Forschungsgemeinschaften
- **Gibt es Lücken in der Forschung?** → Ja! Wir haben **45 Löcher** gefunden, wo Wissenschaftler Konzeptpaare untersuchen, aber nie alle drei zusammen
- **Welche Verbindungen fehlen?** → Wir haben 200 Verknüpfungen vorhergesagt, die existieren sollten, aber nicht vorhanden sind
- **Welche Konzepte verbinden verschiedene Forschungsgruppen?** → „Tokamak", „Elektron" und „Magnetfeld" überbrücken die meisten Gemeinschaften
- **Folgt das Vokabular natürlichen Gesetzen?** → Ja! Die Häufigkeitsverteilung der Entitäten folgt dem Zipfschen Gesetz (Potenzgesetz)
- **Kann man eine formale Ontologie bauen?** → Ja! Wir haben automatisch eine OWL-2-Ontologie mit 2.487 Tripeln aus der Graphstruktur erzeugt
- **Kann man Fragen stellen?** → Ja! Eine Graph-Abfrageschnittstelle ermöglicht die interaktive Erforschung von Pfaden, Trends und Lücken

## Warum ist das wichtig?

Wenn wir die **blinden Flecken** in der Fusionsforschung finden, können wir vorschlagen, wo Wissenschaftler als Nächstes hinschauen sollten. Es ist wie ein Navi für wissenschaftliche Entdeckungen — statt ziellos herumzuirren, kann man sehen, wo das unerforschte Gebiet liegt.

Außerdem haben wir eine **Datenakquisitions-Pipeline** gebaut, die Open-Access-Paper herunterladen kann (einschließlich arXiv-Preprints als Rückfalloption für kostenpflichtige Paper) und sie in einen Knowledge-Graph-Builder einspeist für noch tiefere Analyse.

## Wie sieht das Ganze aus?

Das Projekt erzeugt **32 Diagramme**, eine interaktive Web-Visualisierung, in der man herumklicken kann, eine OWL-Ontologie-Datei und einen „Lückenbericht" mit konkreten Forschungsideen, die noch niemand ausprobiert hat. Es gibt auch ein Kommandozeilen-Abfragetool, mit dem man z.B. `"tokamak -> stellarator"` eingeben kann, um den kürzesten Pfad zwischen Konzepten zu finden.

## Der Tech-Stack (falls es dich interessiert)

Python, Neo4j (eine Graphdatenbank) und jede Menge Mathe (Topologie, Spektraltheorie, Informationstheorie, Formale Begriffsanalyse, Zipfsches Gesetz). Dazu eine Datenakquisitions-Pipeline mit OpenAlex für Open-Access-Metadaten und arXiv für Preprints. Läuft alles in Docker auf diesem Rechner.

---

*Entstanden mit ☕ und zu vielen späten Nächten im GAIA Lab, FH Südwestfalen*
