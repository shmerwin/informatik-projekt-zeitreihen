# Dokumentation bauen

Benötigt werden eine LaTeX-Distribution mit `pdflatex` sowie Biber. Im
Verzeichnis `Dokumentation` erzeugt folgender Lauf die finale `thesis.pdf`:

```text
pdflatex -interaction=nonstopmode -halt-on-error thesis.tex
biber thesis
pdflatex -interaction=nonstopmode -halt-on-error thesis.tex
pdflatex -interaction=nonstopmode -halt-on-error thesis.tex
```

Wenn GNU Make installiert ist, führt `make` dieselbe Abfolge aus. Zentrale
Angaben des Titelblatts stehen in `TeXFiles/000_Settings.tex`.
















