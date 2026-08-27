MASTER = thesis

.PHONY: all clean

all:
	pdflatex -interaction=nonstopmode -halt-on-error $(MASTER).tex
	biber $(MASTER)
	pdflatex -interaction=nonstopmode -halt-on-error $(MASTER).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MASTER).tex

clean:
	$(RM) $(MASTER).aux $(MASTER).bbl $(MASTER).bcf $(MASTER).blg \
		$(MASTER).log $(MASTER).out $(MASTER).run.xml $(MASTER).toc
