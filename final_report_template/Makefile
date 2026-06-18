MAIN = main

.PHONY: all clean distclean

all:
	latexmk -xelatex -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).blg

distclean:
	latexmk -C $(MAIN).tex
