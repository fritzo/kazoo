
MAIN = matching.pdf

current_target: $(MAIN)

system_diagram.pdf: system_diagram.fig
	fig2dev -L pdf system_diagram.fig system_diagram.pdf
synth_windows.pdf: figures.py
	python figures.py

FIGS = system_diagram.pdf synth_windows.pdf

design.pdf: design.tex $(FIGS)
	pdflatex design.tex

matching.pdf: matching.tex
	pdflatex matching.tex

#----( misc )------------------------------------------------------------------

all: design.pdf matching.pdf

todo:
	@grep '.*(T[0-9]\+\(.[0-9]\+\)*) ' *.text *.tex *.py --color=yes

view: $(MAIN)
	evince $(MAIN) &

clean:
	rm -f *.aux *.log *.dvi *.pdf

