INPUT_NAME := paper
OUTPUT_NAME := paper

ALL_OUTPUTS := $(OUTPUT_NAME).pdf

.PHONY: all clean
all: $(ALL_OUTPUTS)

$(ALL_OUTPUTS): $(INPUT_NAME).md $(INPUT_NAME).bib
	@docker run --rm --platform linux/amd64 \
	  -v "$(CURDIR):/data" \
	  -u $(shell id -u):$(shell id -g) \
	  -e JOURNAL=joss \
	  openjournals/inara \
	  -o pdf $(INPUT_NAME).md

clean:
	@rm -f $(ALL_OUTPUTS)
