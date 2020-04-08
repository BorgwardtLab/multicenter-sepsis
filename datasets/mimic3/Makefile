all: data/split_info.pkl

.DELETE_ON_ERROR:
.PHONY: all tests

# data/combined.pck: data/extracted
# 	pipenv run python ../physionet2019/scripts/convert_data_to_binary.py --folders $^ --output $@

data/extracted: data/downloads/mimic_mv.zip
	-mkdir -p data/extracted
	unzip -q -j $< -d $@
	cd $@ && for f in *.psv; do mv $$f p$$f; done

data/downloads/mimic_mv.zip:
	-mkdir -p $(@D)
	curl https://polybox.ethz.ch/index.php/s/X2doEsI1suUsAqs/download -o $@

data/split_info.pkl: data/extracted
	pipenv run python ../physionet2019/scripts/create_splits.py

clean:
	rm -r data/*
