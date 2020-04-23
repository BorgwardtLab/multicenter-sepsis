all: data/split_info.pkl

.DELETE_ON_ERROR:
.PHONY: all tests

data/extracted: ../downloads/hirid.zip
	-mkdir -p data/extracted
	unzip -q -j $< -d $@
	cd $@ && for f in *.psv; do mv $$f p$$f; done

data/split_info.pkl: data/extracted
	pipenv run python ../physionet2019/scripts/create_splits.py

clean:
	rm -r data/*
