all: data/split_info.pkl

.DELETE_ON_ERROR:
.PHONY: all tests

data/combined.pck: data/extracted
	pipenv run python scripts/convert_data_to_binary.py --folders $^ --output $@

data/extracted: data/downloads/training_setB.zip 
        #data/downloads/training_setA.zip 
	mkdir -p data/extracted
	unzip -q -j $(word 1,$^) -d $@
	#unzip -q -j $(word 2,$^) -d $@

data/downloads:
	# Download files
	mkdir -p $@ && cd $@ && curl \
	    https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip \
	    https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip \
	    -O -O

data/downloads/training_setA.zip: data/downloads
data/downloads/training_setB.zip: data/downloads

data/split_info.pkl: data/extracted
	pipenv run python scripts/create_splits.py

clean:
	rm -r data/*
