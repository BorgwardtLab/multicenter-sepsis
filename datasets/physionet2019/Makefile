all: data/extracted
splits: data/split_info.pkl

.DELETE_ON_ERROR:
.PHONY: all clean

# data/combined.pck: data/extracted
# 	pipenv run python scripts/convert_data_to_binary.py --folders $^ --output $@

data/extracted: ../downloads/challenge.zip
        #data/downloads/training_setB.zip 
	mkdir -p data/extracted
	unzip -q -j $(word 1,$^) -d $@
	cd $@ && for f in *.psv; do mv $$f p$$f; done

# data/downloads:
# 	# Download files (currently not used)
# 	mkdir -p $@ && cd $@ && curl \
# 	    https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip \
# 	    -O -O

# data/downloads/training_setA.zip: data/downloads
# data/downloads/training_setB.zip: data/downloads

data/split_info.pkl: data/extracted
	pipenv run python scripts/create_splits.py

clean:
	rm -r data/*
