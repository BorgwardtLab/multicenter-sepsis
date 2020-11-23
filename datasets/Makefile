all: demo/data aumc/data eicu/data hirid/data mimic3/data physionet2019/data 
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
.DELETE_ON_ERROR:
SUBDIRS = physionet2019 eicu hirid mimic3 
SUBDIRSCLEAN=$(addsuffix _clean,$(SUBDIRS))
.PHONY: all clean tests $(SUBDIRS) $(SUBDIRSCLEAN)

demo/data:
	cd demo; make
aumc/data:
	cd aumc; make
eicu/data:
	cd eicu; make 
hirid/data: 
	cd hirid; make 
mimic3/data: 
	cd mimic3; make 
physionet2019/data: 
	cd physionet2019; make 

clean:
	cd demo; make clean; cd ../
	cd aumc; make clean; cd ../
	cd eicu; make clean; cd ../  
	cd hirid; make clean; cd ../
	cd mimic3; make clean; cd ../
	cd physionet2019; make clean; cd ../
