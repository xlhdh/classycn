python runcrf.py 'data/sjw/*' 1000000 'd' 1 1 > 0001.txt
python runcrf.py 'data/sjw/*' 1000000 'd' 1 0 > 0002.txt
python runcrf.py 'data/sjw/*' 1652528 'd' 1 1 > 0003.txt
python runcrf.py 'data/sjw/*' 1652528 'd' 1 0 > 0004.txt


python runhmm.py 'data/24s/*' 68040 1 > 0005.txt
python runhmm.py 'data/24s/*' 68040 0 > 0006.txt
#python runhmm.py 'data/sjw/*' 1000000 1 > 0007.txt
python runhmm.py 'data/sjw/*' 1000000 0 > 0008.txt
python runhmm.py 'data/sjw/*' 1652528 1 > 0009.txt
python runhmm.py 'data/sjw/*' 1652528 0 > 0010.txt

python runcrf-100kiter.py 'data/24s/*' 68040 'd' 1 1 > 0011.txt
python runcrf-100kiter.py 'data/24s/*' 68040 'd' 1 0 > 0012.txt
