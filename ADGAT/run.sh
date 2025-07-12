# metattack
python3 ADGAT/train.py --dataset cora --th1 0.75 --th2 0.9 --ptb 0.05 --attack meta
python3 ADGAT/train.py --dataset citeseer --th1 0.7 --th2 0.85 --ptb 0.05 --attack meta
python3 ADGAT/train.py --dataset cora_ml --th1 0.8 --th2 0.95 --ptb 0.05 --attack meta

# nettack
python3 ADGAT/train.py --dataset cora --th1 0.72 --th2 0.8 --ptb 0.05 --attack nettack

# hattack
python3 ADGAT/train.py --dataset citeseer --th1 0.7 --th2 0.85 --ptb 0.05 --attack hattack