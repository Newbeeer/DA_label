#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 1234

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --seed 1 --dann

CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src svhn --tgt mnist --orth --seed 1 --dann

CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src svhn --tgt mnist --seed 1 --dann

CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 123 --dann

CUDA_VISIBLE_DEVICES=0 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --seed 123 --dann



