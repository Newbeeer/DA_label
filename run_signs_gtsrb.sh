#CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src mnist --tgt svhn --orth --seed 1234

CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --orth --seed 123 --dann

CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --seed 123 --dann

CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --orth --seed 1234 --dann

CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --seed 1234 --dann

CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --orth --seed 12 --dann

CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src signs --tgt gtsrb --seed 12 --dann

