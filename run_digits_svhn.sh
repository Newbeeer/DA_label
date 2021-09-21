#CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src digits --tgt svhn --seed 1 --dann

#CUDA_VISIBLE_DEVICES=3 python3 vada_train.py --r 0.7 0.3 --src digits --tgt svhn --seed 1  --orth

#CUDA_VISIBLE_DEVICES=2 python3 vada_train.py --r 0.7 0.3 --src digits --tgt svhn --seed 12 --dann

CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src digits --tgt svhn --seed 12  --orth

CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src digits --tgt svhn --seed 12

CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src digits --tgt svhn --seed 123  --orth

CUDA_VISIBLE_DEVICES=1 python3 vada_train.py --r 0.7 0.3 --src digits --tgt svhn --seed 123