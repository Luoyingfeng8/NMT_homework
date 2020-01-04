
# trian
CUDA_VISIBLE_DEVICES=2 nohup python train.py -data data/fr-en/data.pt -save data/fr-en  &

# decode
CUDA_VISIBLE_DEVCES=2 python  translate.py -model data/test/checkpoint_test.pt -src data/test/val.de.atok -vocab data/test/data.pt  -output data/test/predict.txt

