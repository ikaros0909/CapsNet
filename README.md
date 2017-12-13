# CapsNet 
- CapsNet을 이용하여 MNIST 손글씨 인식 테스트를 하고 있습니다.

##hyperparameter description
- --batch_size=100 batch size
- --epochs=1 이포크 학습 반복수
- --lam_recon=0.392 784 * 0.0005, paper uses sum of SE, here uses MSE 0.392
- --num_routing=2 routing 수(disit caps에서 primary caps에 routing 되는 횟수)
- --shift_fraction=0.2 비율만큼 쉬프트(이동)하여 학습
- --debug=1 디버그모드여부(1일경우 디버그 파일을 생성함)
- --save_dir='./result/trained_model_test.h5' train모드일경우 모델을 저장할 파일명 지정
- --is_training=0 학습모드 여부(1이면 train, 0이면 test)
- --weights='/trained_model.h5' 테스트할때 불러들일 모델 파일명
- --lr=0.001 learning rate

## Command 창에서 실행
- python capsulenet_test.py --batch_size=100 --epochs=1 --lam_recon=0.392 --num_routing=2 --shift_fraction=0.2 --debug=1 --save_dir='./result/trained_model_test.h5' --is_training=0 --weights='/trained_model.h5' --lr=0.001

# Capsnet_related
Resources of Capsule Network

## Papers & Thesis
- Optimizing Neural Networks that Generate Images, 2014 [(pdf)](http://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf)
- Dynamic Routing Between Capsules, 2017 [(pdf)](https://arxiv.org/abs/1710.09829)
- MATRIX CAPSULES WITH EM ROUTING, 2018 [(pdf)](https://openreview.net/pdf?id=HJWLfGWRb)

## Educational Resources
- Capsule Networks: An Improvement to Convolutional Networks [(video)](https://www.youtube.com/watch?v=VKoLGnq15RM)
- Geoffrey Hinton Capsule theory [(video)](https://www.youtube.com/watch?v=6S1_WqE55UQ)
- What is a CapsNet or Capsule Network? [(blog)](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc)
- Capsule Networks (CapsNets) – Tutorial [(video)](https://www.youtube.com/watch?v=pPN8d0E3900)

## Codes & Implementations
- unsupervised capsule network [(link)](https://github.com/mrkulk/Unsupervised-Capsule-Network)
- capsule network : [(pytorch)](https://github.com/gram-ai/capsule-networks) [(tensorflow)](https://github.com/naturomics/CapsNet-Tensorflow) [(keras)](https://github.com/XifengGuo/CapsNet-Keras) [(mxnet)](https://github.com/Soonhwan-Kwon/capsnet.mxnet) [(matlab)](https://github.com/yechengxi/LightCapsNet)
- matrix capsules with em capsules : [(tensorflow)](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow)
- adversarial attack to capsnet [(link)](https://github.com/jaesik817/adv_attack_capsnet)
- capsnet for fashion mnist [(link)](https://github.com/XifengGuo/CapsNet-Fashion-MNIST)
- capsnet for traffic sign [(link)](https://github.com/thibo73800/capsnet-traffic-sign-classifier)
- capsnet with jupyter notebook [(link)](https://github.com/rrqq/CapsNet-tensorflow-jupyter)

