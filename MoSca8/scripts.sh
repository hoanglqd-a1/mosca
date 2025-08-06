# python mosca_precompute.py --ws /datasets/nvidia/Balloon1 --cfg ./profile/nvidia/nvidia_prep.yaml
# python mosca_precompute.py --ws /datasets/nvidia/Balloon2 --cfg ./profile/nvidia/nvidia_prep.yaml
# python mosca_precompute.py --ws /datasets/nvidia/Umbrella --cfg ./profile/nvidia/nvidia_prep.yaml

# python mosca_reconstruct.py --ws /datasets/nvidia/Balloon1 --cfg ./profile/nvidia/nvidia_fit_colfree1.yaml
# python mosca_reconstruct.py --ws /datasets/nvidia/Balloon2 --cfg ./profile/nvidia/nvidia_fit_colfree1.yaml
# python mosca_reconstruct.py --ws /datasets/nvidia/Jumping  --cfg ./profile/nvidia/nvidia_fit_colfree1.yaml
# python mosca_reconstruct.py --ws /datasets/nvidia/Playground  --cfg ./profile/nvidia/nvidia_fit_colfree1.yaml
# python mosca_reconstruct.py --ws /datasets/nvidia/Skating --cfg ./profile/nvidia/nvidia_fit_colfree1.yaml
# python mosca_reconstruct.py --ws /datasets/nvidia/Umbrella --cfg ./profile/nvidia/nvidia_fit_colfree1.yaml

# python mosca_precompute.py --ws /datasets/iphone/block --cfg ./profile/iphone/iphone_prep.yaml
# python mosca_precompute.py --ws /datasets/iphone/teddy --cfg ./profile/iphone/iphone_prep.yaml

# python mosca_reconstruct.py --ws /datasets/iphone/apple --cfg ./profile/iphone/iphone_fit_colfree.yaml
python mosca_reconstruct.py --ws /datasets/iphone/spin --cfg ./profile/iphone/iphone_fit_colfree.yaml
python mosca_reconstruct.py --ws /datasets/iphone/space-out --cfg ./profile/iphone/iphone_fit_colfree.yaml
python mosca_reconstruct.py --ws /datasets/iphone/block --cfg ./profile/iphone/iphone_fit_colfree.yaml
python mosca_reconstruct.py --ws /datasets/iphone/teddy --cfg ./profile/iphone/iphone_fit_colfree.yaml
python mosca_reconstruct.py --ws /datasets/iphone/wheel --cfg ./profile/iphone/iphone_fit_colfree.yaml