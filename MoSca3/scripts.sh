python mosca_precompute.py --ws /datasets/iphone/spin --cfg ./profile/iphone/iphone_prep.yaml 
python mosca_reconstruct.py --ws /datasets/iphone/spin --cfg ./profile/iphone/iphone_fit_colfree.yaml
python mosca_precompute.py --ws /datasets/iphone/wheel --cfg ./profile/iphone/iphone_prep.yaml
python mosca_reconstruct.py --ws /datasets/iphone/wheel --cfg ./profile/iphone/iphone_fit_colfree.yaml
