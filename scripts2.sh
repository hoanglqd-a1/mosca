# python mosca_precompute.py --ws /datasets/nerfds_4/as_2/ --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_4/basin/ --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_4/bell_2/ --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_4/cup_2/ --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_4/plate/ --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_4/press/ --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_4/sieve/ --cfg ./profile/nerfds/nerfds_prep.yaml --mask

# python mosca_reconstruct.py --ws /datasets/nerfds_4/as_2 --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_4/cup_2 --cfg ./profile/nerfds/nerfds_fit.yaml
# python mosca_reconstruct.py --ws /datasets/nerfds_4/sieve --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_4/press --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_4/basin --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_4/plate --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_4/bell_2 --cfg ./profile/nerfds/nerfds_fit.yaml