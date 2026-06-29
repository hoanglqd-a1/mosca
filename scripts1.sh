python mosca_precompute.py --ws /datasets/nerfds_3/as --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/basin --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/bell --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/cup --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/plate --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/press --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/sieve --cfg ./profile/nerfds/nerfds_prep.yaml --mask

# python mosca_reconstruct.py --ws /datasets/nerfds_3/as --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_3/cup --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_3/sieve --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_3/bell --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_3/basin --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_3/plate --cfg ./profile/nerfds/nerfds_fit.yaml
python mosca_reconstruct.py --ws /datasets/nerfds_3/press --cfg ./profile/nerfds/nerfds_fit.yaml
