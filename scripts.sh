# python processing_utils/nerfds_to_iphone_format.py --input_dir /datasets/nerfds/as_novel_view --output_dir /datasets/nerfds_3/as --scale_ratio 2
# python processing_utils/nerfds_to_iphone_format.py --input_dir /datasets/nerfds/basin_novel_view --output_dir /datasets/nerfds_3/basin --scale_ratio 2
# python processing_utils/nerfds_to_iphone_format.py --input_dir /datasets/nerfds/bell_novel_view --output_dir /datasets/nerfds_3/bell --scale_ratio 2
# python processing_utils/nerfds_to_iphone_format.py --input_dir /datasets/nerfds/cup_novel_view --output_dir /datasets/nerfds_3/cup --scale_ratio 2
# python processing_utils/nerfds_to_iphone_format.py --input_dir /datasets/nerfds/plate_novel_view --output_dir /datasets/nerfds_3/plate --scale_ratio 2
# python processing_utils/nerfds_to_iphone_format.py --input_dir /datasets/nerfds/press_novel_view --output_dir /datasets/nerfds_3/press --scale_ratio 2
# python processing_utils/nerfds_to_iphone_format.py --input_dir /datasets/nerfds/sieve_novel_view --output_dir /datasets/nerfds_3/sieve --scale_ratio 2

# python mosca_precompute.py --ws /datasets/nerfds_3/as --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_3/basin --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_3/bell --cfg ./profile/nerfds/nerfds_prep.yaml --mask
# python mosca_precompute.py --ws /datasets/nerfds_3/cup --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/plate --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/press --cfg ./profile/nerfds/nerfds_prep.yaml --mask
python mosca_precompute.py --ws /datasets/nerfds_3/sieve --cfg ./profile/nerfds/nerfds_prep.yaml --mask
