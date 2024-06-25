python -m venv .venv
./.venv/Sctipt/activate
pip install -r requirements.txt
cd repos/Palette
python .\eval_v2.py -s ..\..\data\processed\BP_npy\0123_08_align_norm3_256\p00\scale_test.npy -d .\experiments\train_BPv8_complete\results\test\19