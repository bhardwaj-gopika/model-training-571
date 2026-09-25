NEW_DIR=/sdf/data/ad/ard-online/FACET-II_Training_Data/2026-09-18
ONE_FILE=$(ls $NEW_DIR/*_571.h5 | head -1)
python inspect_standard_h5.py "$ONE_FILE" --max-batches 1 | grep -A2 -i "bin_size\|VCC"
python build_dataset_from_standard.py "$ONE_FILE" --output /tmp/probe.csv --normalize --min-alive-frac 0.9 --probe

NEW_DIR=/sdf/data/ad/ard-online/FACET-II_Training_Data/2026-09-18
ls $NEW_DIR/*_571.h5 > file_list_571.txt
wc -l file_list_571.txt
N_FILES=$(wc -l < file_list_571.txt)
OUTPUT_DIR=shards-broad SHOTS_PER_BATCH=10 sbatch --array=0-$((N_FILES-1))%8 make_dataset.sh

bash concat_shards.sh shards-broad dataset-broad.csv

python split_dataset.py dataset-broad.csv --seed 42

sbatch gpu.sh

python analyze_covariance2.py --model-dir model-output-571-broad --test dataset-broad-test.csv --output-dir analysis-broad --dropout 0

python infer_covariance2.py --model-dir model-output-571-broad --input-csv dataset-broad-test.csv --output-dir inference-output-broad --dropout 0

pip install --force-reinstall git+https://github.com/slaclab/facet2_inj_ml_model_571.git


python end_to_end_demo_haai.py --dataset-csv dataset-broad.csv --provenance-csv dataset-broad.provenance.csv --haai-dir /sdf/data/ad/ard-online/FACET-II_Training_Data/2026-09-18 --num-samples 5 --output-dir demo-output-broad
