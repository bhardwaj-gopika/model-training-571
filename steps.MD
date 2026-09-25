
  # 1. Point at the new data directory. The default DATA_DIR in make_dataset.sh is
  #    the old scraped_data path, but that's not actually referenced — it uses
  #    file_list_571.txt directly. So just regenerate the list:
  NEW_DIR=/sdf/data/ad/ard-online/FACET-II_Training_Data/2026-09-18
  ls $NEW_DIR/*_571.h5 > file_list_571.txt
  wc -l file_list_571.txt
  N_FILES=$(wc -l < file_list_571.txt)
  
  # 2. Submit array job (WRAP_PHASE=1 and INPUT_FILTER=1 are defaults, keep them)
  OUTPUT_DIR=shards-broad SHOTS_PER_BATCH=100 \
      sbatch --array=0-$((N_FILES-1))%8 make_dataset.sh
  
  # 3. Monitor
  squeue -u $USER
  # When all tasks finish:
  ls shards-broad/*.csv | wc -l           # should equal N_FILES
  tail -n 3 logs/build571_*_*.out | tail -30   # spot-check no errors
  
  Then concat + validate
  
  bash concat_shards.sh shards-broad dataset-broad.csv
  wc -l dataset-broad.csv


bash concat_shards.sh shards-broad dataset-broad.csv

python split_dataset.py dataset-broad.csv --seed 42

sbatch gpu.sh

python analyze_covariance2.py --model-dir model-output-571-broad --test dataset-broad-test.csv --output-dir analysis-broad --dropout 0

python infer_covariance2.py --model-dir model-output-571-broad --input-csv dataset-broad-test.csv --output-dir inference-output-broad --dropout 0

pip install --force-reinstall git+https://github.com/slaclab/facet2_inj_ml_model_571.git


python end_to_end_demo_haai.py --dataset-csv dataset-broad.csv --provenance-csv dataset-broad.provenance.csv --haai-dir /sdf/data/ad/ard-online/FACET-II_Training_Data/2026-09-18 --num-samples 5 --output-dir demo-output-broad
