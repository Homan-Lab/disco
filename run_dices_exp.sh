nvidia-smi
echo "GPU ID?"
read gpu_id

echo "Running Pre Processing of Dices Dataset for Disco"

dataset_name=dices_dataset_990

python3 preprocess_dices.py

 for split_name in train dev test
 do
     python gen_disco_dataset.py --inp_dir=./datasets/"$dataset_name"/processed/disco/ --out_dir=./experimental_data/"$dataset_name"/ --annotator_item_fname="$dataset_name"_"$split_name"_AIL.csv --item_lab_fname="$dataset_name"_"$split_name"_IL.csv --annotator_lab_fname="$dataset_name"_"$split_name"_AL.csv --embeddings=Xi_"$split_name".npy --split_name="$split_name"
 done

echo "Training DisCo Models"
python3 train_disco_sweep.py --config ./config_files/"$dataset_name".cfg --sweep_id rit_pl/dices_test/0y3ggr29 --gpu_id $gpu_id --run_count 1

echo "Identifying best model"
python3 predict_disco.py --config ./config_files/"$dataset_name".cfg --sweep_id rit_pl/dices_test/0y3ggr29 --gpu_id $gpu_id

echo "Evaluating Disco model and Predictions"
python3 eval_model.py --data_dir=./experimental_data/"$dataset_name"/ --model_fname=./experimental_data/"$dataset_name"/trained_model.disco --split_name=test --dataset_name="$dataset_name" --wandb_name=dices_test --empirical_fname=./datasets/"$dataset_name"/processed/disco/"$dataset_name"_test_AIL_data.csv --gpu_id=$gpu_id