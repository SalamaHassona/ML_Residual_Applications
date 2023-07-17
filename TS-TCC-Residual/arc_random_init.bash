

mkdir logs_arc_random_init_01_l1
mkdir logs_arc_random_init_05_l1

mkdir logs_arc_random_init_01_l1_1
mkdir logs_arc_random_init_05_l1_1

mkdir logs_arc_random_init_01_l2
mkdir logs_arc_random_init_05_l2




nohup python3.7 main.py --experiment_description arc_l1_random_init_v1 --run_description run_1 --seed 0 --training_mode random_init --selected_dataset arc_fine_train_super_random_01_l1 --logs_save_dir logs_arc_random_init_01_l1 --device cuda:0 --home_path ./ > main_out/main.py_arc_random_init_01_l1.out_1 2>&1 &
nohup python3.7 main.py --experiment_description arc_l1_random_init_v1 --run_description run_1 --seed 0 --training_mode random_init --selected_dataset arc_fine_train_super_random_05_l1 --logs_save_dir logs_arc_random_init_05_l1 --device cuda:1 --home_path ./ > main_out/main.py_arc_random_init_05_l1.out_1 2>&1 &



nohup python3.7 main.py --experiment_description arc_l1_1_random_init_v1 --run_description run_1 --seed 0 --training_mode random_init --selected_dataset arc_fine_train_super_random_01_l1_1 --logs_save_dir logs_arc_random_init_01_l1_1 --device cuda:1 --home_path ./ > main_out/main.py_arc_random_init_01_l1_1.out_1 2>&1 &
nohup python3.7 main.py --experiment_description arc_l1_1_random_init_v1 --run_description run_1 --seed 0 --training_mode random_init --selected_dataset arc_fine_train_super_random_05_l1_1 --logs_save_dir logs_arc_random_init_05_l1_1 --device cuda:0 --home_path ./ > main_out/main.py_arc_random_init_05_l1_1.out_1 2>&1 &



nohup python3.7 main.py --experiment_description arc_l2_random_init_v1 --run_description run_1 --seed 0 --training_mode random_init --selected_dataset arc_fine_train_super_random_01_l2 --logs_save_dir logs_arc_random_init_01_l2 --device cuda:0 --home_path ./ > main_out/main.py_arc_random_init_01_l2.out_1 2>&1 &
nohup python3.7 main.py --experiment_description arc_l2_random_init_v1 --run_description run_1 --seed 0 --training_mode random_init --selected_dataset arc_fine_train_super_random_05_l2 --logs_save_dir logs_arc_random_init_05_l2 --device cuda:1 --home_path ./ > main_out/main.py_arc_random_init_05_l2.out_1 2>&1 &
