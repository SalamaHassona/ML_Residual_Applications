

cp -r logs_arc_random_init_01_l1 logs_arc_eval_random_init_01_l1
cp -r logs_arc_random_init_05_l1 logs_arc_eval_random_init_05_l1

cp -r logs_arc_random_init_01_l1_1 logs_arc_eval_random_init_01_l1_1
cp -r logs_arc_random_init_05_l1_1 logs_arc_eval_random_init_05_l1_1

cp -r logs_arc_random_init_01_l2 logs_arc_eval_random_init_01_l2
cp -r logs_arc_random_init_05_l2 logs_arc_eval_random_init_05_l2




nohup python3.7 main_eval_rand.py --experiment_description arc_l1_random_init_v1 --run_description run_1 --seed 0 --training_mode eval --selected_dataset arc_l1_eval --logs_save_dir logs_arc_eval_random_init_01_l1 --device cuda:0 --home_path ./ > main_out/main_eval_rand.py_arc_eval_random_init_01_l1.out_1 2>&1 &
nohup python3.7 main_eval_rand.py --experiment_description arc_l1_random_init_v1 --run_description run_1 --seed 0 --training_mode eval --selected_dataset arc_l1_eval --logs_save_dir logs_arc_eval_random_init_05_l1 --device cuda:1 --home_path ./ > main_out/main_eval_rand.py_arc_eval_random_init_05_l1.out_1 2>&1 &



nohup python3.7 main_eval_rand.py --experiment_description arc_l1_1_random_init_v1 --run_description run_1 --seed 0 --training_mode eval --selected_dataset arc_l1_1_eval --logs_save_dir logs_arc_eval_random_init_01_l1_1 --device cuda:1 --home_path ./ > main_out/main_eval_rand.py_arc_eval_random_init_01_l1_1.out_1 2>&1 &
nohup python3.7 main_eval_rand.py --experiment_description arc_l1_1_random_init_v1 --run_description run_1 --seed 0 --training_mode eval --selected_dataset arc_l1_1_eval --logs_save_dir logs_arc_eval_random_init_05_l1_1 --device cuda:0 --home_path ./ > main_out/main_eval_rand.py_arc_eval_random_init_05_l1_1.out_1 2>&1 &



nohup python3.7 main_eval_rand.py --experiment_description arc_l2_random_init_v1 --run_description run_1 --seed 0 --training_mode eval --selected_dataset arc_l2_eval --logs_save_dir logs_arc_eval_random_init_01_l2 --device cuda:0 --home_path ./ > main_out/main_eval_rand.py_arc_eval_random_init_01_l2.out_1 2>&1 &
nohup python3.7 main_eval_rand.py --experiment_description arc_l2_random_init_v1 --run_description run_1 --seed 0 --training_mode eval --selected_dataset arc_l2_eval --logs_save_dir logs_arc_eval_random_init_05_l2 --device cuda:1 --home_path ./ > main_out/main_eval_rand.py_arc_eval_random_init_05_l2.out_1 2>&1 &
