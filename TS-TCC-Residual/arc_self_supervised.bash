nohup python3.7 main.py --experiment_description arc_l1_self_supervised_v1 --run_description run_1 --seed 0 --training_mode self_supervised --selected_dataset arc_l1 --logs_save_dir logs_arc_l1 --device cuda:0 --home_path ./ > main_out/main.py_arc_l1_self_supervised_1.out_1 2>&1 &

nohup python3.7 preprocess_arc_l1.py ./ > preprocess_arc_l1.py_out_1 2>&1 &

nohup python3.7 preprocess_arc_l1_1.py ./ > preprocess_arc_l1_1.py_out_1 2>&1 &

nohup python3.7 main.py --experiment_description arc_l1_1_self_supervised_v1 --run_description run_1 --seed 0 --training_mode self_supervised --selected_dataset arc_l1_1 --logs_save_dir logs_arc_l1_1 --device cuda:0 --home_path ./ > main_out/main.py_arc_l1_1_self_supervised_1.out_1 2>&1 &

nohup python3.7 preprocess_arc_l2.py ./ > preprocess_arc_l2.py_out_1 2>&1 &

nohup python3.7 main.py --experiment_description arc_l2_self_supervised_v1 --run_description run_1 --seed 0 --training_mode self_supervised --selected_dataset arc_l2 --logs_save_dir logs_arc_l2 --device cuda:0 --home_path ./ > main_out/main.py_arc_l2_self_supervised_1.out_1 2>&1 &