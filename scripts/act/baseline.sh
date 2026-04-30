seed=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
demo_path="$SCRIPT_DIR/../../datasets/atec_task_e/trajectory_filtered.hdf5"

echo "$demo_path"
total_iters=100000
batch_size=256
log_freq=1000
save_freq=10000
wandb_entity=""


# RGB based
for demos in 100; do
  python train_task_e.py \
    --demo_path $demo_path \
    --num_demos $demos \
    --include_rgb \
    --total_iters $total_iters \
    --batch_size $batch_size \
    --log_freq $log_freq \
    --save_freq $save_freq \
    --seed $seed \
    --exp_name act-task-e-rgb-${demos}demos-seed${seed} \
    --wandb_entity $wandb_entity \
    --track
done