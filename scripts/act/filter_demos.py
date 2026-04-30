"""Filter HDF5 trajectory data: remove timesteps where the robot is not moving.

Uses consecutive action differences: steps where max|action[t] - action[t-1]| < threshold
are considered stationary and removed.

Usage:
    python scripts/act/filter_demos.py \
        --input datasets/atec_task_e/trajectory.hdf5 \
        --output datasets/atec_task_e/trajectory_filtered.hdf5 \
        --threshold 0.001
"""

import argparse
import h5py
import numpy as np


def compute_moving_mask(actions, threshold):
    """Return a boolean mask (length T) marking steps where the robot is moving.

    A step is 'moving' if max|action[t] - action[t-1]| >= threshold.
    """
    delta = np.zeros_like(actions)
    delta[1:] = np.abs(actions[1:] - actions[:-1])
    return delta.max(axis=1) >= threshold


def main():
    parser = argparse.ArgumentParser(description="Filter stationary steps from demo HDF5.")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 path")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Steps with max|action_delta| < threshold are removed (default: 0.001)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print statistics, don't write output")
    args = parser.parse_args()

    total_before = 0
    total_after  = 0

    with h5py.File(args.input, "r") as fin:
        traj_keys = sorted(fin.keys(), key=lambda k: int(k.split("_")[1]))

        if args.dry_run:
            print(f"{'traj':>10s}  {'before':>7s}  {'after':>7s}  {'removed':>7s}  {'removed%':>8s}")
            for key in traj_keys:
                actions = fin[key]["actions"][:]
                mask = compute_moving_mask(actions, args.threshold)
                n_before = len(actions)
                n_after  = mask.sum()
                total_before += n_before
                total_after  += n_after
                print(f"{key:>10s}  {n_before:7d}  {n_after:7d}  "
                      f"{n_before - n_after:7d}  {(1 - n_after/n_before)*100:7.1f}%")
            print(f"\nTotal: {total_before} → {total_after} "
                  f"(removed {total_before - total_after}, "
                  f"{(1 - total_after/total_before)*100:.1f}%)")
            return

        with h5py.File(args.output, "w") as fout:
            for key in traj_keys:
                grp_in = fin[key]
                actions = grp_in["actions"][:]
                mask = compute_moving_mask(actions, args.threshold)

                n_before = len(actions)
                n_after  = mask.sum()
                total_before += n_before
                total_after  += n_after

                grp_out = fout.create_group(key)

                # Filter all (T, ...) datasets with the same mask
                for ds_name in grp_in.keys():
                    if ds_name == "images":
                        # Handle nested image group
                        img_grp = grp_out.create_group("images")
                        for img_key in grp_in["images"].keys():
                            data = grp_in[f"images/{img_key}"][:]
                            img_grp.create_dataset(img_key, data=data[mask], compression="gzip")
                    elif isinstance(grp_in[ds_name], h5py.Dataset):
                        data = grp_in[ds_name][:]
                        if data.shape[0] == n_before:
                            grp_out.create_dataset(ds_name, data=data[mask], compression="gzip")
                        else:
                            # Non-temporal dataset, copy as-is
                            grp_out.create_dataset(ds_name, data=data, compression="gzip")

                print(f"  {key}: {n_before} → {n_after} steps "
                      f"(removed {n_before - n_after})")

    print(f"\nTotal: {total_before} → {total_after} "
          f"(removed {total_before - total_after}, "
          f"{(1 - total_after/total_before)*100:.1f}%)")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
