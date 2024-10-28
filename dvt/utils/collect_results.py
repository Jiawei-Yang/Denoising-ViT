import json
import os


def collect_results(eval_dir, tasks=["voc_seg", "ade", "nyu"], criteria="best"):
    table = {}

    # Iterate over tasks (voc_seg, ade, nyu)
    for task in tasks:
        best_log_entry = None

        if criteria == "best":
            if task in ["voc_seg", "ade"]:
                key = "mIoU"
                best_metric = 0
            elif task == "nyu":
                key = "a1"
                best_metric = 0

        task_folder = os.path.join(eval_dir, task)
        if not os.path.exists(task_folder):
            table[task] = None
            continue

        # Get all json files in the directory
        json_files = [f for f in os.listdir(task_folder) if f.endswith(".json")]
        if len(json_files) == 0:
            table[task] = None
            continue

        # Sort and use the latest file
        json_files.sort()
        json_file_path = os.path.join(task_folder, json_files[-1])

        with open(json_file_path, "r") as file:
            for line in file:
                try:
                    log_entry = json.loads(line)
                    if "mode" in log_entry and log_entry["mode"] == "val":
                        metric = log_entry.get(key, None)

                        if metric is None:
                            continue

                        # Compare the metric and update the best result
                        if task in ["voc_seg", "ade"] and metric > best_metric:
                            best_metric = metric
                            best_log_entry = log_entry
                        elif task == "nyu" and metric > best_metric:
                            best_metric = metric
                            best_log_entry = log_entry
                except json.JSONDecodeError:
                    continue

        table[task] = best_log_entry if best_log_entry else None

    return table


def safe_get_value(entry, key, default=0):
    """Helper function to safely retrieve a value from a dictionary, return 0 if None."""
    return entry.get(key, default) if entry else default


if __name__ == "__main__":
    table = {}
    # eval_dir = "work_dirs_eval/0828_baselines"
    # eval_dir = "work_dirs_eval/0829_denoiser"
    # eval_dir = "work_dirs_eval/0720_denoiser"
    # eval_dir = "work_dirs_eval/0828_denoiser"
    # eval_dir = "work_dirs_eval/0906_denoiser_2"
    # eval_dir = "work_dirs_eval/0906_baselines"
    # eval_dir = "work_dirs_eval/0907_denoiser"
    # eval_dir = "work_dirs_eval/1013_denoiser_all_models"
    eval_dir = "work_dirs_eval/1014_baselines"
    # eval_dir = "/mnt/yonglong_nfs_4/jiawei/project/DVT-eccv-update/work_dirs_eval/0828_baselines"
    # eval_dir = "/mnt/yonglong_nfs_4/jiawei/project/DVT-eccv-update/to_compress/work_dirs_eval/0906_baselines"
    # eval_dir = "/mnt/yonglong_nfs_4/jiawei/project/DVT-eccv-update/to_compress/work_dirs_eval/0907_denoiser"
    eval_dir = "work_dirs_eval/1014_denoiser"
    # eval_dir = "work_dirs_eval/1013_denoiser"
    eval_dir = "work_dirs_eval/1027_distillation_ablation"
    # eval_dir = "work_dirs_eval/1014_denoiser"
    # eval_dir = "work_dirs_eval/1027_distillation2"
    eval_dir = "work_dirs_eval/1027_distillation_fp32"

    # Iterate through all models
    for model in os.listdir(eval_dir):
        model_result = collect_results(f"{eval_dir}/{model}")
        table[model] = model_result

    # Print header
    print(
        "Model".ljust(60),
        "VOC_mIoU".ljust(10),
        "VOC_mAcc".ljust(10),
        "ADE_mIoU".ljust(10),
        "ADE_mAcc".ljust(10),
        "NYU_RMSE".ljust(10),
        "NYU_abs_rel".ljust(10),
        "NYU_a1".ljust(10),
    )

    # Print results for each model
    for model, result in table.items():
        if result is not None:
            voc_miou = safe_get_value(result["voc_seg"], "mIoU") * 100
            voc_macc = safe_get_value(result["voc_seg"], "mAcc") * 100
            ade_miou = safe_get_value(result["ade"], "mIoU") * 100
            ade_macc = safe_get_value(result["ade"], "mAcc") * 100
            nyu_rmse = safe_get_value(result["nyu"], "rmse")
            nyu_abs_rel = safe_get_value(result["nyu"], "abs_rel")
            nyu_a1 = safe_get_value(result["nyu"], "a1") * 100

            print(
                model.ljust(60),
                f"{voc_miou:.2f}".ljust(10),
                f"{voc_macc:.2f}".ljust(10),
                f"{ade_miou:.2f}".ljust(10),
                f"{ade_macc:.2f}".ljust(10),
                f"{nyu_rmse:.4f}".ljust(10),
                f"{nyu_abs_rel:.4f}".ljust(10),
                f"{nyu_a1:.2f}%".ljust(10),
            )
        else:
            # Handle the case where result is None (missing data)
            print(
                model.ljust(60),
                "-1".ljust(10),
                "-1".ljust(10),
                "-1".ljust(10),
                "-1".ljust(10),
                "-1".ljust(10),
                "-1".ljust(10),
                "-1".ljust(10),
            )
