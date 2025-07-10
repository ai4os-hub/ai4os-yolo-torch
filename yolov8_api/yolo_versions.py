import os
import yaml
import importlib.util
def get_task_type_from_model_name(model_name):
    if '-seg' in model_name:
        return 'seg'
    elif '-cls' in model_name:
        return 'cls'
    elif '-obb' in model_name:
        return 'obb'
    else:
        return 'det'  # default is detection

ALLOWED_TASKS = {'-cls', '-seg', '-obb'}
TASK_ORDER = {'det': 0, 'cls': 1, 'obb': 2, 'seg': 3}
def get_all_local_yolo_versions_and_variants():
    spec = importlib.util.find_spec("ultralytics")
    base_path = os.path.dirname(spec.origin)
    models_dir = os.path.join(base_path, "cfg", "models")
    yolo_versions = {}

    if not os.path.isdir(models_dir):
        return {}

    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        variants = []
        for file in os.listdir(folder_path):
            if file.endswith('.yaml') and 'yolo' in file.lower():
                # Extract numeric part for sorting
                if folder.startswith('v'):
                    try:
                        num_version = int(folder.lstrip('v'))
                    except ValueError:
                        continue
                else:
                    try:
                        num_version = int(folder)
                    except ValueError:
                        continue
                if num_version >= 8:
                    file_path = os.path.join(folder_path, file)
                    try:
                        with open(file_path, 'r') as f:
                            cfg = yaml.safe_load(f)
                        if 'scales' in cfg and len(cfg['scales']) > 1:
                            base_name = file.replace('.yaml', '')
                            if '-' in base_name:
                                model_part, task_part = base_name.rsplit('-', 1)
                                task_suffix = f'-{task_part}'
                            else:
                                model_part = base_name
                                task_suffix = ''
                            if not task_suffix or task_suffix in ALLOWED_TASKS:
                                variants.extend(
                                    [f"{model_part}{scale}{task_suffix}"
                                     for scale in cfg['scales'].keys()]
                                )
                        else:
                            variants.append(file[:-5] if file.endswith('.yaml') else file)
                    except Exception:
                        variants.append(file[:-5] if file.endswith('.yaml') else file)
        if variants:
            yolo_versions[folder] = variants

    # Sort by numeric version, ascending
    def version_key(v):
        try:
            return int(v.lstrip('v'))
        except ValueError:
            return float('inf')

    sorted_yolo_versions = dict(sorted(yolo_versions.items(), key=lambda item: version_key(item[0])))
    for version in sorted_yolo_versions:
        sorted_yolo_versions[version] = sorted(
            sorted_yolo_versions[version],
            key=lambda x: (TASK_ORDER[get_task_type_from_model_name(x)], x)
        )

    return sorted_yolo_versions
if __name__ == "__main__":
    yolo_versions = get_all_local_yolo_versions_and_variants()
    for version, variants in yolo_versions.items():
        print(f"Version: {version}")
        for variant in variants:
            print(f"  - {variant}")
    print("Total YOLO versions and variants found:", len(yolo_versions))
