# from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
# from batchgenerators.utilities.file_and_folder_operations import join, subdirs, subfiles, maybe_mkdir_p
# from nnunetv2.paths import nnUNet_raw
#
# if __name__ == '__main__':
#     """
#     this dataset does not copy the data into nnunet format and just links to existing data. The dataset can only be
#     used from one machine because the paths in the dataset.json are hard coded
#     """
#     extracted_BraTS2024_GLI_dir = 'E:/project_coding/try_run37/nnUNet/nnUNet_raw/train'
#     nnunet_dataset_name = 'BraTS2024-BraTS-GLI'
#     nnunet_dataset_id = 226
#     dataset_name = f'Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}'
#     dataset_dir = join(nnUNet_raw, dataset_name)
#     # print(dataset_dir)
#     maybe_mkdir_p(dataset_dir)
#
#     dataset = {}
#     casenames = subdirs(extracted_BraTS2024_GLI_dir, join=False)
#     for c in casenames:
#         dataset[c] = {
#             'label': join(extracted_BraTS2024_GLI_dir, c, c + '-seg.nii.gz'),
#             'images': [
#                 join(extracted_BraTS2024_GLI_dir, c, c + '-t1n.nii.gz'),
#                 join(extracted_BraTS2024_GLI_dir, c, c + '-t1c.nii.gz'),
#                 join(extracted_BraTS2024_GLI_dir, c, c + '-t2w.nii.gz'),
#                 join(extracted_BraTS2024_GLI_dir, c, c + '-t2f.nii.gz')
#             ]
#         }
#     labels = {
#         'background': 0,
#         'NETC': 1,
#         'SNFH': 2,
#         'ET': 3,
#         'RC': 4,
#     }
#
#     generate_dataset_json(
#         dataset_dir,
#         {
#             0: 'T1',
#             1: "T1C",
#             2: "T2W",
#             3: "T2F"
#         },
#         labels,
#         num_training_cases=len(dataset),
#         file_ending='.nii.gz',
#         regions_class_order=None,
#         dataset_name=dataset_name,
#         reference='https://www.synapse.org/Synapse:syn53708249/wiki/627500',
#         license='see https://www.synapse.org/Synapse:syn53708249/wiki/627508',
#         dataset=dataset,
#         description='This dataset does not copy the data into nnunet format and just links to existing data. '
#                     'The dataset can only be used from one machine because the paths in the dataset.json are hard coded'
#     )

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import join, subdirs, maybe_mkdir_p
from nnunetv2.paths import nnUNet_raw
import shutil
import os

if __name__ == '__main__':
    """
    此脚本将 BraTS2024 数据集拆分到 nnU-Net 的 imagesTr/labelsTr 文件夹中，
    并生成 dataset.json 描述文件。
    """
    extracted_BraTS2024_GLI_dir = 'E:/Data/Brats2024/split_data0525/test'
    nnunet_dataset_name = 'BraTS2024-BraTS-GLI'
    nnunet_dataset_id = 226
    dataset_name = f'Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}'

    # 在 nnUNet_raw 下创建输出文件夹
    dataset_dir = join(nnUNet_raw, dataset_name)
    imagesTr = join(dataset_dir, 'imagesTs')
    labelsTr = join(dataset_dir, 'labelsTs')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # 扫描所有病例目录
    casenames = subdirs(extracted_BraTS2024_GLI_dir, join=False)
    for c in casenames:
        case_dir = join(extracted_BraTS2024_GLI_dir, c)

        # 拷贝或链接四个模态到 imagesTr，命名为 {case}_0000.nii.gz ... {case}_0003.nii.gz
        modalities = ['t1n', 't1c', 't2w', 't2f']
        for i, suffix in enumerate(modalities):
            src = join(case_dir, f"{c}-{suffix}.nii.gz")
            dst = join(imagesTr, f"{c}_{i:04d}.nii.gz")
            # 若想软链接可改为 os.symlink(src, dst)
            shutil.copy(src, dst)

        # 拷贝分割标签到 labelsTr，命名为 {case}.nii.gz
        shutil.copy(join(case_dir, f"{c}-seg.nii.gz"),
                    join(labelsTr, f"{c}.nii.gz"))

    # 定义模态名称和标签映射
    channel_names = {0: 'T1', 1: "T1C", 2: "T2W", 3: "T2F"}
    labels = {
        'background': 0,
        'NETC': 1,
        'SNFH': 2,
        'ET': 3,
        'RC': 4,
    }
    # labels = {
    #     'background': 0,
    #     'whole tumor': (1, 2, 3),
    #     'tumor core': (1, 3),
    #     'enhancing tumor': (3,),
    #     'rc':(4,)
    # }

    # 生成 dataset.json
    generate_dataset_json(
        dataset_dir,
        channel_names,
        labels,
        num_training_cases=len(casenames),
        file_ending='.nii.gz',
        regions_class_order=None,
        dataset_name=dataset_name,
        reference='https://www.synapse.org/Synapse:syn53708249/wiki/627500',
        license='see https://www.synapse.org/Synapse:syn53708249/wiki/627508',
        description=('This dataset does not copy the data into nnunet format '
                     'and just links to existing data. The dataset can only '
                     'be used from one machine because the paths in the '
                     'dataset.json are hard coded'),
    )

    print(f"Finished conversion for BraTS2024 → {dataset_dir}")
