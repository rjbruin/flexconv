"""
Utility to pre-compute a cache of any dataset supporting the caching feature.
"""
import argparse

from datasets import ImagenetDownsampled


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--data_root', type=str, required=True)
    args = p.parse_args()

    for partition in ["train", "val"]:
        dataset = ImagenetDownsampled(partition, args.data_root, use_ram=True, use_cache=False)

        if not hasattr(dataset, 'save') or not hasattr(dataset, 'load'):
            raise ValueError(f"Dataset {args.dataset} does not support caching.")

        dataset.save()
