"""
Source: https://gitlab.tudelft.nl/attilalengyel/hpc-imagenet/
Used with permission from Attila Lengyel.
"""
from subprocess import call
import os.path

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.tfrecord as tfrec
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


class TFRecordPipeline(Pipeline):
    def __init__(
        self,
        tfrecord_path,
        tfrecord_idx_path,
        device_id,
        num_gpus,
        batch_size=64,
        num_threads=2,
        dali_cpu=False,
        augment=False,
        crop=224,
        size=256,
    ):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id)

        self.augment = augment

        self.input = ops.TFRecordReader(
            path=tfrecord_path,
            index_path=tfrecord_idx_path,
            features={
                "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
            },
            num_shards=num_gpus,
            shard_id=device_id,
        )

        # Specify devices to use
        dali_device = "cpu" if dali_cpu else "gpu"
        decoder_device = "cpu" if dali_cpu else "mixed"
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet without additional reallocations.
        # Not sure if this is needed for TFRecords though.
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0

        # Decoder and data augmentation
        if augment == "standard":
            # To use for training
            self.decode = ops.ImageDecoderRandomCrop(
                device=decoder_device,
                output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100,
            )
            self.resize = ops.Resize(
                device=dali_device,
                resize_x=crop,
                resize_y=crop,
                interp_type=types.INTERP_TRIANGULAR,
            )
            self.coin = ops.CoinFlip(probability=0.5)
        elif augment == "none":
            # To use for validation
            self.decode = ops.ImageDecoder(
                device=decoder_device,
                output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
            )
            self.resize = ops.Resize(
                device=dali_device,
                resize_shorter=size,
                interp_type=types.INTERP_TRIANGULAR,
            )
        else:
            raise NotImplementedError(f"augment = {augment}")

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        inputs = self.input()

        images = inputs["image/encoded"]
        images = self.decode(images)
        images = self.resize(images)
        if self.augment == "standard":
            rng = self.coin()
            images = self.cmnp(images.gpu(), mirror=rng)
        elif self.augment == "none":
            images = self.cmnp(images.gpu())
        else:
            raise NotImplementedError(f"augment = {self.augment}")

        labels = inputs["image/class/label"] - 1
        return [images, labels]


def imagenet_tfrecord(
    root,
    split,
    batch_size,
    num_threads,
    device_id,
    num_gpus,
    dali_cpu=False,
    augment="none",
):
    """
    PyTorch dataloader for ImageNet TFRecord files.

    Args:
        root (str): Location of the 'tfrecords' ImageNet directory.
        split (str): Split to use, either 'train' or 'val'.
        batch_size (int): Batch size per GPU (default=64).
        num_threads (int): Number of dataloader workers to use per sub-process.
        device_id (int): ID of the GPU corresponding to the current subprocess. Dataset
            will be divided over all subprocesses.
        num_gpus (int): Total number of GPUS available.
        dali_cpu (bool): Set True to perform part of data loading on CPU instead of GPU (default=False).
        augment (bool): Whether or not to apply data augmentation (random cropping,
            horizontal flips).

    Returns:
        PyTorch dataloader.

    """

    # List all tfrecord files in directory
    tf_files = os.listdir(os.path.join(root, split, "data"))

    # Create dir for idx files if not exists
    idx_files_dir = os.path.join(root, split, "idx_files")
    if not os.path.exists(idx_files_dir):
        os.mkdir(idx_files_dir)

    tfrec_path_list = []
    idx_path_list = []
    n_samples = 0
    # Create idx files and create TFRecordPipelines
    for tf_file in tf_files:
        # Path of tf_file and idx file
        tfrec_path = os.path.join(root, split, "data", tf_file)
        tfrec_path_list.append(tfrec_path)
        idx_path = os.path.join(idx_files_dir, tf_file + "_idx")
        idx_path_list.append(idx_path)
        # Create idx file for tf_file by calling tfrecord2idx script
        if not os.path.isfile(idx_path):
            call(["tfrecord2idx", tfrec_path, idx_path])
        with open(idx_path, "r") as f:
            n_samples += len(f.readlines())
    # Create TFRecordPipeline for each TFRecord file
    pipe = TFRecordPipeline(
        tfrecord_path=tfrec_path_list,
        tfrecord_idx_path=idx_path_list,
        device_id=device_id,
        num_gpus=num_gpus,
        batch_size=batch_size,
        num_threads=num_threads,
        augment=augment,
        dali_cpu=dali_cpu,
    )
    pipe.build()

    dataloader = DALIClassificationIterator(
        pipelines=pipe, fill_last_batch=False, size=(n_samples // num_gpus + 1)
    )
    return dataloader
