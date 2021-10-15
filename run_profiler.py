# general
import os
import tqdm
from io import StringIO

# project
from run_experiment import setup, model_and_datasets

# torch
import torch
from torch.profiler import profile, ProfilerActivity

# Loggers and config
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(
    cfg: OmegaConf,
):
    setup(cfg)
    model, dataloaders = model_and_datasets(cfg)

    print(cfg)

    n_batches = cfg.profile.n_batches

    if cfg.profile.data_mode == "dataset":
        all_data = dataloaders["train"]
    elif cfg.profile.data_mode == "fake":
        all_data = [
            (
                torch.rand(
                    (
                        cfg.train.batch_size,
                        3,
                        cfg.profile.data_shape,
                        cfg.profile.data_shape,
                    )
                ),
                None,
            )
            for _ in range(n_batches)
        ]

    train = True
    with torch.set_grad_enabled(train):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profile.directory
            ),
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=n_batches - 5),
        ) as prof:
            batch = 0
            for data in tqdm.tqdm(all_data, total=n_batches):
                if cfg.dataset == "Imagenet":
                    data = (data[0]["data"], data[0]["label"].squeeze(1))
                inputs, _ = data
                inputs = inputs.to(cfg.device)

                inputs = torch.dropout(inputs, cfg.net.dropout_in, train)
                _ = model(inputs)

                prof.step()

                batch += 1
                if batch >= n_batches:
                    break

    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=100)
    print(table)

    # Parse table to save as CSV
    f = StringIO(table)
    data = []

    # Use the dashes on the first line to know the section indices
    sections = f.readline().split()

    def parse_line(line):
        line_data = []
        i = 0
        for section in sections:
            start = i
            end = i + len(section)
            line_data.append(line[start:end].strip().replace(",", ";"))

            # Skip two spaces after each section
            i += len(section) + 2
        return line_data

    # Parse rest of header
    data.append(parse_line(f.readline()))
    _ = f.readline()

    # Parse body
    for line in f:
        # Stop when we reach the end of the table
        if "-----" in line.strip():
            break
        data.append(parse_line(line))

    # Write to timings file
    with open(os.path.join(cfg.profile.directory, "timings.csv"), "w") as fw:
        for line in data:
            fw.write(",".join(line) + "\n")


if __name__ == "__main__":
    main()
