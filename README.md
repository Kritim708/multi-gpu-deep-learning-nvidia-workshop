# Multi-GPU Deep Learning Project – NVIDIA Workshop

This repository contains a project created as part of the NVIDIA workshop on Data Parallelism. It demonstrates multi-GPU training of a deep neural network using PyTorch, showcasing how to efficiently distribute data across multiple GPUs to accelerate the training process.

## Project Overview

The main features and files of this project are:

- **assessment.py**  
  The core script implementing multi-GPU deep learning using PyTorch. It sets up the neural network, loads and distributes training data, and orchestrates parallel training across available GPUs utilizing data parallelism routines. The script handles model initialization, optimizer setup, training and evaluation loops, and logging results. It demonstrates how to leverage PyTorch's `DataParallel` or `DistributedDataParallel` modules for faster model training using multiple NVIDIA GPUs.

- **assessment_print.py**  
  A utility script that displays results, summaries, or outputs generated from the main training routine. Use this for quick inspection or formatted result visualization post-training.

- **data/**  
  Directory containing your dataset or input files. The neural network in `assessment.py` expects the data to be present here in a defined format (commonly images or tabular data for deep learning tasks).

- **results/**  
  This folder is used to store the outputs, trained model weights, logs, or analytics produced during the training process.

- **training_data/**  
  Supplementary training datasets or additional data files that are used for model training and validation.

- **requirements.txt**  
  Lists the Python dependencies for the project. The key requirement here is PyTorch (for deep learning and multi-GPU support), along with any other utility libraries.

- **execution_command.txt**  
  Sample command(s) to execute the main training script, including necessary arguments or flags for multi-GPU settings.

## How To Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Kritim708/multi-gpu-deep-learning-nvidia-workshop.git
    cd multi-gpu-deep-learning-nvidia-workshop
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your data:**
   - Place your dataset in the `data/` and/or `training_data/` directories as required by your experiment. Sample data is provided for you. 

4. **Start training:**
    - Check `execution_command.txt` for example commands. A typical command might look like:
    ```bash
    python assessment.py --num_gpus 2 --epochs 10 --batch_size 32
    ```
    Adjust the arguments based on your hardware and dataset.

5. **Results:**
    - After training, review outputs in the `results/` directory.
    - Use `assessment_print.py` as needed to visualize or summarize results.

## Key Concepts Demonstrated

- Data parallelism using multiple NVIDIA GPUs.
- Efficient data loaders and batching for distributed training.
- PyTorch best practices for scalable model development.
- Experiment logging and result management.

## Requirements

- **Python 3.x**
- **PyTorch** (check `requirements.txt` for specific version)
- **NVIDIA GPU(s)** with CUDA drivers installed.

## License & Attribution

This repository was developed as a learning project for the NVIDIA Data Parallelism workshop. Feel free to use or extend it for educational and research purposes.

---

If you have any questions or wish to report issues, please use the repository's Issues page.
