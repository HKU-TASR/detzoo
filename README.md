# detzoo

## Introduction

`detzoo` is a comprehensive Python library designed to bring together state-of-the-art (SOTA) object detection algorithms in a single, easy-to-use package. Developed with PyTorch, `detzoo` offers a unified interface for both training and inference using popular datasets such as  PASCAL VOC and MS COCO. The library caters to a range of users from computer vision researchers to practitioners in fields like AI security. With `detzoo`, accessing and deploying SOTA object detection models becomes straightforward, as it includes pre-trained models and customizable training configurations.

## Motivation

In the rapidly evolving field of object detection, researchers and developers often face the challenge of accessing and implementing the latest algorithms. Whether it is for advancing the state-of-the-art, establishing new baselines, or conducting comparative studies for new research papers, the need for a centralized, reliable, and up-to-date resource is paramount. This need extends beyond the computer vision community; researchers in AI security and other fields also require access to these algorithms to test and evaluate new threats.

Unfortunately, the reality is that sourcing these algorithms can be daunting. Researchers frequently encounter disparate implementations, encounter compatibility issues, and spend considerable time fixing bugs and adapting code. `detzoo` addresses these challenges by providing a unified collection of well-maintained, easy-to-integrate SOTA object detection models. By simplifying access to these tools, `detzoo` not only accelerates research and development in object detection but also facilitates cross-disciplinary applications, contributing to broader advancements in AI and computer vision.

## Features

`detzoo` is built with a range of features that make it a powerful tool for researchers and practitioners in the field of object detection. Here are some of the key features:

- **Support for Popular Datasets**: `detzoo` is designed to work seamlessly with widely-used datasets in the field of object detection, including [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html), [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) and [MS COCO 2017](https://cocodataset.org/#home). This integration allows users to easily train and test models with standard benchmarks in the industry.
- **Collection of SOTA Algorithms**: The library includes a comprehensive collection of the latest and most popular state-of-the-art object detection algorithms. These algorithms are continually updated and maintained to ensure users have access to the most advanced tools in the field.
- **Unified and User-Friendly Interface**: One of the core strengths of `detzoo` is its unified interface, which simplifies the process of working with different object detection models. This user-friendly interface ensures that researchers and developers can focus on their experiments and applications, rather than dealing with the complexities of integrating various algorithms.
- **Efficient Training and Testing**: With `detzoo`, users can quickly set up training and testing environments. The library is optimized for performance, allowing for faster experimentation cycles, which is crucial for research and development.
- **Ready-to-Use Pre-Trained Models**: To further accelerate development and research, `detzoo` provides a range of pre-trained models. These models can be used as baselines or for quick deployment, saving users the time and resources required for training models from scratch.
- **Ease of Deployment**: `detzoo` is designed not only for training and testing but also for easy deployment of object detection models. Whether it's for academic research or practical applications, users can efficiently deploy trained models to meet their specific needs.

Each of these features is crafted to ensure that `detzoo` meets the needs of its users, providing a reliable, efficient, and easy-to-use platform for object detection tasks. Whether for advancing research, developing new applications, or conducting comparative studies, `detzoo` stands as a valuable resource in the object detection community.

## Installation

### Requirements

Before installing `detzoo`, ensure that your system meets the following requirements:

- **Python**: Version 3.10.13 or newer. `detzoo` is developed and tested on this version of Python for maximum compatibility and performance.
- **PyTorch**: Version 2.0 or newer. This is a key dependency for running the object detection models.
- **Other Dependencies**: All other dependencies are listed in the `requirements.txt` file in the `detzoo` repository. These include necessary libraries and frameworks used by `detzoo`.

It is recommended to use a virtual environment to avoid any conflicts with existing packages on your system.

### Installation Guide

Follow these steps to install `detzoo`:

1. **Clone the Repository** (optional if you're installing directly from PyPI):

   ```sh
   git clone https://github.com/tju2050633/detzoo.git
   cd detzoo
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```sh
   conda create --name detzoo python=3.10.13
   conda activate detzoo
   ```

3. **Install Dependencies**:

   - If you have cloned the repository:

     ```
     pip install -r requirements.txt
     ```

   - If you are installing directly from PyPI, this step is not necessary as dependencies will be installed automatically.

4. **Install `detzoo`**:

   - If you have cloned the repository:

     ```sh
     pip install .
     ```

   - To install directly from PyPI:

     ```sh
     pip install detzoo
     ```

5. **Verify Installation**: After installation, you can verify that `detzoo` is installed correctly by running:

   ```sh
   python -c "import detzoo; print(detzoo.__version__)"
   ```

   This should print the installed version of `detzoo`.

6. **Post-Installation Steps**:

   - Depending on your specific use case, you might need to download specific pre-trained models or datasets. Refer to the `detzoo` documentation for guidance on this.

You are now ready to use `detzoo` for your object detection tasks. For detailed usage instructions, refer to the Quick Start guide or the comprehensive documentation.

## Quick Start



## Tutorials



## Documentation



## Supported Algorithms

`detzoo` includes a selection of state-of-the-art object detection algorithms, each implemented in PyTorch. Here are some of the key algorithms currently supported, along with their source code repositories:

1. **Faster R-CNN**:
   - **Description**: Faster R-CNN is a highly influential model that combines Region Proposal Networks (RPN) with Fast R-CNN for efficient and accurate object detection.
   - **Use Case**: Ideal for applications where both speed and accuracy are crucial.
   - **Source Code**: 
   - **Implemented in `detzoo`**: A PyTorch-based implementation optimized for ease of use and flexibility.
2. **Feature Pyramid Networks (FPN)**:
   - **Description**: FPN is an architecture that uses a top-down approach with lateral connections to build high-level semantic feature maps at all scales, enhancing object detection performance, especially for small objects.
   - **Use Case**: Great for detecting objects across a range of sizes and in complex images.
   - **Source Code**: 
3. **YOLOv4**:
   - **Description**: YOLOv4 is known for its speed and accuracy, making real-time object detection possible. It applies advancements like Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), and Cross mini-Batch Normalization (CmBN).
   - **Use Case**: Perfect for real-time detection scenarios, such as video analysis and autonomous driving.
   - **Source Code**: 

## Contributing



## License

`detzoo` is open-sourced under the MIT License. This license permits almost unrestricted freedom to use, modify, and distribute the software, provided that the original copyright and license notice are included with any substantial portion of the code.

For more details, see the [LICENSE](https://github.com/tju2050633/detzoo/main/LICENSE) file in the repository.

## Citations

If you find `detzoo` useful in your research, we kindly request that you cite it in your publications. This helps us to get recognition for our work and allows others to find and use our project. You can cite `detzoo` using the following Bibtex entry:

```tex
@misc{detzoo2024,
  author = {Jialin LU},
  title = {detzoo: A Collection of State-of-the-Art Object Detection Algorithms},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tju2050633/detzoo}}
}
```

## Acknowledgements



## Contact

