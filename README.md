# **Deep Learning for IR and Visible Image Fusion**

### **Project Overview**
This project explores advanced deep learning techniques for fusing infrared (IR) and visible images, aimed at improving image quality and reducing processing times. Traditional image fusion methods like Wavelet Transforms, Principal Component Analysis (PCA), and Low-Rank Latent Representation (LRR) serve as benchmarks for performance. These methods, however, suffer from long processing times and poor retention of high-frequency details.

To overcome these limitations, we have implemented deep learning models, including Generative Adversarial Networks (GANs) and diffusion models, which significantly improve fusion performance. In addition, the Laplacian Gaussian Pyramid method has been employed to enhance edge preservation and detail retention in fused images.

The models are designed for real-time applications such as surveillance, medical imaging, and autonomous vehicles, and will be deployed on the NVIDIA Jetson Nano for efficient, low-latency processing.

### **Key Features**
- **Conventional Methods**: Wavelet Transforms, PCA, and LRR implemented in MATLAB for benchmarking.
- **Deep Learning Models**: GANs and diffusion models implemented for improved quality and speed.
- **Laplacian Gaussian Pyramid**: Enhances detail retention in image fusion.
- **Real-Time Deployment**: Models to be deployed on NVIDIA Jetson Nano with inference times under 100 milliseconds.
- **Performance Metrics**: Evaluations using PSNR, SSIM, and qualitative visual assessments.

### **Getting Started**
Follow the instructions below to set up and run the project.

#### **Prerequisites**
- **Jetson Nano**: JetPack SDK (Ubuntu 18.04-based)
- **Development Machine**: Ubuntu 18.04+ or Windows with WSL2
- **Python**: 3.6 or later
- **TensorFlow**: 2.x or compatible version
- **PyTorch**: 1.x or compatible version
- **OpenCV**: 4.x or later

#### **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
