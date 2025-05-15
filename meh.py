sudo tee /etc/profile.d/cuda.sh >/dev/null <<'EOF'
> # NVIDIA CUDA 11.7 environment
> export CUDA_HOME=/usr/local/cuda
> export PATH=\${CUDA_HOME}/bin:\${PATH}
> export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH}
> EOF
shouvon@dgx1-DGX-Station:~$ source /etc/profile.d/cuda.sh
shouvon@dgx1-DGX-Station:~$ which nvcc
Command 'which' is available in the following places
 * /bin/which
 * /usr/bin/which
The command could not be located because '/usr/bin:/bin' is not included in the PATH environment variable.
which: command not found
shouvon@dgx1-DGX-Station:~$ nvcc --version

Command 'nvcc' not found, but can be installed with:

sudo apt install nvidia-cuda-toolkit

shouvon@dgx1-DGX-Station:~$ nvidia-smi
Command 'nvidia-smi' is available in '/usr/bin/nvidia-smi'
The command could not be located because '/usr/bin' is not included in the PATH environment variable.
nvidia-smi: command not found
