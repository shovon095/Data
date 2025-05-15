Welcome to NVIDIA DGX Station Version 4.0.7 (GNU/Linux 4.15.0-213-generic x86_64)
Last login: Wed May 14 23:39:16 2025 from 129.207.35.122
shouvon@dgx1-DGX-Station:~$ which nvcc
/usr/local/cuda/bin/nvcc
shouvon@dgx1-DGX-Station:~$ ls -l /usr/local/cuda
lrwxrwxrwx 1 root root 22 May 14 22:57 /usr/local/cuda -> /etc/alternatives/cuda
shouvon@dgx1-DGX-Station:~$ sudo ln -sfn /usr/local/cuda-11.7 /usr/local/cuda
[sudo] password for shouvon:
shouvon@dgx1-DGX-Station:~$ source /etc/profile.d/cuda.sh
-bash: /etc/profile.d/cuda.sh: No such file or directory
