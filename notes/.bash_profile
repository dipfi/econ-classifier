# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH

env2lmod

module load gcc/6.3.0 python_gpu/3.7.4 cuda/11.1.1 cudnn/8.1.0.77

module load eth_proxy

cd /cluster/work/lawecon/Projects/Ash_Durrer/dev/scripts

source venv/bin/activate