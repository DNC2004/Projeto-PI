import subprocess
import sys

try:
    import torch 
    import torchvision

except ImportError:
    print("\nDEBUG -- Bibliotecas não encontradas...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\nDEBUG -- Imports instalados!")
    
    except Exception as e:
        print(f"\nDEBUG -- Falha ao instalar: {e}")

import torch
import torchvision

import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

print("DEBUG -- Bibliotecas importadas com sucesso")

print("Bom dia!")

