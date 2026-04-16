# Projeto de Processamento de Imagem

O projeto consiste numa aplição em Python que dada uma fotografia de um tabuleiro de jogo 4x4, numerado de 1 a 15:

Numa primeira instância vai transformar a imagem numa onde apenas consta o tabuleiro em si.
A partir desta vamos retirar os números presentes no tabuleiro, bem como a sua posição, este processo será feito de duas formas diferentes:
  1. *Template Matching*: A partir de templates de cada número vamos encontrar o seu *match* na fotografia original;
  2. *CNN*: Vamos criar uma rede neuronal que após treinada será capaz de retirar os números da fotografia
---
## Etapas do Projeto

1. Obter uma imagem com apenas o tabuleiro de jogo 
2. Realizar o *template matching* 
3. Obter a matriz do tabuleiro de jogo da fotografia 
4. Construir a *CNN* 
5. Repetir a **etapa 3** 
   
## 🛠️ Tecnologias
* **Linguagem:** Python 3.10+
* **Visão Computacional:** OpenCV
* **Deep Learning:** PyTorch
* **IDE:** Spyder e VsCode
---

Atualizado a: 19/01/2026


Nota Final: 19 (0-20)

