import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from encontrar_cantos_otimizado import encontrar_cantos
from find_nums import nums_template_match


# Opções dos tabuleiros
IMAGE_PATHS = {
    "Jogo Distorcido": "Imagens/jogo_distorced.jpg",
    "Jogo Fundo Castanho": "Imagens/jogo_stor.jpg",
    "Jogo Fundo Branco": "Imagens/jogo_perp.jpeg",
    "Jogo Fundo Preto": "Imagens/jogo_blackgorund.jpg",
    "Jogo com Sombras": "Imagens/jogo_scuro.jpg",
    "Jogo Fundo Roxo": "Imagens/jogo_purp.jpg",
    "Jogo Fundo Vermelho": "Imagens/jogo_red.jpg",
}

class GameBoardApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Deteção Jogo do 15")
        self.geometry("900x700")

        # Imagem Selecionada
        self.selected_key = tk.StringVar(value=list(IMAGE_PATHS.keys())[0])

        # Outras Opções
        ttk.Label(self, text="Escolha a imagem:").pack(pady=5)
        ttk.OptionMenu(
            self, self.selected_key, self.selected_key.get(),
            *IMAGE_PATHS.keys(), command=self.load_image
        ).pack()

        ttk.Button(self, text="Executar deteção", command=self.run_detection).pack(pady=10)
        self.image_label = ttk.Label(self)
        self.image_label.pack(pady=10)
        self.load_image(self.selected_key.get())

    def load_image(self, key):
        path = IMAGE_PATHS[key]
        img = Image.open(path)
        
        self.update_idletasks() 
        orig_w, orig_h = img.size

        max_w = self.winfo_width() - 50 
        max_h = self.winfo_height() - 150 

        scale = min(max_w / orig_w, max_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def run_detection(self):
        path = IMAGE_PATHS[self.selected_key.get()]
        encontrar_cantos(path)
        nums_template_match("imagem_final.png")

if __name__ == "__main__":
    app = GameBoardApp() 
    app.mainloop()
