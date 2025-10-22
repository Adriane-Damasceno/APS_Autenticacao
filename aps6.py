import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


class FacialAuthSystem:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sistema de Autenticação - MMA")
        self.root.geometry("1000x700")
        self.root.configure(bg='#21402b')

        # Centralizar a janela
        self.center_window()

        # Configurar estilo
        self.setup_custom_style()

        # Dados de usuários
        self.users: Dict[str, Dict[str, Any]] = {
            "funcionario": {"password": "123func", "level": 1, "name": "Funcionário"},
            "diretor": {"password": "123dir", "level": 2, "name": "Diretor de Divisão"},
            "admin": {"password": "123admin", "level": 3, "name": "Ministro"},
            "usuario1": {"password": "senha1", "level": 1, "name": "Analista Ambiental"},
            "usuario2": {"password": "senha2", "level": 1, "name": "Técnico Ambiental"}
        }

        # =============================================================================
        # CONFIGURAÇÃO DA FOTO DO ADMINISTRADOR
        # =============================================================================
        self.admin_photo_path = "teste.jpg"  # DEIXE VAZIO PARA TESTE INICIAL
        self.admin_face_features = None
        # =============================================================================

        # Inicializar variáveis da câmera
        self.cap: Optional[cv2.VideoCapture] = None
        self.capturing: bool = False
        self.current_frame: Optional[np.ndarray] = None
        self.video_thread: Optional[threading.Thread] = None

        # Carregar classificador de faces
        self.face_cascade = None
        self.load_face_cascade()

        # Carregar características faciais do admin se a foto existir
        if self.admin_photo_path and os.path.exists(self.admin_photo_path):
            self.load_admin_face_features()

        self.show_login_screen()

    def load_face_cascade(self) -> None:
        """Carrega o classificador de faces do OpenCV"""
        try:
            cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
            if self.face_cascade.empty():
                raise Exception("Classificador de faces não carregado")
            print("Classificador de faces carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar classificador: {e}")
            self.face_cascade = None

    def load_admin_face_features(self) -> None:
        """Carrega as características faciais do administrador usando OpenCV"""
        try:
            if self.face_cascade is None:
                raise Exception("Classificador de faces não disponível")

            # Carregar imagem do admin
            admin_image = cv2.imread(self.admin_photo_path)
            if admin_image is None:
                raise Exception(f"Não foi possível carregar a imagem: {self.admin_photo_path}")

            # Converter para escala de cinza
            gray = cv2.cvtColor(admin_image, cv2.COLOR_BGR2GRAY)

            # Detectar rostos
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Recortar o rosto
                face_roi = gray[y:y + h, x:x + w]

                # Extrair características simples
                features = self.extract_face_features(face_roi)
                self.admin_face_features = features
                print("Características faciais do admin carregadas com sucesso!")
                print(f"Rosto detectado: {x}, {y}, {w}, {h}")
            else:
                print("Nenhum rosto detectado na foto do administrador")
                self.admin_face_features = None

        except Exception as e:
            print(f"Erro ao carregar foto do admin: {e}")
            self.admin_face_features = None

    def extract_face_features(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Extrai características simples do rosto usando OpenCV"""
        try:
            # Redimensionar para tamanho padrão
            face_standard = cv2.resize(face_image, (100, 100))

            # Calcular histograma normalizado
            hist = cv2.calcHist([face_standard], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Calcar características de textura (LBP simples)
            lbp_features = self.calculate_texture_features(face_standard)

            return {
                'histogram': hist,
                'texture': lbp_features,
                'face_standard': face_standard
            }
        except Exception as e:
            print(f"Erro ao extrair características: {e}")
            return {}

    def calculate_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Calcula características de textura simples"""
        # Usar filtros simples para textura
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Calcular magnitude do gradiente
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return gradient_magnitude.flatten()

    def compare_faces(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Compara duas faces baseado em suas características"""
        try:
            if not features1 or not features2:
                return 0.0

            # Comparar histogramas usando correlação
            hist_corr = cv2.compareHist(features1['histogram'], features2['histogram'], cv2.HISTCMP_CORREL)

            # Converter para valor entre 0 e 1
            hist_similarity = max(0.0, (hist_corr + 1) / 2.0)

            # Comparação simples de textura (correlação entre gradientes)
            if 'texture' in features1 and 'texture' in features2:
                texture_corr = np.corrcoef(features1['texture'], features2['texture'])[0, 1]
                if np.isnan(texture_corr):
                    texture_similarity = 0.0
                else:
                    texture_similarity = max(0.0, texture_corr)
            else:
                texture_similarity = 0.5  # Valor neutro se não houver textura

            # Combinação ponderada
            similarity = (hist_similarity * 0.7) + (texture_similarity * 0.3)

            return similarity
        except Exception as e:
            print(f"Erro na comparação: {e}")
            return 0.0

    def center_window(self) -> None:
        """Centraliza a janela na tela"""
        self.root.update_idletasks()
        width = 1000
        height = 700
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_custom_style(self) -> None:
        """Configura estilos personalizados"""
        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'primary': '#3ad631',
            'secondary': '##21402b',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'background': '#21402b',
            'card_bg': '#345c32',
            'header_bg': '#162616'
        }

    def show_login_screen(self) -> None:
        """Tela de login principal"""
        self.stop_camera()

        for widget in self.root.winfo_children():
            widget.destroy()

        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['header_bg'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="Ministério do Meio Ambiente",
            font=("Arial", 16, "bold"),
            bg=self.colors['header_bg'],
            fg='white'
        ).pack(pady=20)

        main_frame = ttk.Frame(self.root, padding="40")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        title_label = tk.Label(
            main_frame,
            text="Sistema de Monitoramento Ambiental",
            font=("Arial", 20, "bold"),
            bg=self.colors['background'],
            fg='white'
        )
        title_label.pack(pady=20)

        # Card de login
        login_card = tk.Frame(main_frame, bg=self.colors['card_bg'], relief='raised', bd=2)
        login_card.pack(pady=20, padx=100, fill=tk.X)

        # Campo de login
        tk.Label(
            login_card,
            text="Usuário:",
            font=("Arial", 12, "bold"),
            bg=self.colors['card_bg'],
            fg='white'
        ).pack(anchor=tk.W, pady=(20, 5), padx=20)

        self.login_entry = tk.Entry(login_card, font=("Arial", 12), width=30)
        self.login_entry.pack(fill=tk.X, pady=5, padx=20)

        # Campo de senha
        tk.Label(
            login_card,
            text="Senha:",
            font=("Arial", 12, "bold"),
            bg=self.colors['card_bg'],
            fg='white'
        ).pack(anchor=tk.W, pady=(20, 5), padx=20)

        self.password_entry = tk.Entry(login_card, font=("Arial", 12), show="•", width=30)
        self.password_entry.pack(fill=tk.X, pady=5, padx=20)

        # Botão de login
        login_button = tk.Button(
            login_card,
            text="ENTRAR",
            command=self.standard_auth,
            bg=self.colors['primary'],
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=20,
            pady=10
        )
        login_button.pack(pady=30, padx=20)

        # Status
        status_text = "Sistema de Informações Estratégicas"
        if self.admin_photo_path and os.path.exists(self.admin_photo_path) and self.admin_face_features is not None:
            status_text += "\n✅ Foto admin configurada: validação facial ativa"
        else:
            status_text += "\n⚠️ Foto admin não configurada: usando validação simulada"

        self.status_label = tk.Label(
            main_frame,
            text=status_text,
            font=("Arial", 10),
            bg=self.colors['background'],
            fg='#bdc3c7',
            justify=tk.CENTER
        )
        self.status_label.pack(pady=10)

        # Configurar Enter para login
        self.login_entry.focus()
        self.password_entry.bind('<Return>', lambda event: self.standard_auth())

    def standard_auth(self) -> None:
        """Autenticação com login e senha"""
        login = self.login_entry.get().strip()
        password = self.password_entry.get()

        if not login or not password:
            messagebox.showerror("Erro", "Por favor, preencha login e senha")
            return

        if login in self.users and self.users[login]["password"] == password:
            level = self.users[login]["level"]

            # Se for nível 3 (ministro), verificar se tem foto configurada
            if level == 3:
                if not self.admin_photo_path or not os.path.exists(self.admin_photo_path):
                    messagebox.showwarning(
                        "Foto não configurada",
                        "Foto não configurada.\n\n"
                        "Usando validação simulada para teste."
                    )
                elif self.admin_face_features is None:
                    messagebox.showwarning(
                        "Foto inválida",
                        "Não foi possível processar a foto.\n\n"
                        "Usando validação simulada para teste."
                    )

            self.redirect_after_auth(level, login)
        else:
            messagebox.showerror("Erro", "Credenciais inválidas")

    def redirect_after_auth(self, level: int, username: str) -> None:
        """Redireciona para a tela apropriada"""
        if level == 1:
            self.show_level1_screen(username)
        elif level == 2:
            self.show_level2_screen(username)
        elif level == 3:
            self.start_facial_auth(username)

    def start_facial_auth(self, username: str) -> None:
        """Inicia validação facial"""
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['header_bg'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="Validação - Nível 3",
            font=("Arial", 14, "bold"),
            bg=self.colors['header_bg'],
            fg='white'
        ).pack(pady=15)

        # Título
        tk.Label(
            main_frame,
            text="Verificação Facial",
            font=("Arial", 18, "bold"),
            bg=self.colors['background'],
            fg='white'
        ).pack(pady=20)

        # Instruções
        instructions = "Posicione seu rosto na câmera para validação"
        if self.admin_face_features is not None:
            instructions += "\n✅ Comparando com foto cadastrada"
        else:
            instructions += "\n⚠️ Usando validação simulada"

        tk.Label(
            main_frame,
            text=instructions,
            font=("Arial", 10),
            bg=self.colors['background'],
            fg='#bdc3c7'
        ).pack(pady=5)

        # Área da câmera
        self.camera_label = tk.Label(
            main_frame,
            text="Iniciando câmera...",
            bg='black',
            fg='white',
            font=('Arial', 12)
        )
        self.camera_label.pack(pady=20, padx=50, fill=tk.BOTH, expand=True)

        # Botões
        button_frame = tk.Frame(main_frame, bg=self.colors['background'])
        button_frame.pack(pady=20)

        tk.Button(
            button_frame,
            text="Validar Rosto",
            command=self.validate_face,
            bg=self.colors['warning'],
            fg='white',
            font=("Arial", 10, "bold"),
            padx=20
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            button_frame,
            text="Voltar ao Login",
            command=self.show_login_screen,
            bg=self.colors['card_bg'],
            fg='white',
            font=("Arial", 10),
            padx=20
        ).pack(side=tk.LEFT, padx=10)

        # Iniciar câmera
        self.capturing = True
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a câmera")
            self.show_login_screen()
            return

        # Thread para captura de vídeo
        self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.video_thread.start()

    def update_frame(self) -> None:
        """Atualiza o frame da câmera"""
        while self.capturing:
            ret, frame = self.cap.read()
            if ret:
                try:
                    # Converter BGR para RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detectar rostos se o classificador estiver carregado
                    if self.face_cascade is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

                        # Desenhar retângulos nos rostos
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Armazenar frame atual
                    self.current_frame = frame.copy()

                    # Redimensionar e converter para ImageTk
                    frame_rgb = cv2.resize(frame_rgb, (640, 480))
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # Atualizar interface
                    self.camera_label.imgtk = imgtk
                    self.camera_label.configure(image=imgtk)

                except Exception as e:
                    print(f"Erro no frame: {e}")

            time.sleep(0.03)

    def validate_face(self) -> None:
        """Valida o rosto comparando com a foto do ministro/admin"""
        if self.current_frame is None:
            messagebox.showerror("Erro", "Nenhuma imagem capturada")
            return

        try:
            # Verificar se há rostos na imagem
            if self.face_cascade is None:
                messagebox.showerror("Erro", "Sistema de detecção não disponível")
                return

            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

            if len(faces) == 0:
                messagebox.showerror("Erro", "Nenhum rosto detectado na imagem")
                return

            # Se tem características do ministro/admin, fazer comparação real
            if self.admin_face_features is not None:
                x, y, w, h = faces[0]
                # Recortar o rosto
                face_roi = gray[y:y + h, x:x + w]

                # Extrair características do rosto capturado
                current_features = self.extract_face_features(face_roi)

                if not current_features:
                    messagebox.showerror("Erro", "Não foi possível extrair características do rosto")
                    return

                # Comparar com as características do ministro
                similarity = self.compare_faces(self.admin_face_features, current_features)

                print(f"Similaridade detectada: {similarity:.3f}")

                # Threshold para considerar match
                if similarity > 0.4:
                    messagebox.showinfo("Sucesso",
                                        f"✅ Validação facial confirmada!\nSimilaridade: {similarity:.3f}\nAcesso concedido ao painel ministerial.")
                    self.stop_camera()
                    self.show_level3_screen("Ministro")
                else:
                    messagebox.showerror("Falha",
                                         f"❌ Rosto não corresponde ao cadastro.\nSimilaridade: {similarity:.3f}\nAcesso negado.")
            else:
                # Validação simulada (fallback)
                self.simulated_face_validation()

        except Exception as e:
            messagebox.showerror("Erro", f"Erro na validação facial: {e}")

    def simulated_face_validation(self) -> None:
        """Validação facial simulada (fallback)"""
        try:
            if self.face_cascade is not None and self.current_frame is not None:
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

                if len(faces) > 0:
                    # Simular processamento
                    self.camera_label.config(text="Validando identidade...")
                    self.root.update()
                    time.sleep(2)

                    # 80% de chance de sucesso
                    success = np.random.random() > 0.2

                    if success:
                        messagebox.showinfo("Sucesso", "✅ Validação simulada: Identidade confirmada!")
                        self.stop_camera()
                        self.show_level3_screen("Ministro")
                    else:
                        messagebox.showerror("Falha", "❌ Validação: Falha na verificação. Tente novamente.")
                        self.camera_label.config(text="Câmera ativa")
                else:
                    messagebox.showerror("Erro", "Nenhum rosto detectado")
            else:
                messagebox.showerror("Erro", "Sistema de detecção não disponível")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro na validação simulada: {e}")

    def stop_camera(self) -> None:
        """Para a captura da câmera"""
        self.capturing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.current_frame = None

    def create_folder_widget(self, parent, name, color='#3498db'):
        """Cria um widget de pasta"""
        folder_frame = tk.Frame(parent, bg=color, relief='raised', bd=1, width=120, height=100)
        folder_frame.pack_propagate(False)

        tk.Label(
            folder_frame,
            text="📁",
            font=("Arial", 24),
            bg=color,
            fg='white'
        ).pack(pady=(15, 5))

        tk.Label(
            folder_frame,
            text=name,
            font=("Arial", 9),
            bg=color,
            fg='white',
            wraplength=100,
            justify=tk.CENTER
        ).pack(pady=(0, 10))

        return folder_frame

    def show_level1_screen(self, username: str) -> None:
        """Tela do funcionário - Nível 1 (Acesso Público)"""
        self.stop_camera()
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text=f"Painel do Funcionário - {username}",
            font=("Arial", 16, "bold"),
            bg=self.colors['primary'],
            fg='white'
        ).pack(side=tk.LEFT, padx=20, pady=20)

        tk.Button(
            header_frame,
            text="Sair",
            command=self.show_login_screen,
            bg=self.colors['danger'],
            fg='white',
            font=("Arial", 10, "bold"),
            padx=15
        ).pack(side=tk.RIGHT, padx=20, pady=20)

        # Conteúdo
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título da seção
        tk.Label(
            content_frame,
            text="INFORMAÇÕES DE NÍVEL 1 - ACESSO PÚBLICO",
            font=("Arial", 14, "bold"),
            bg=self.colors['background'],
            fg='white'
        ).pack(pady=(0, 20))

        # Grid de pastas
        folders_frame = tk.Frame(content_frame, bg=self.colors['background'])
        folders_frame.pack(fill=tk.BOTH, expand=True)

        # Pastas do nível 1
        folders_level1 = [
            ("Relatórios Públicos", "Relatórios anuais de monitoramento"),
            ("Legislação Ambiental", "Leis e regulamentos públicos"),
            ("Dados Abertos", "Dados públicos de qualidade da água"),
            ("Educação Ambiental", "Materiais educativos e campanhas"),
            ("Licenciamentos", "Processos de licenciamento ambiental"),
            ("Fiscalização", "Ações de fiscalização registradas")
        ]

        for i, (name, desc) in enumerate(folders_level1):
            row = i // 3
            col = i % 3
            folder = self.create_folder_widget(folders_frame, name, '#27ae60')
            folder.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

            # Tooltip
            self.create_tooltip(folder, desc)

        # Configurar grid
        for i in range(3):
            folders_frame.columnconfigure(i, weight=1)
        for i in range(2):
            folders_frame.rowconfigure(i, weight=1)

    def show_level2_screen(self, username: str) -> None:
        """Tela do diretor - Nível 2 (Acesso Restrito)"""
        self.stop_camera()
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['warning'], height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text=f"Painel do Diretor - {username}",
            font=("Arial", 16, "bold"),
            bg=self.colors['warning'],
            fg='white'
        ).pack(side=tk.LEFT, padx=20, pady=20)

        tk.Button(
            header_frame,
            text="Sair",
            command=self.show_login_screen,
            bg=self.colors['danger'],
            fg='white',
            font=("Arial", 10, "bold"),
            padx=15
        ).pack(side=tk.RIGHT, padx=20, pady=20)

        # Conteúdo
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título da seção
        tk.Label(
            content_frame,
            text="INFORMAÇÕES DE NÍVEL 2 - ACESSO RESTRITO A DIRETORES",
            font=("Arial", 14, "bold"),
            bg=self.colors['background'],
            fg='white'
        ).pack(pady=(0, 20))

        # Grid de pastas
        folders_frame = tk.Frame(content_frame, bg=self.colors['background'])
        folders_frame.pack(fill=tk.BOTH, expand=True)

        # Pastas do nível 2
        folders_level2 = [
            ("Propriedades Monitoradas", "Lista de propriedades sob investigação"),
            ("Agrotóxicos Proibidos", "Relatório de substâncias banidas"),
            ("Contaminações Detectadas", "Casos confirmados de contaminação"),
            ("Ações Corretivas", "Planos de ação e medidas tomadas"),
            ("Relatórios Internos", "Análises técnicas internas"),
            ("Alertas Regionais", "Áreas com alto risco de contaminação")
        ]

        for i, (name, desc) in enumerate(folders_level2):
            row = i // 3
            col = i % 3
            folder = self.create_folder_widget(folders_frame, name, '#f39c12')
            folder.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

            # Tooltip
            self.create_tooltip(folder, desc)

        # Configurar grid
        for i in range(3):
            folders_frame.columnconfigure(i, weight=1)
        for i in range(2):
            folders_frame.rowconfigure(i, weight=1)

    def show_level3_screen(self, username: str) -> None:
        """Tela do ministro - Nível 3 (Acesso Máximo)"""
        self.stop_camera()
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['danger'], height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text=f"Painel Ministerial - {username}",
            font=("Arial", 16, "bold"),
            bg=self.colors['danger'],
            fg='white'
        ).pack(side=tk.LEFT, padx=20, pady=20)

        tk.Button(
            header_frame,
            text="Sair",
            command=self.show_login_screen,
            bg=self.colors['primary'],
            fg='white',
            font=("Arial", 10, "bold"),
            padx=15
        ).pack(side=tk.RIGHT, padx=20, pady=20)

        # Conteúdo
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título da seção
        tk.Label(
            content_frame,
            text="INFORMAÇÕES DE NÍVEL 3 - ACESSO EXCLUSIVO MINISTERIAL",
            font=("Arial", 14, "bold"),
            bg=self.colors['background'],
            fg='white'
        ).pack(pady=(0, 20))

        # Grid de pastas
        folders_frame = tk.Frame(content_frame, bg=self.colors['background'])
        folders_frame.pack(fill=tk.BOTH, expand=True)

        # Pastas do nível 3
        folders_level3 = [
            ("Relatórios Estratégicos", "Análises de impacto nacional"),
            ("Operações Especiais", "Operações sigilosas em andamento"),
            ("Dados Sensíveis", "Informações classificadas como secretas"),
            ("Investigação MP", "Acompanhamento de ações do Ministério Público"),
            ("Crise Hídrica", "Plano de contingência para desastres"),
            ("Acordos Internacionais", "Tratados e acordos ambientais")
        ]

        for i, (name, desc) in enumerate(folders_level3):
            row = i // 3
            col = i % 3
            folder = self.create_folder_widget(folders_frame, name, '#e74c3c')
            folder.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

            # Tooltip
            self.create_tooltip(folder, desc)

        # Configurar grid
        for i in range(3):
            folders_frame.columnconfigure(i, weight=1)
        for i in range(2):
            folders_frame.rowconfigure(i, weight=1)

    def create_tooltip(self, widget, text):
        """Cria um tooltip para os widgets"""

        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = tk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()

            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def run(self) -> None:
        """Executa a aplicação"""
        try:
            self.root.mainloop()
        finally:
            self.stop_camera()


# Executar a aplicação
if __name__ == "__main__":
    app = FacialAuthSystem()
    app.run()