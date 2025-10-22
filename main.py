import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps


class FacialAuthSystem:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sistema de Autenticação Avançado")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Centralizar a janela na tela
        self.center_window()

        # Configurar estilo personalizado
        self.setup_custom_style()

        # Dados de usuários
        self.users: Dict[str, Dict[str, Any]] = {
            "funcionario": {"password": "func123", "level": 0},
            "admin": {"password": "admin123", "level": 2},
            "user1": {"password": "senha1", "level": 1},
            "user2": {"password": "senha2", "level": 1}
        }

        # Variáveis para captura de vídeo
        self.cap: Optional[cv2.VideoCapture] = None
        self.capturing: bool = False
        self.current_frame: Optional[np.ndarray] = None
        self.video_thread: Optional[threading.Thread] = None

        # Carregar o classificador Haar Cascade
        cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))

        self.show_login_screen()

    def center_window(self) -> None:
        """Centraliza a janela na tela"""
        self.root.update_idletasks()
        width = 900
        height = 700
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_custom_style(self) -> None:
        """Configura estilos personalizados para a aplicação"""
        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'primary': '#3498db',
            'secondary': '#2c3e50',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#34495e',
            'background': '#2c3e50',
            'card_bg': '#34495e'
        }

        style.configure('Custom.TFrame', background=self.colors['background'])
        style.configure('Card.TFrame', background=self.colors['card_bg'], relief='raised', borderwidth=2)
        style.configure('Center.TFrame', background=self.colors['background'])

        style.configure('Primary.TButton',
                        background=self.colors['primary'],
                        foreground='white',
                        focuscolor='none',
                        borderwidth=0,
                        focusthickness=0,
                        padding=(20, 10))
        style.map('Primary.TButton',
                  background=[('active', '#2980b9'), ('pressed', '#21618c')])

        style.configure('Secondary.TButton',
                        background=self.colors['dark'],
                        foreground='white',
                        borderwidth=0,
                        padding=(15, 8))
        style.map('Secondary.TButton',
                  background=[('active', '#2c3e50'), ('pressed', '#1a252f')])

        style.configure('Title.TLabel',
                        background=self.colors['background'],
                        foreground=self.colors['light'],
                        font=('Arial', 20, 'bold'))

        style.configure('Subtitle.TLabel',
                        background=self.colors['background'],
                        foreground=self.colors['primary'],
                        font=('Arial', 14))

        style.configure('Normal.TLabel',
                        background=self.colors['card_bg'],
                        foreground=self.colors['light'],
                        font=('Arial', 11))

        style.configure('Custom.TEntry',
                        fieldbackground=self.colors['light'],
                        foreground=self.colors['dark'],
                        borderwidth=2,
                        focusthickness=2,
                        focuscolor=self.colors['primary'])

        style.configure('Custom.TLabelframe',
                        background=self.colors['background'],
                        foreground=self.colors['light'])
        style.configure('Custom.TLabelframe.Label',
                        background=self.colors['background'],
                        foreground=self.colors['primary'])

    def on_closing(self) -> None:
        """Método chamado ao fechar a aplicação"""
        self.capturing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def show_login_screen(self) -> None:
        """Exibe a tela de login principal com campos"""
        # Limpar a tela anterior
        for widget in self.root.winfo_children():
            widget.destroy()

        # Frame principal
        main_frame = ttk.Frame(self.root, style='Custom.TFrame', padding="40")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurar pesos para centralização
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)  # Coluna extra para centralização
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)  # Conteúdo principal
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Título
        title_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        title_frame.grid(row=0, column=0, columnspan=3, pady=(20, 10), sticky=(tk.W, tk.E))
        title_frame.columnconfigure(0, weight=1)

        ttk.Label(
            title_frame,
            text="🔐 ",
            font=("Arial", 24),
            background=self.colors['background'],
            foreground=self.colors['primary']
        ).grid(row=0, column=0)

        ttk.Label(
            title_frame,
            text="Sistema de Autenticação",
            style='Title.TLabel'
        ).grid(row=1, column=0, pady=5)

        ttk.Label(
            title_frame,
            text="Acesso Seguro",
            style='Subtitle.TLabel'
        ).grid(row=2, column=0, pady=5)

        # Container central para o card de login
        center_container = ttk.Frame(main_frame, style='Center.TFrame')
        center_container.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=20)
        center_container.columnconfigure(0, weight=1)

        # Card de login
        login_card = ttk.Frame(center_container, style='Card.TFrame', padding="30")
        login_card.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=50)
        login_card.columnconfigure(0, weight=1)

        # Campo de login
        ttk.Label(
            login_card,
            text="Usuário:",
            style='Normal.TLabel',
            font=('Arial', 12, 'bold')
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        self.login_entry = ttk.Entry(
            login_card,
            font=("Arial", 12),
            width=25,
            style='Custom.TEntry'
        )
        self.login_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Campo de senha
        ttk.Label(
            login_card,
            text="Senha:",
            style='Normal.TLabel',
            font=('Arial', 12, 'bold')
        ).grid(row=2, column=0, sticky=tk.W, pady=(0, 10))

        self.password_entry = ttk.Entry(
            login_card,
            font=("Arial", 12),
            show="•",
            width=25,
            style='Custom.TEntry'
        )
        self.password_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 30))

        # Botão de autenticação
        login_button = ttk.Button(
            login_card,
            text="🔓 ENTRAR NO SISTEMA",
            command=self.standard_auth,
            style='Primary.TButton'
        )
        login_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)

        # Container para informações (centralizado)
        info_container = ttk.Frame(main_frame, style='Center.TFrame')
        info_container.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        info_container.columnconfigure(0, weight=1)

        # Informação sobre níveis de acesso
        info_frame = ttk.LabelFrame(
            info_container,
            text=" ℹ️  Níveis de Acesso",
            style='Custom.TLabelframe',
            padding="15"
        )
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(0, weight=1)

        levels_info = [
            ("🎯 Nível 0 - Funcionário", "Acesso básico ao sistema"),
            ("👤 Nível 1 - Usuário", "Acesso intermediário com mais recursos"),
            ("👑 Nível 2 - Administrador", "Acesso completo + verificação facial")
        ]

        for i, (level, desc) in enumerate(levels_info):
            level_frame = ttk.Frame(info_frame, style='Custom.TFrame')
            level_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=5)

            ttk.Label(
                level_frame,
                text=level,
                style='Normal.TLabel',
                font=('Arial', 10, 'bold')
            ).pack(side=tk.LEFT, anchor='w')

            ttk.Label(
                level_frame,
                text=f" - {desc}",
                style='Normal.TLabel',
                font=('Arial', 10)
            ).pack(side=tk.LEFT, anchor='w')

        # Status
        status_container = ttk.Frame(main_frame, style='Center.TFrame')
        status_container.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=10)
        status_container.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            status_container,
            text="Digite suas credenciais para acessar o sistema",
            style='Normal.TLabel',
            foreground='#bdc3c7'
        )
        self.status_label.grid(row=0, column=0)

        # Focar no campo de login e configurar Enter para submeter
        self.login_entry.focus()
        self.password_entry.bind('<Return>', lambda event: self.standard_auth())

    def create_card(self, parent, title: str, content: list, row: int, column: int) -> None:
        """Cria um card estilizado para conteúdo"""
        card = ttk.Frame(parent, style='Card.TFrame', padding="20")
        card.grid(row=row, column=column, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(
            card,
            text=title,
            style='Normal.TLabel',
            font=('Arial', 14, 'bold')
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 15))

        for i, item in enumerate(content, 1):
            ttk.Label(
                card,
                text=f"• {item}",
                style='Normal.TLabel',
                font=('Arial', 11)
            ).grid(row=i, column=0, sticky=tk.W, pady=2)

        return card

    def standard_auth(self) -> None:
        """Realiza autenticação com login e senha para todos os usuários"""
        login = self.login_entry.get().strip()
        password = self.password_entry.get()

        if not login or not password:
            messagebox.showerror("Erro", "Por favor, preencha login e senha")
            return

        if login in self.users and self.users[login]["password"] == password:
            level = self.users[login]["level"]
            self.status_label.config(
                text=f"✅ Autenticado como {login} - Nível {level}",
                foreground=self.colors['success']
            )

            self.root.after(1000, lambda: self.redirect_after_auth(level, login))
        else:
            self.status_label.config(
                text="❌ Credenciais inválidas - Tente novamente",
                foreground=self.colors['danger']
            )
            messagebox.showerror("Erro", "Credenciais inválidas")

    def redirect_after_auth(self, level: int, username: str) -> None:
        """Redireciona após autenticação bem-sucedida"""
        if level == 0:
            self.show_level0_screen(username)
        elif level == 1:
            self.show_level1_screen(username)
        elif level == 2:
            self.start_facial_auth(username)

    def start_facial_auth(self, username: str) -> None:
        """Inicia o processo de autenticação facial com câmera"""
        # Tela de verificação facial
        for widget in self.root.winfo_children():
            widget.destroy()

        auth_frame = ttk.Frame(self.root, style='Custom.TFrame', padding="30")
        auth_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurar grid para centralização
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        auth_frame.columnconfigure(0, weight=1)
        auth_frame.columnconfigure(1, weight=1)
        auth_frame.columnconfigure(2, weight=1)
        auth_frame.rowconfigure(0, weight=0)  # Cabeçalho
        auth_frame.rowconfigure(1, weight=0)  # Subtítulo
        auth_frame.rowconfigure(2, weight=1)  # Câmera
        auth_frame.rowconfigure(3, weight=0)  # Botões

        # Cabeçalho
        header_frame = ttk.Frame(auth_frame, style='Custom.TFrame')
        header_frame.grid(row=0, column=1, pady=(0, 20), sticky=(tk.W, tk.E))
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(
            header_frame,
            text="👑 ",
            font=("Arial", 24),
            background=self.colors['background'],
            foreground=self.colors['primary']
        ).grid(row=0, column=0)

        ttk.Label(
            header_frame,
            text=f"Verificação Facial - {username}",
            style='Title.TLabel'
        ).grid(row=1, column=0, pady=5)

        # Subtítulo
        subtitle_frame = ttk.Frame(auth_frame, style='Custom.TFrame')
        subtitle_frame.grid(row=1, column=1, pady=(0, 20), sticky=(tk.W, tk.E))
        subtitle_frame.columnconfigure(0, weight=1)

        ttk.Label(
            subtitle_frame,
            text="🔒 Última etapa: validação facial necessária para acesso administrativo",
            style='Subtitle.TLabel'
        ).grid(row=0, column=0)

        # Container central para a câmera
        camera_container = ttk.Frame(auth_frame, style='Custom.TFrame')
        camera_container.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        camera_container.columnconfigure(0, weight=1)
        camera_container.rowconfigure(0, weight=1)

        # Card da câmera
        camera_card = ttk.Frame(camera_container, style='Card.TFrame', padding="15")
        camera_card.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        camera_card.columnconfigure(0, weight=1)
        camera_card.rowconfigure(0, weight=1)

        # Área de visualização da câmera
        self.camera_label = ttk.Label(
            camera_card,
            text="🎥 Iniciando câmera...\n\nAguardando ativação do dispositivo",
            background='#1a1a1a',
            foreground='white',
            font=('Arial', 12),
            justify=tk.CENTER
        )
        self.camera_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=50, pady=50)

        # Container central para botões
        buttons_container = ttk.Frame(auth_frame, style='Custom.TFrame')
        buttons_container.grid(row=3, column=1, pady=20, sticky=(tk.W, tk.E))
        buttons_container.columnconfigure(0, weight=1)

        buttons_frame = ttk.Frame(buttons_container, style='Custom.TFrame')
        buttons_frame.grid(row=0, column=0)

        ttk.Button(
            buttons_frame,
            text="📷 Validar Rosto",
            command=self.validate_face,
            style='Primary.TButton'
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            buttons_frame,
            text="↩️ Voltar ao Login",
            command=self.show_login_screen,
            style='Secondary.TButton'
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
        """Atualiza o frame da câmera em tempo real"""
        while self.capturing:
            ret, frame = self.cap.read()
            if ret:
                try:
                    # Converter BGR para RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detectar rostos
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                    # Desenhar retângulos nos rostos detectados
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(frame_rgb, 'Rosto Detectado', (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Armazenar frame atual para validação
                    self.current_frame = frame

                    # Redimensionar e converter para ImageTk
                    frame_rgb = cv2.resize(frame_rgb, (640, 480))
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # Atualizar a interface na thread principal
                    self.root.after(0, self.update_camera_label, imgtk)

                except Exception as e:
                    print(f"Erro ao processar frame: {e}")

            time.sleep(0.03)

    def update_camera_label(self, imgtk: ImageTk.PhotoImage) -> None:
        """Atualiza o label da câmera (deve ser chamado na thread principal)"""
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

    def validate_face(self) -> None:
        """Valida o rosto detectado"""
        if self.current_frame is None:
            messagebox.showerror("Erro", "Nenhuma imagem capturada")
            return

        try:
            # Converter para escala de cinza para detecção
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Atualizar interface para mostrar processamento
                self.camera_label.config(
                    text="🔍 Processando verificação facial...\n\nAguarde...",
                    image=''
                )

                # Simular processo de validação
                self.root.after(2000, lambda: self.finish_face_validation(len(faces) > 0))
            else:
                messagebox.showerror("Erro", "Nenhum rosto detectado na imagem")

        except Exception as e:
            messagebox.showerror("Erro", f"Falha no processamento facial: {e}")

    def finish_face_validation(self, face_detected: bool) -> None:
        """Finaliza a validação facial"""
        if face_detected:
            success = np.random.random() > 0.2  # 80% de chance de sucesso

            if success:
                messagebox.showinfo("Sucesso", "✅ Rosto validado com sucesso!")
                self.capturing = False
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                self.show_admin_screen()
            else:
                messagebox.showerror("Falha", "❌ Falha na validação facial. Tente novamente.")
                # Restaurar a visualização da câmera
                self.capturing = True
                self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
                self.video_thread.start()

    def show_level0_screen(self, username: str) -> None:
        """Exibe a tela de nível 0 (Funcionário) com design moderno"""
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self.root, style='Custom.TFrame', padding="30")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # Cabeçalho
        header_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        header_frame.grid(row=0, column=1, pady=(0, 30), sticky=(tk.W, tk.E))
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(
            header_frame,
            text="🎯 ",
            font=("Arial", 24),
            background=self.colors['background'],
            foreground=self.colors['primary']
        ).grid(row=0, column=0)

        ttk.Label(
            header_frame,
            text="Painel do Funcionário",
            style='Title.TLabel'
        ).grid(row=1, column=0, pady=5)

        ttk.Label(
            header_frame,
            text=f"(Nível 0) - {username}",
            style='Subtitle.TLabel'
        ).grid(row=2, column=0, pady=5)

        # Conteúdo
        content_container = ttk.Frame(main_frame, style='Custom.TFrame')
        content_container.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=20)
        content_container.columnconfigure(0, weight=1)

        content_frame = ttk.Frame(content_container, style='Custom.TFrame')
        content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)

        recursos = [
            "Visualizar dados de entrada",
            "Consultar informações básicas",
            "Acessar relatórios simples",
            "Registrar ponto eletrônico",
            "Visualizar calendário corporativo",
            "Solicitar férias"
        ]
        self.create_card(content_frame, "📊 Recursos Disponíveis", recursos, 0, 0)

        # Card de informações
        info = [
            "Acesso básico ao sistema",
            "Permissões limitadas",
            "Horário comercial: 08:00-18:00",
            "Suporte: interno@empresa.com"
        ]
        self.create_card(content_frame, "ℹ️ Informações", info, 0, 1)

        # Botão de sair
        button_container = ttk.Frame(main_frame, style='Custom.TFrame')
        button_container.grid(row=2, column=1, pady=30, sticky=(tk.W, tk.E))
        button_container.columnconfigure(0, weight=1)

        ttk.Button(
            button_container,
            text="🚪 Sair do Sistema",
            command=self.show_login_screen,
            style='Secondary.TButton'
        ).grid(row=0, column=0)

    def show_level1_screen(self, username: str) -> None:
        """Exibe a tela de nível 1 (Usuário Autenticado)"""
        # Implementação similar ao nível 0
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self.root, style='Custom.TFrame', padding="30")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # Cabeçalho
        header_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        header_frame.grid(row=0, column=1, pady=(0, 30), sticky=(tk.W, tk.E))
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(
            header_frame,
            text="👤 ",
            font=("Arial", 24),
            background=self.colors['background'],
            foreground=self.colors['primary']
        ).grid(row=0, column=0)

        ttk.Label(
            header_frame,
            text="Painel do Usuário",
            style='Title.TLabel'
        ).grid(row=1, column=0, pady=5)

        ttk.Label(
            header_frame,
            text=f"(Nível 1) - {username}",
            style='Subtitle.TLabel'
        ).grid(row=2, column=0, pady=5)

        # Conteúdo
        content_container = ttk.Frame(main_frame, style='Custom.TFrame')
        content_container.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=20)
        content_container.columnconfigure(0, weight=1)

        # Botão
        button_container = ttk.Frame(main_frame, style='Custom.TFrame')
        button_container.grid(row=2, column=1, pady=30, sticky=(tk.W, tk.E))
        button_container.columnconfigure(0, weight=1)

        ttk.Button(
            button_container,
            text="🚪 Sair do Sistema",
            command=self.show_login_screen,
            style='Secondary.TButton'
        ).grid(row=0, column=0)

    def show_admin_screen(self) -> None:
        """Exibe a tela de nível 2 (Administrador)"""
        # Implementação similar
        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self.root, style='Custom.TFrame', padding="30")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        header_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        header_frame.grid(row=0, column=1, pady=(0, 30), sticky=(tk.W, tk.E))
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(
            header_frame,
            text="👑 ",
            font=("Arial", 24),
            background=self.colors['background'],
            foreground=self.colors['primary']
        ).grid(row=0, column=0)

        ttk.Label(
            header_frame,
            text="Painel de Administração",
            style='Title.TLabel'
        ).grid(row=1, column=0, pady=5)

        ttk.Label(
            header_frame,
            text="(Nível 2) - Acesso Completo",
            style='Subtitle.TLabel'
        ).grid(row=2, column=0, pady=5)

        button_container = ttk.Frame(main_frame, style='Custom.TFrame')
        button_container.grid(row=2, column=1, pady=30, sticky=(tk.W, tk.E))
        button_container.columnconfigure(0, weight=1)

        ttk.Button(
            button_container,
            text="🚪 Sair do Sistema",
            command=self.show_login_screen,
            style='Secondary.TButton'
        ).grid(row=0, column=0)

    def run(self) -> None:
        """Executa a aplicação"""
        try:
            self.root.mainloop()
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()


# Executar a aplicação
if __name__ == "__main__":
    app = FacialAuthSystem()
    app.run()