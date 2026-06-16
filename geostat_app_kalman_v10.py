import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import gstools as gs
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.stats import boxcox, zscore
from scipy.special import inv_boxcox
from matplotlib.path import Path
import pygad
import time
from itertools import product as itertools_product
import warnings
warnings.filterwarnings('ignore')


class GeostatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis geoestadístico comparativo")
        self.root.geometry("1400x900")
        
        # Variables de datos
        self.df_muestreo = None
        self.df_malla = None
        self.parametro_col = None
        self.puntos_fijos = []  # Índices de puntos fijos
        self.puntos_opcionales = []  # Índices de puntos opcionales
        self.hull_expandido = None  # Para visualizar el área con buffer
        
        # Pesos de pozos (priorización por importancia de cada pozo)
        self.pesos_pozos = None          # Array de pesos por pozo (None = sin pesos)
        self.usar_pesos_pozos = tk.BooleanVar(value=False)
        self.col_peso_pozo = None        # Nombre de la columna detectada
        
        # Variable para indicar si la malla tiene pesos
        self.malla_ponderada = False
        self.pesos_malla = None  # Array de pesos de la malla
        
        # Parámetros del variograma
        self.var_range = tk.DoubleVar(value=100)
        self.var_sill = tk.DoubleVar(value=1.0)
        self.var_nugget = tk.DoubleVar(value=0.1)
        self.num_nodos_malla = tk.IntVar(value=500)
        self.tipo_malla = tk.StringVar(value="rectangular")
        
        # Modelo de variograma y transformación de datos
        self.tipo_modelo = tk.StringVar(value="Spherical")
        self.matern_nu = tk.DoubleVar(value=1.5)
        self.tipo_transformacion = tk.StringVar(value="Ninguna")
        self.z_transformado = None
        self.lambda_boxcox = None
        
        # Multiparámetro
        self.parametros_disponibles = []
        self.parametros_config = {}     # {nombre: dict con range, sill, nugget, modelo, etc.}
        self.parametro_actual_nombre = None
        self.param_activo_var = tk.BooleanVar(value=True)
        self.param_peso_var = tk.DoubleVar(value=1.0)
        
        # Auto-actualización del variograma (timer con debounce)
        self._vario_update_job = None
        
        # Variables para algoritmo genético
        self.max_puntos = tk.IntVar(value=20)
        self.usar_opcionales = tk.BooleanVar(value=True)
        self.usar_malla = tk.BooleanVar(value=False)
        
        # Parámetros de PyGAD
        self.ga_num_generations = tk.IntVar(value=30)
        self.ga_sol_per_pop = tk.IntVar(value=20)
        self.ga_num_parents_mating = tk.IntVar(value=4)
        self.ga_mutation_percent = tk.IntVar(value=10)
        self.ga_parent_selection = tk.StringVar(value="rank")
        self.ga_crossover_type = tk.StringVar(value="single_point")
        self.ga_mutation_type = tk.StringVar(value="random")
        self.ga_metodo_evaluacion = tk.StringVar(value="kriging_ordinario")
        
        # Resultados
        self.resultados_secuencial = None
        self.resultados_genetico = None
        self.resultados_kalman = None
        self.tiempo_secuencial = 0
        self.tiempo_genetico = 0
        self.tiempo_kalman = 0
        
        # Experimentación automática del AG
        self.resultados_experimentacion = []
        self.exp_corriendo = False
        
        self.crear_interfaz()
        
        # Registrar trazas para auto-actualización del variograma
        self.var_range.trace_add('write', self._on_variograma_param_change)
        self.var_sill.trace_add('write', self._on_variograma_param_change)
        self.var_nugget.trace_add('write', self._on_variograma_param_change)
        self.matern_nu.trace_add('write', self._on_variograma_param_change)
        self.tipo_modelo.trace_add('write', self._on_variograma_param_change)
    
    def crear_interfaz(self):
        # Frame principal con notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Carga de datos
        tab_datos = ttk.Frame(notebook)
        notebook.add(tab_datos, text="Datos")
        self.crear_tab_datos(tab_datos)
        
        # Tab 2: Parámetros del variograma
        tab_variograma = ttk.Frame(notebook)
        notebook.add(tab_variograma, text="Variograma")
        self.crear_tab_variograma(tab_variograma)
        
        # Tab 3: Malla de estimación
        tab_malla = ttk.Frame(notebook)
        notebook.add(tab_malla, text="Malla de estimación")
        self.crear_tab_malla(tab_malla)
        
        # Tab 4: Análisis comparativo
        tab_analisis = ttk.Frame(notebook)
        notebook.add(tab_analisis, text="Análisis comparativo")
        self.crear_tab_analisis(tab_analisis)
        
        # Tab 5: Experimentación automática AG
        tab_experimentacion = ttk.Frame(notebook)
        notebook.add(tab_experimentacion, text="Experimentación AG")
        self.crear_tab_experimentacion(tab_experimentacion)
        
        # Tab 6: Resultados
        tab_resultados = ttk.Frame(notebook)
        notebook.add(tab_resultados, text="Resultados")
        self.crear_tab_resultados(tab_resultados)
    
    def crear_tab_datos(self, parent):
        frame = ttk.LabelFrame(parent, text="Carga de datos de muestreo", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Cargar archivo Excel con columnas: clave, x, y, parámetro").pack(pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Selección de archivo", command=self.cargar_muestreo).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Selección de puntos fijos/opcionales", 
                  command=self.seleccionar_puntos_fijos).pack(side='left', padx=5)
        
        # Frame para información
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill='x', pady=5)
        self.lbl_info_datos = ttk.Label(info_frame, text="")
        self.lbl_info_datos.pack()
        
        # Frame para pesos de pozos
        peso_pozo_frame = ttk.LabelFrame(frame, text="Pesos de pozos (columna opcional: peso_pozo)", padding=5)
        peso_pozo_frame.pack(fill='x', pady=5)
        
        self.chk_usar_pesos_pozos = ttk.Checkbutton(peso_pozo_frame, 
            text="Usar pesos de pozos en la priorización", variable=self.usar_pesos_pozos)
        self.chk_usar_pesos_pozos.pack(side='left', padx=5)
        
        self.lbl_pesos_pozos = ttk.Label(peso_pozo_frame, text="Sin columna de pesos detectada", 
                                          font=('Arial', 9), foreground='gray')
        self.lbl_pesos_pozos.pack(side='left', padx=10)
        
        # Frame para Treeview con scrollbars
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill='both', expand=True, pady=10)
        
        # Scrollbars
        scroll_y = ttk.Scrollbar(tree_frame, orient='vertical')
        scroll_x = ttk.Scrollbar(tree_frame, orient='horizontal')
        
        # Treeview
        self.tree_datos = ttk.Treeview(tree_frame, 
                                        yscrollcommand=scroll_y.set,
                                        xscrollcommand=scroll_x.set,
                                        height=15)
        
        scroll_y.config(command=self.tree_datos.yview)
        scroll_x.config(command=self.tree_datos.xview)
        
        # Posicionar elementos
        self.tree_datos.grid(row=0, column=0, sticky='nsew')
        scroll_y.grid(row=0, column=1, sticky='ns')
        scroll_x.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def crear_tab_variograma(self, parent):
        # Frame con scroll
        canvas_var = tk.Canvas(parent)
        scrollbar_var = ttk.Scrollbar(parent, orient="vertical", command=canvas_var.yview)
        scrollable_var = ttk.Frame(canvas_var)
        
        scrollable_var.bind(
            "<Configure>",
            lambda e: canvas_var.configure(scrollregion=canvas_var.bbox("all"))
        )
        window_id = canvas_var.create_window((0, 0), window=scrollable_var, anchor="nw")
        canvas_var.configure(yscrollcommand=scrollbar_var.set)
        
        def _on_canvas_configure(event):
            canvas_var.itemconfig(window_id, width=event.width)
        canvas_var.bind('<Configure>', _on_canvas_configure)
        
        canvas_var.pack(side="left", fill="both", expand=True)
        scrollbar_var.pack(side="right", fill="y")
        
        frame = ttk.Frame(scrollable_var)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ============================================================
        # Tabla resumen de TODOS los parámetros
        # ============================================================
        tabla_frame = ttk.LabelFrame(frame, text="Resumen de parámetros (clic en una fila para editar)", padding=10)
        tabla_frame.pack(fill='x', pady=5)
        
        btn_tabla = ttk.Frame(tabla_frame)
        btn_tabla.pack(fill='x', pady=(0, 5))
        
        ttk.Button(btn_tabla, text="Estimar todos automáticamente",
                  command=lambda: self._estimar_todos_parametros(mostrar_msg=True)).pack(side='left', padx=5)
        ttk.Button(btn_tabla, text="Actualizar tabla",
                  command=self._refrescar_tabla_parametros).pack(side='left', padx=5)
        
        cols_tabla = ('parametro', 'activo', 'peso', 'modelo', 'transformacion',
                     'range', 'p_sill', 'nugget', 'sill')
        
        tree_var_frame = ttk.Frame(tabla_frame)
        tree_var_frame.pack(fill='x')
        
        scroll_y_tv = ttk.Scrollbar(tree_var_frame, orient='vertical')
        self.tree_params_vario = ttk.Treeview(tree_var_frame, columns=cols_tabla, show='headings',
                                               yscrollcommand=scroll_y_tv.set, height=6)
        scroll_y_tv.config(command=self.tree_params_vario.yview)
        
        col_cfg = {
            'parametro': ('Parámetro', 120), 'activo': ('Activo', 50), 'peso': ('Peso', 55),
            'modelo': ('Modelo', 90), 'transformacion': ('Transf.', 90),
            'range': ('Range', 80), 'p_sill': ('P. Sill', 80),
            'nugget': ('Nugget', 80), 'sill': ('Sill', 80)
        }
        for col_id, (heading, width) in col_cfg.items():
            self.tree_params_vario.heading(col_id, text=heading)
            self.tree_params_vario.column(col_id, width=width, minwidth=40)
        
        self.tree_params_vario.grid(row=0, column=0, sticky='nsew')
        scroll_y_tv.grid(row=0, column=1, sticky='ns')
        tree_var_frame.grid_columnconfigure(0, weight=1)
        
        self.tree_params_vario.bind('<<TreeviewSelect>>', self._on_seleccion_tabla_param)
        
        self.lbl_param_editando = ttk.Label(tabla_frame, text="Cargue datos para ver parámetros",
                                             font=('Arial', 10, 'bold'), foreground='blue')
        self.lbl_param_editando.pack(pady=(5, 0))
        
        # ============================================================
        # Activación y peso
        # ============================================================
        activo_frame = ttk.LabelFrame(frame, text="Activación y peso del parámetro seleccionado", padding=10)
        activo_frame.pack(fill='x', pady=5)
        
        ttk.Checkbutton(activo_frame, text="Activo",
                        variable=self.param_activo_var).grid(row=0, column=0, padx=5, pady=3, sticky='w')
        ttk.Label(activo_frame, text="Peso:").grid(row=0, column=1, padx=5, pady=3)
        ttk.Entry(activo_frame, textvariable=self.param_peso_var, width=8).grid(row=0, column=2, padx=5, pady=3)
        
        # ============================================================
        # Transformación de datos
        # ============================================================
        transf_frame = ttk.LabelFrame(frame, text="Transformación de datos", padding=10)
        transf_frame.pack(fill='x', pady=5)
        transf_frame.columnconfigure(2, weight=1)
        
        ttk.Label(transf_frame, text="Transformación:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        transf_combo = ttk.Combobox(transf_frame, textvariable=self.tipo_transformacion,
                                     values=["Ninguna", "Log (ln)", "Log10", "Raíz cuadrada", 
                                             "Box-Cox", "Normal Score"],
                                     state='readonly', width=18)
        transf_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(transf_frame, text="Aplicar transformación", 
                  command=self._aplicar_y_actualizar_transformacion).grid(row=0, column=2, padx=10, pady=5)
        
        self.lbl_transf_info = ttk.Label(transf_frame, text="Transformación: Ninguna", 
                                          font=('Arial', 9), foreground='gray')
        self.lbl_transf_info.grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky='w')
        
        # ============================================================
        # Modelo de variograma
        # ============================================================
        modelo_frame = ttk.LabelFrame(frame, text="Modelo de variograma", padding=10)
        modelo_frame.pack(fill='x', pady=5)
        modelo_frame.columnconfigure(3, weight=1)
        
        ttk.Label(modelo_frame, text="Tipo de modelo:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        modelo_combo = ttk.Combobox(modelo_frame, textvariable=self.tipo_modelo,
                                     values=["Spherical", "Exponential", "Gaussian", 
                                             "Matérn", "Stable", "Linear", "Cubic"],
                                     state='readonly', width=18)
        modelo_combo.grid(row=0, column=1, padx=5, pady=5)
        
        self.lbl_param_extra = ttk.Label(modelo_frame, text="ν (Matérn) / α (Stable):")
        self.lbl_param_extra.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        ttk.Entry(modelo_frame, textvariable=self.matern_nu, width=8).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(modelo_frame, 
                 text="Matérn: ν=0.5→Exponential, ν→∞→Gaussian  |  Stable: α=1→Exp, α=2→Gauss",
                 font=('Arial', 8), foreground='gray').grid(row=1, column=0, columnspan=4, padx=5, pady=1, sticky='w')
        
        # ============================================================
        # Parámetros del variograma (se actualizan en tiempo real)
        # ============================================================
        params_frame = ttk.LabelFrame(frame, 
            text="Parámetros del variograma (la gráfica se actualiza automáticamente)", padding=10)
        params_frame.pack(fill='x', pady=5)
        
        ttk.Label(params_frame, text="Range:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.var_range, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Partial Sill:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.var_sill, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Nugget:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.var_nugget, width=15).grid(row=2, column=1, padx=5, pady=5)
        
        btn_params = ttk.Frame(params_frame)
        btn_params.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_params, text="Estimar este parámetro",
                  command=self.estimar_parametros).pack(side='left', padx=5)
        ttk.Button(btn_params, text="Guardar cambios en tabla",
                  command=self._guardar_edicion_a_config).pack(side='left', padx=5)
        
        # Frame para gráfica del variograma
        self.fig_vario = Figure(figsize=(8, 5))
        self.canvas_vario = FigureCanvasTkAgg(self.fig_vario, frame)
        self.canvas_vario.get_tk_widget().pack(pady=10)
    
    # ================================================================
    # Auto-actualización del variograma con debounce
    # ================================================================
    def _on_variograma_param_change(self, *args):
        """
        Callback cuando cambian range, sill, nugget, modelo o matern_nu.
        Usa debounce de 400ms para no recalcular en cada tecla.
        """
        if self.df_muestreo is None:
            return
        if self._vario_update_job is not None:
            self.root.after_cancel(self._vario_update_job)
        self._vario_update_job = self.root.after(400, self._actualizar_variograma_auto)
    
    def _actualizar_variograma_auto(self):
        """Ejecuta recalcular_variograma si los valores son válidos"""
        self._vario_update_job = None
        try:
            # Verificar que los valores sean numéricos válidos
            r = self.var_range.get()
            s = self.var_sill.get()
            n = self.var_nugget.get()
            if r > 0 and s >= 0 and n >= 0:
                self.recalcular_variograma()
        except (tk.TclError, ValueError):
            pass  # Valor parcial mientras se escribe
    
    # ================================================================
    # Gestión de tabla de parámetros
    # ================================================================
    def _refrescar_tabla_parametros(self):
        """Reconstruye la tabla de parámetros"""
        if not hasattr(self, 'tree_params_vario'):
            return
        for item in self.tree_params_vario.get_children():
            self.tree_params_vario.delete(item)
        
        for pname, cfg in self.parametros_config.items():
            sill = cfg['var_sill'] + cfg['var_nugget']
            modelo_str = cfg['tipo_modelo']
            if cfg['tipo_modelo'] == 'Matérn':
                modelo_str += f" ν={cfg['matern_nu']:.1f}"
            elif cfg['tipo_modelo'] == 'Stable':
                modelo_str += f" α={cfg['matern_nu']:.1f}"
            
            self.tree_params_vario.insert('', 'end', iid=pname, values=(
                pname, 'Sí' if cfg['activo'] else 'No', f"{cfg['peso']:.2f}",
                modelo_str, cfg['tipo_transformacion'],
                f"{cfg['var_range']:.2f}", f"{cfg['var_sill']:.4f}",
                f"{cfg['var_nugget']:.4f}", f"{sill:.4f}"
            ))
    
    def _on_seleccion_tabla_param(self, event=None):
        """Al hacer clic en una fila, guarda el anterior y carga el nuevo"""
        sel = self.tree_params_vario.selection()
        if not sel:
            return
        pname = sel[0]
        if pname not in self.parametros_config:
            return
        self._guardar_edicion_a_config(silencioso=True)
        self._cargar_param_en_edicion(pname)
        self.recalcular_variograma()
    
    def _cargar_param_en_edicion(self, pname):
        """Carga valores de parametros_config[pname] en los campos de edición"""
        if pname not in self.parametros_config:
            return
        cfg = self.parametros_config[pname]
        self.parametro_actual_nombre = pname
        self.parametro_col = pname
        
        self.param_activo_var.set(cfg['activo'])
        self.param_peso_var.set(cfg['peso'])
        self.var_range.set(cfg['var_range'])
        self.var_sill.set(cfg['var_sill'])
        self.var_nugget.set(cfg['var_nugget'])
        self.tipo_modelo.set(cfg['tipo_modelo'])
        self.matern_nu.set(cfg['matern_nu'])
        self.tipo_transformacion.set(cfg['tipo_transformacion'])
        self.z_transformado = cfg['z_transformado'].copy() if cfg['z_transformado'] is not None else None
        self.lambda_boxcox = cfg['lambda_boxcox']
        
        try:
            self.lbl_param_editando.config(
                text=f"Editando: {pname}  |  Peso: {cfg['peso']:.2f}  |  "
                     f"{'Activo' if cfg['activo'] else 'INACTIVO'}")
        except:
            pass
        
        try:
            if self.z_transformado is not None:
                z_t = self.z_transformado
                self.lbl_transf_info.config(
                    text=f"Transf: {cfg['tipo_transformacion']}  |  "
                         f"min={z_t.min():.4f}, max={z_t.max():.4f}, "
                         f"media={z_t.mean():.4f}, var={np.var(z_t):.4f}",
                    foreground='blue')
            else:
                self.lbl_transf_info.config(text="Transformación: Ninguna", foreground='gray')
        except:
            pass
    
    def _guardar_edicion_a_config(self, silencioso=False):
        """Guarda valores de los campos de edición en parametros_config"""
        pname = self.parametro_actual_nombre
        if pname is None or pname not in self.parametros_config:
            if not silencioso:
                messagebox.showwarning("Advertencia", "Seleccione un parámetro de la tabla primero")
            return
        
        cfg = self.parametros_config[pname]
        try:
            cfg['activo'] = self.param_activo_var.get()
            cfg['peso'] = max(self.param_peso_var.get(), 0.0)
            cfg['var_range'] = self.var_range.get()
            cfg['var_sill'] = self.var_sill.get()
            cfg['var_nugget'] = self.var_nugget.get()
            cfg['tipo_modelo'] = self.tipo_modelo.get()
            cfg['matern_nu'] = self.matern_nu.get()
            cfg['tipo_transformacion'] = self.tipo_transformacion.get()
            if self.z_transformado is not None:
                cfg['z_transformado'] = self.z_transformado.copy()
            cfg['lambda_boxcox'] = self.lambda_boxcox
        except (tk.TclError, ValueError):
            if not silencioso:
                messagebox.showwarning("Advertencia", "Algún valor numérico no es válido")
            return
        
        self._refrescar_tabla_parametros()
        
        if not silencioso:
            messagebox.showinfo("Guardado", f"Cambios guardados para '{pname}'")
    
    def _estimar_parametros_para(self, pname):
        """Estima range, sill, nugget para un parámetro específico sin tocar la UI"""
        if self.df_muestreo is None or pname not in self.parametros_config:
            return
        
        cfg = self.parametros_config[pname]
        x = self.df_muestreo['x'].values
        y = self.df_muestreo['y'].values
        z = cfg['z_transformado'] if cfg['z_transformado'] is not None else \
            self.df_muestreo[pname].values.astype(float)
        
        try:
            coords = np.column_stack([x, y])
            dists = cdist(coords, coords)
            max_dist = np.max(dists[dists > 0])
            
            n_puntos = len(x)
            if n_puntos < 10: n_bins = 3
            elif n_puntos < 20: n_bins = 5
            elif n_puntos < 50: n_bins = 10
            else: n_bins = 15
            
            bins = np.linspace(0, max_dist, n_bins + 1)
            try:
                bin_center, gamma = gs.vario_estimate((x, y), z, bin_edges=bins)
            except:
                bin_center, gamma = gs.vario_estimate((x, y), z, bin_edges=n_bins)
            
            estimated_range = max_dist * 0.65
            varianza_total = np.var(z)
            
            if len(gamma) > 2:
                estimated_sill = np.mean(gamma[-min(3, len(gamma)):]) * 0.85
            else:
                estimated_sill = varianza_total * 0.7
            
            if len(gamma) > 0:
                estimated_nugget = min(gamma[0], varianza_total * 0.3)
            else:
                estimated_nugget = varianza_total * 0.15
            
            if estimated_nugget + estimated_sill > varianza_total:
                estimated_sill = varianza_total * 0.7
                estimated_nugget = varianza_total * 0.2
            
            cfg['var_range'] = round(estimated_range, 2)
            cfg['var_sill'] = round(estimated_sill, 4)
            cfg['var_nugget'] = round(estimated_nugget, 4)
        except:
            pass
    
    def _estimar_todos_parametros(self, mostrar_msg=True):
        """Estima variograma para TODOS los parámetros cargados"""
        if not self.parametros_config:
            return
        n_ok = 0
        for pname in self.parametros_config:
            try:
                self._estimar_parametros_para(pname)
                n_ok += 1
            except:
                pass
        
        if self.parametro_actual_nombre in self.parametros_config:
            self._cargar_param_en_edicion(self.parametro_actual_nombre)
        
        self._refrescar_tabla_parametros()
        
        if self.df_muestreo is not None:
            self.recalcular_variograma()
        
        if mostrar_msg:
            messagebox.showinfo("Estimación completada",
                f"Parámetros estimados para {n_ok}/{len(self.parametros_config)} parámetros")

    def _aplicar_y_actualizar_transformacion(self):
        """Aplica la transformación y actualiza el variograma"""
        if self.df_muestreo is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos de muestreo")
            return
        
        exito = self.aplicar_transformacion()
        
        if exito:
            z_t = self.obtener_datos_transformados()
            info = self.info_transformacion()
            
            stats = (f"Transformación: {info}  |  "
                    f"min={z_t.min():.4f}, max={z_t.max():.4f}, "
                    f"media={z_t.mean():.4f}, var={np.var(z_t):.4f}")
            self.lbl_transf_info.config(text=stats, foreground='blue')
            
            # Guardar transformación en config
            pname = self.parametro_actual_nombre
            if pname and pname in self.parametros_config:
                self.parametros_config[pname]['tipo_transformacion'] = self.tipo_transformacion.get()
                self.parametros_config[pname]['z_transformado'] = self.z_transformado.copy() if self.z_transformado is not None else None
                self.parametros_config[pname]['lambda_boxcox'] = self.lambda_boxcox
            
            # Recalcular variograma con datos transformados
            self.estimar_parametros()
    
    def crear_tab_malla(self, parent):
        frame = ttk.LabelFrame(parent, text="Malla de estimación", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Generación automática
        gen_frame = ttk.LabelFrame(frame, text="Generación automática (sin pesos)", padding=10)
        gen_frame.pack(pady=10, fill='x')
        
        ttk.Label(gen_frame, text="Número de nodos:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(gen_frame, textvariable=self.num_nodos_malla, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(gen_frame, text="Tipo de malla:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Radiobutton(gen_frame, text="Rectangular", variable=self.tipo_malla, 
                       value="rectangular").grid(row=1, column=1, sticky='w')
        ttk.Radiobutton(gen_frame, text="Ajustada", variable=self.tipo_malla, 
                       value="ajustada").grid(row=2, column=1, sticky='w')
        
        ttk.Button(gen_frame, text="Generar malla", 
                  command=self.generar_malla).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Carga desde archivo CON PESOS
        carga_frame = ttk.LabelFrame(frame, text="Cargar desde archivo (con pesos opcionales)", padding=10)
        carga_frame.pack(pady=10, fill='x')
        
        ttk.Label(carga_frame, text="Columnas requeridas: clave, x, y", 
                 font=('Arial', 9)).pack(pady=2)
        ttk.Label(carga_frame, text="Columna opcional: peso (valores numéricos positivos)", 
                 font=('Arial', 9, 'italic')).pack(pady=2)
        ttk.Label(carga_frame, text="Los pesos ponderan la importancia de cada nodo en el cálculo de varianza.", 
                 font=('Arial', 8), foreground='gray').pack(pady=2)
        ttk.Label(carga_frame, text="Zonas con pesos altos darán mayor prioridad a los pozos cercanos.", 
                 font=('Arial', 8), foreground='gray').pack(pady=2)
        
        ttk.Button(carga_frame, text="Cargar malla desde Excel", 
                  command=self.cargar_malla).pack(pady=10)
        
        # Información de la malla cargada
        self.lbl_info_malla = ttk.Label(frame, text="", font=('Arial', 10))
        self.lbl_info_malla.pack(pady=5)
        
        # Visualización
        self.fig_malla = Figure(figsize=(8, 5))
        self.canvas_malla = FigureCanvasTkAgg(self.fig_malla, frame)
        self.canvas_malla.get_tk_widget().pack(pady=10)
    
    def crear_tab_analisis(self, parent):
        # Frame con scroll para contenido extenso
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        _win_id_an = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e, c=canvas, w=_win_id_an: c.itemconfig(w, width=e.width))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        frame = ttk.LabelFrame(scrollable_frame, text="Configuración de análisis", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configuración general
        config_frame = ttk.LabelFrame(frame, text="Parámetros generales", padding=10)
        config_frame.pack(pady=10, fill='x')
        
        ttk.Label(config_frame, text="Número máximo de puntos a seleccionar:").grid(
            row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(config_frame, textvariable=self.max_puntos, width=15).grid(
            row=0, column=1, padx=5, pady=5)
        
        # Configuración algoritmo genético - Opciones básicas
        ag_frame = ttk.LabelFrame(frame, text="Opciones de algoritmo genético", padding=10)
        ag_frame.pack(pady=10, fill='x')
        
        ttk.Label(ag_frame, text="Método de evaluación del fitness:", 
                 font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        ttk.Radiobutton(ag_frame, text="Kriging Ordinario (más preciso, más lento)", 
                        variable=self.ga_metodo_evaluacion, 
                        value="kriging_ordinario").grid(row=1, column=0, padx=20, pady=2, sticky='w')
        ttk.Radiobutton(ag_frame, text="Filtro de Kalman / Kriging Simple (más rápido, matrices precomputadas)", 
                        variable=self.ga_metodo_evaluacion, 
                        value="filtro_kalman").grid(row=2, column=0, padx=20, pady=2, sticky='w')
        
        ttk.Label(ag_frame, 
                 text="Nota: Las varianzas reportadas siempre se recalculan con Kriging Ordinario para comparabilidad.",
                 font=('Arial', 8), foreground='gray').grid(row=3, column=0, padx=20, pady=3, sticky='w')
        
        # Parámetros de PyGAD
        pygad_frame = ttk.LabelFrame(frame, text="Parámetros de PyGAD", padding=10)
        pygad_frame.pack(pady=10, fill='both', expand=True)
        
        # Frame izquierdo - Parámetros numéricos
        left_frame = ttk.Frame(pygad_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        ttk.Label(left_frame, text="Número de generaciones:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(left_frame, textvariable=self.ga_num_generations, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(left_frame, text="Tamaño de población:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(left_frame, textvariable=self.ga_sol_per_pop, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(left_frame, text="Padres para apareamiento:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(left_frame, textvariable=self.ga_num_parents_mating, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(left_frame, text="% Mutación:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(left_frame, textvariable=self.ga_mutation_percent, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        # Frame derecho - Parámetros categóricos
        right_frame = ttk.Frame(pygad_frame)
        right_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        ttk.Label(right_frame, text="Selección de padres:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        parent_selection_combo = ttk.Combobox(right_frame, textvariable=self.ga_parent_selection, 
                                             values=["sss", "rws", "sus", "rank", "random", "tournament"], 
                                             state='readonly', width=15)
        parent_selection_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(right_frame, text="Tipo de cruce:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        crossover_combo = ttk.Combobox(right_frame, textvariable=self.ga_crossover_type, 
                                      values=["single_point", "two_points", "uniform", "scattered"], 
                                      state='readonly', width=15)
        crossover_combo.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(right_frame, text="Tipo de mutación:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        mutation_combo = ttk.Combobox(right_frame, textvariable=self.ga_mutation_type, 
                                     values=["random", "swap", "inversion", "scramble", "adaptive"], 
                                     state='readonly', width=15)
        mutation_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Botón para restaurar valores por defecto
        ttk.Button(right_frame, text="Restaurar valores por defecto", 
                  command=self.restaurar_parametros_ga).grid(row=3, column=0, columnspan=2, pady=10)
        
        # ============================================================
        # Descripción del filtro de Kalman
        # ============================================================
        kalman_frame = ttk.LabelFrame(frame, text="Filtro de Kalman - Información", padding=10)
        kalman_frame.pack(pady=10, fill='x')
        
        ttk.Label(kalman_frame, 
                 text="Fase 1: El filtro de Kalman determina el ORDEN óptimo de selección de pozos",
                 font=('Arial', 9)).pack(anchor='w', pady=1)
        ttk.Label(kalman_frame, 
                 text="mediante actualizaciones rank-1 de la covarianza (kriging simple, muy rápido).",
                 font=('Arial', 9)).pack(anchor='w', pady=1)
        ttk.Label(kalman_frame, 
                 text="Fase 2: Se recalcula la varianza con KRIGING ORDINARIO (misma métrica que",
                 font=('Arial', 9)).pack(anchor='w', pady=1)
        ttk.Label(kalman_frame, 
                 text="inclusiones sucesivas) para garantizar comparabilidad directa entre métodos.",
                 font=('Arial', 9)).pack(anchor='w', pady=1)
        ttk.Label(kalman_frame, 
                 text="Ventaja: orden optimizado más rápido; las varianzas reportadas son directamente comparables.",
                 font=('Arial', 9, 'italic'), foreground='blue').pack(anchor='w', pady=1)
        
        # ============================================================
        # Botones de ejecución
        # ============================================================
        botones_frame = ttk.Frame(frame)
        botones_frame.pack(pady=10)
        
        ttk.Button(botones_frame, text="Ejecutar inclusiones sucesivas (Kriging Ordinario)", 
                  command=self.ejecutar_secuencial, 
                  style='Accent.TButton').pack(pady=5, fill='x')
        
        ttk.Button(botones_frame, text="Ejecutar algoritmo genético", 
                  command=self.ejecutar_genetico,
                  style='Accent.TButton').pack(pady=5, fill='x')
        
        ttk.Button(botones_frame, text="Ejecutar filtro de Kalman", 
                  command=self.ejecutar_kalman,
                  style='Accent.TButton').pack(pady=5, fill='x')
        
        # Separador
        ttk.Separator(botones_frame, orient='horizontal').pack(fill='x', pady=8)
        
        ttk.Button(botones_frame, text="Ejecutar los tres métodos", 
                  command=self.ejecutar_todos,
                  style='Accent.TButton').pack(pady=5, fill='x')
    
    def crear_tab_experimentacion(self, parent):
        """Pestaña de experimentación automática del algoritmo genético"""
        # Frame con scroll
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        _win_id_exp = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e, c=canvas, w=_win_id_exp: c.itemconfig(w, width=e.width))
        
        # Habilitar scroll con rueda del ratón
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = ttk.Frame(scrollable_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(main_frame, text="Experimentación automática de parámetros del Algoritmo Genético",
                 font=('Arial', 12, 'bold')).pack(pady=5)
        ttk.Label(main_frame, text="Define intervalos para cada parámetro. Se ejecutarán todas las combinaciones.",
                 font=('Arial', 9), foreground='gray').pack(pady=2)
        
        # ============================================================
        # Método de evaluación del fitness
        # ============================================================
        metodo_frame = ttk.LabelFrame(main_frame, text="Método de evaluación del fitness", padding=10)
        metodo_frame.pack(fill='x', pady=5)
        
        self.exp_metodo = tk.StringVar(value="kriging_ordinario")
        ttk.Radiobutton(metodo_frame, text="Kriging Ordinario (más preciso, más lento)",
                        variable=self.exp_metodo, value="kriging_ordinario").pack(anchor='w', padx=10)
        ttk.Radiobutton(metodo_frame, text="Filtro de Kalman / Kriging Simple (más rápido)",
                        variable=self.exp_metodo, value="filtro_kalman").pack(anchor='w', padx=10)
        ttk.Label(metodo_frame, text="Las varianzas finales siempre se reportan con Kriging Ordinario.",
                 font=('Arial', 8), foreground='gray').pack(anchor='w', padx=10, pady=2)
        
        # ============================================================
        # Número de puntos a evaluar
        # ============================================================
        puntos_frame = ttk.LabelFrame(main_frame, text="Puntos de evaluación", padding=10)
        puntos_frame.pack(fill='x', pady=5)
        
        self.exp_max_puntos = tk.IntVar(value=20)
        ttk.Label(puntos_frame, text="Número máximo de puntos a seleccionar:").grid(
            row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(puntos_frame, textvariable=self.exp_max_puntos, width=10).grid(
            row=0, column=1, padx=5, pady=5)
        
        # ============================================================
        # Parámetros numéricos con intervalos (min, max, paso)
        # ============================================================
        num_frame = ttk.LabelFrame(main_frame, text="Parámetros numéricos (mínimo, máximo, paso)", padding=10)
        num_frame.pack(fill='x', pady=5)
        
        # Encabezados
        ttk.Label(num_frame, text="Parámetro", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=3)
        ttk.Label(num_frame, text="Mínimo", font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5, pady=3)
        ttk.Label(num_frame, text="Máximo", font=('Arial', 9, 'bold')).grid(row=0, column=2, padx=5, pady=3)
        ttk.Label(num_frame, text="Paso", font=('Arial', 9, 'bold')).grid(row=0, column=3, padx=5, pady=3)
        ttk.Label(num_frame, text="Valores", font=('Arial', 9, 'bold')).grid(row=0, column=4, padx=5, pady=3)
        
        # Número de generaciones
        self.exp_gen_min = tk.IntVar(value=30)
        self.exp_gen_max = tk.IntVar(value=30)
        self.exp_gen_paso = tk.IntVar(value=10)
        
        ttk.Label(num_frame, text="Núm. generaciones:").grid(row=1, column=0, padx=5, pady=3, sticky='w')
        ttk.Entry(num_frame, textvariable=self.exp_gen_min, width=8).grid(row=1, column=1, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_gen_max, width=8).grid(row=1, column=2, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_gen_paso, width=8).grid(row=1, column=3, padx=3, pady=3)
        self.lbl_exp_gen = ttk.Label(num_frame, text="[30]", foreground='blue')
        self.lbl_exp_gen.grid(row=1, column=4, padx=5, pady=3)
        
        # Tamaño de población
        self.exp_pop_min = tk.IntVar(value=20)
        self.exp_pop_max = tk.IntVar(value=20)
        self.exp_pop_paso = tk.IntVar(value=10)
        
        ttk.Label(num_frame, text="Tamaño de población:").grid(row=2, column=0, padx=5, pady=3, sticky='w')
        ttk.Entry(num_frame, textvariable=self.exp_pop_min, width=8).grid(row=2, column=1, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_pop_max, width=8).grid(row=2, column=2, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_pop_paso, width=8).grid(row=2, column=3, padx=3, pady=3)
        self.lbl_exp_pop = ttk.Label(num_frame, text="[20]", foreground='blue')
        self.lbl_exp_pop.grid(row=2, column=4, padx=5, pady=3)
        
        # Padres para apareamiento
        self.exp_parents_min = tk.IntVar(value=4)
        self.exp_parents_max = tk.IntVar(value=4)
        self.exp_parents_paso = tk.IntVar(value=2)
        
        ttk.Label(num_frame, text="Padres para apareamiento:").grid(row=3, column=0, padx=5, pady=3, sticky='w')
        ttk.Entry(num_frame, textvariable=self.exp_parents_min, width=8).grid(row=3, column=1, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_parents_max, width=8).grid(row=3, column=2, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_parents_paso, width=8).grid(row=3, column=3, padx=3, pady=3)
        self.lbl_exp_parents = ttk.Label(num_frame, text="[4]", foreground='blue')
        self.lbl_exp_parents.grid(row=3, column=4, padx=5, pady=3)
        
        # % Mutación
        self.exp_mut_min = tk.IntVar(value=10)
        self.exp_mut_max = tk.IntVar(value=10)
        self.exp_mut_paso = tk.IntVar(value=5)
        
        ttk.Label(num_frame, text="% Mutación:").grid(row=4, column=0, padx=5, pady=3, sticky='w')
        ttk.Entry(num_frame, textvariable=self.exp_mut_min, width=8).grid(row=4, column=1, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_mut_max, width=8).grid(row=4, column=2, padx=3, pady=3)
        ttk.Entry(num_frame, textvariable=self.exp_mut_paso, width=8).grid(row=4, column=3, padx=3, pady=3)
        self.lbl_exp_mut = ttk.Label(num_frame, text="[10]", foreground='blue')
        self.lbl_exp_mut.grid(row=4, column=4, padx=5, pady=3)
        
        # Botón para previsualizar valores
        ttk.Button(num_frame, text="Previsualizar valores", 
                  command=self.previsualizar_valores_exp).grid(row=5, column=0, columnspan=5, pady=8)
        
        # ============================================================
        # Parámetros categóricos (checkboxes múltiples)
        # ============================================================
        cat_frame = ttk.LabelFrame(main_frame, text="Parámetros categóricos (seleccionar opciones a probar)", padding=10)
        cat_frame.pack(fill='x', pady=5)
        
        # --- Selección de padres ---
        ttk.Label(cat_frame, text="Selección de padres:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, padx=5, pady=3, sticky='w')
        
        self.exp_parent_types = {}
        parent_options = ["sss", "rws", "sus", "rank", "random", "tournament"]
        for i, opt in enumerate(parent_options):
            var = tk.BooleanVar(value=(opt == "rank"))
            self.exp_parent_types[opt] = var
            ttk.Checkbutton(cat_frame, text=opt, variable=var).grid(
                row=0, column=i+1, padx=4, pady=3)
        
        # --- Tipo de cruce ---
        ttk.Label(cat_frame, text="Tipo de cruce:", font=('Arial', 9, 'bold')).grid(
            row=1, column=0, padx=5, pady=3, sticky='w')
        
        self.exp_crossover_types = {}
        crossover_options = ["single_point", "two_points", "uniform", "scattered"]
        for i, opt in enumerate(crossover_options):
            var = tk.BooleanVar(value=(opt == "single_point"))
            self.exp_crossover_types[opt] = var
            ttk.Checkbutton(cat_frame, text=opt, variable=var).grid(
                row=1, column=i+1, padx=4, pady=3)
        
        # --- Tipo de mutación ---
        ttk.Label(cat_frame, text="Tipo de mutación:", font=('Arial', 9, 'bold')).grid(
            row=2, column=0, padx=5, pady=3, sticky='w')
        
        self.exp_mutation_types = {}
        mutation_options = ["random", "swap", "inversion", "scramble", "adaptive"]
        for i, opt in enumerate(mutation_options):
            var = tk.BooleanVar(value=(opt == "random"))
            self.exp_mutation_types[opt] = var
            ttk.Checkbutton(cat_frame, text=opt, variable=var).grid(
                row=2, column=i+1, padx=4, pady=3)
        
        # ============================================================
        # Resumen y ejecución
        # ============================================================
        exec_frame = ttk.LabelFrame(main_frame, text="Ejecución", padding=10)
        exec_frame.pack(fill='x', pady=5)
        
        self.lbl_exp_total_combinaciones = ttk.Label(exec_frame, 
            text="Total de combinaciones: --", font=('Arial', 10, 'bold'))
        self.lbl_exp_total_combinaciones.pack(pady=3)
        
        self.lbl_exp_progreso = ttk.Label(exec_frame, text="", font=('Arial', 9))
        self.lbl_exp_progreso.pack(pady=2)
        
        self.exp_progreso_bar = ttk.Progressbar(exec_frame, length=500, mode='determinate')
        self.exp_progreso_bar.pack(pady=5)
        
        btn_exp_frame = ttk.Frame(exec_frame)
        btn_exp_frame.pack(pady=5)
        
        ttk.Button(btn_exp_frame, text="Calcular combinaciones", 
                  command=self.calcular_combinaciones_exp).pack(side='left', padx=5)
        ttk.Button(btn_exp_frame, text="Ejecutar experimentación", 
                  command=self.ejecutar_experimentacion).pack(side='left', padx=5)
        ttk.Button(btn_exp_frame, text="Detener", 
                  command=self.detener_experimentacion).pack(side='left', padx=5)
        
        # ============================================================
        # Tabla de resultados de experimentación
        # ============================================================
        res_frame = ttk.LabelFrame(main_frame, text="Resultados de experimentación", padding=10)
        res_frame.pack(fill='both', expand=True, pady=5)
        
        # Treeview con scrollbars
        tree_exp_frame = ttk.Frame(res_frame)
        tree_exp_frame.pack(fill='both', expand=True)
        
        scroll_y_exp = ttk.Scrollbar(tree_exp_frame, orient='vertical')
        scroll_x_exp = ttk.Scrollbar(tree_exp_frame, orient='horizontal')
        
        cols_exp = ('corrida', 'metodo', 'generaciones', 'poblacion', 'padres', 
                   'mutacion_pct', 'sel_padres', 'cruce', 'mutacion', 
                   'var_final', 'tiempo_s')
        
        self.tree_exp = ttk.Treeview(tree_exp_frame, columns=cols_exp, show='headings',
                                      yscrollcommand=scroll_y_exp.set,
                                      xscrollcommand=scroll_x_exp.set, height=10)
        
        scroll_y_exp.config(command=self.tree_exp.yview)
        scroll_x_exp.config(command=self.tree_exp.xview)
        
        # Configurar columnas
        col_config = {
            'corrida': ('N°', 40), 'metodo': ('Método', 65),
            'generaciones': ('Generac.', 65), 'poblacion': ('Población', 65),
            'padres': ('Padres', 55), 'mutacion_pct': ('Mut.%', 45),
            'sel_padres': ('Sel. padres', 80), 'cruce': ('Cruce', 85),
            'mutacion': ('Mutación', 75), 'var_final': ('Var. final', 80),
            'tiempo_s': ('Tiempo(s)', 70)
        }
        
        for col_id, (heading, width) in col_config.items():
            self.tree_exp.heading(col_id, text=heading, 
                                command=lambda c=col_id: self.ordenar_tabla_exp(c))
            self.tree_exp.column(col_id, width=width, minwidth=40)
        
        self.tree_exp.grid(row=0, column=0, sticky='nsew')
        scroll_y_exp.grid(row=0, column=1, sticky='ns')
        scroll_x_exp.grid(row=1, column=0, sticky='ew')
        
        tree_exp_frame.grid_rowconfigure(0, weight=1)
        tree_exp_frame.grid_columnconfigure(0, weight=1)
        
        # Botones de resultados
        btn_res_frame = ttk.Frame(res_frame)
        btn_res_frame.pack(pady=5)
        
        ttk.Button(btn_res_frame, text="Graficar mejores corridas", 
                  command=self.graficar_experimentacion).pack(side='left', padx=5)
        ttk.Button(btn_res_frame, text="Exportar experimentación a Excel",
                  command=self.exportar_experimentacion).pack(side='left', padx=5)
        ttk.Button(btn_res_frame, text="Limpiar resultados",
                  command=self.limpiar_experimentacion).pack(side='left', padx=5)
        
        # Gráfica
        self.fig_exp = Figure(figsize=(10, 5))
        self.canvas_exp = FigureCanvasTkAgg(self.fig_exp, main_frame)
        self.canvas_exp.get_tk_widget().pack(fill='both', expand=True, pady=5)
    
    def _generar_rango(self, vmin, vmax, paso):
        """Genera lista de valores desde vmin hasta vmax con paso dado"""
        if paso <= 0:
            return [vmin]
        valores = []
        v = vmin
        while v <= vmax:
            valores.append(v)
            v += paso
        if len(valores) == 0:
            valores = [vmin]
        return valores
    
    def previsualizar_valores_exp(self):
        """Muestra los valores que se generarán para cada parámetro numérico"""
        gen_vals = self._generar_rango(self.exp_gen_min.get(), self.exp_gen_max.get(), self.exp_gen_paso.get())
        pop_vals = self._generar_rango(self.exp_pop_min.get(), self.exp_pop_max.get(), self.exp_pop_paso.get())
        par_vals = self._generar_rango(self.exp_parents_min.get(), self.exp_parents_max.get(), self.exp_parents_paso.get())
        mut_vals = self._generar_rango(self.exp_mut_min.get(), self.exp_mut_max.get(), self.exp_mut_paso.get())
        
        self.lbl_exp_gen.config(text=str(gen_vals))
        self.lbl_exp_pop.config(text=str(pop_vals))
        self.lbl_exp_parents.config(text=str(par_vals))
        self.lbl_exp_mut.config(text=str(mut_vals))
        
        self.calcular_combinaciones_exp()
    
    def calcular_combinaciones_exp(self):
        """Calcula y muestra el total de combinaciones"""
        gen_vals = self._generar_rango(self.exp_gen_min.get(), self.exp_gen_max.get(), self.exp_gen_paso.get())
        pop_vals = self._generar_rango(self.exp_pop_min.get(), self.exp_pop_max.get(), self.exp_pop_paso.get())
        par_vals = self._generar_rango(self.exp_parents_min.get(), self.exp_parents_max.get(), self.exp_parents_paso.get())
        mut_vals = self._generar_rango(self.exp_mut_min.get(), self.exp_mut_max.get(), self.exp_mut_paso.get())
        
        sel_tipos = [k for k, v in self.exp_parent_types.items() if v.get()]
        cross_tipos = [k for k, v in self.exp_crossover_types.items() if v.get()]
        mut_tipos = [k for k, v in self.exp_mutation_types.items() if v.get()]
        
        if not sel_tipos: sel_tipos = ["rank"]
        if not cross_tipos: cross_tipos = ["single_point"]
        if not mut_tipos: mut_tipos = ["random"]
        
        total = len(gen_vals) * len(pop_vals) * len(par_vals) * len(mut_vals) * \
                len(sel_tipos) * len(cross_tipos) * len(mut_tipos)
        
        self.lbl_exp_total_combinaciones.config(
            text=f"Total de combinaciones: {total}",
            foreground='red' if total > 50 else 'black'
        )
        return total
    
    def detener_experimentacion(self):
        """Detiene la experimentación en curso"""
        self.exp_corriendo = False
    
    def limpiar_experimentacion(self):
        """Limpia los resultados de experimentación"""
        self.resultados_experimentacion = []
        for item in self.tree_exp.get_children():
            self.tree_exp.delete(item)
        self.fig_exp.clear()
        self.canvas_exp.draw()
        self.lbl_exp_progreso.config(text="")
        self.exp_progreso_bar['value'] = 0
    
    def ordenar_tabla_exp(self, col):
        """Ordena la tabla de experimentación por la columna seleccionada"""
        items = [(self.tree_exp.set(item, col), item) for item in self.tree_exp.get_children()]
        
        # Intentar ordenar numéricamente
        try:
            items.sort(key=lambda t: float(t[0]))
        except ValueError:
            items.sort(key=lambda t: t[0])
        
        for index, (val, item) in enumerate(items):
            self.tree_exp.move(item, '', index)
    
    def ejecutar_experimentacion(self):
        """Ejecuta todas las combinaciones de parámetros del AG"""
        if not self.validar_datos():
            return
        
        # Generar listas de valores
        gen_vals = self._generar_rango(self.exp_gen_min.get(), self.exp_gen_max.get(), self.exp_gen_paso.get())
        pop_vals = self._generar_rango(self.exp_pop_min.get(), self.exp_pop_max.get(), self.exp_pop_paso.get())
        par_vals = self._generar_rango(self.exp_parents_min.get(), self.exp_parents_max.get(), self.exp_parents_paso.get())
        mut_vals = self._generar_rango(self.exp_mut_min.get(), self.exp_mut_max.get(), self.exp_mut_paso.get())
        
        sel_tipos = [k for k, v in self.exp_parent_types.items() if v.get()]
        cross_tipos = [k for k, v in self.exp_crossover_types.items() if v.get()]
        mut_tipos = [k for k, v in self.exp_mutation_types.items() if v.get()]
        
        if not sel_tipos: sel_tipos = ["rank"]
        if not cross_tipos: cross_tipos = ["single_point"]
        if not mut_tipos: mut_tipos = ["random"]
        
        usar_kalman = (self.exp_metodo.get() == "filtro_kalman")
        metodo_str = "Kalman" if usar_kalman else "K.O."
        
        # Generar todas las combinaciones
        combinaciones = list(itertools_product(gen_vals, pop_vals, par_vals, mut_vals,
                                      sel_tipos, cross_tipos, mut_tipos))
        
        total = len(combinaciones)
        if total == 0:
            messagebox.showwarning("Advertencia", "No hay combinaciones para ejecutar")
            return
        
        if total > 100:
            resp = messagebox.askyesno("Confirmación", 
                f"Se ejecutarán {total} combinaciones.\nEsto puede tomar mucho tiempo.\n¿Continuar?")
            if not resp:
                return
        
        # Preparar datos
        max_pts = self.exp_max_puntos.get()
        x_muestreo = self.df_muestreo['x'].values
        y_muestreo = self.df_muestreo['y'].values
        z_muestreo = self.obtener_datos_transformados()
        x_malla = self.df_malla['x'].values
        y_malla = self.df_malla['y'].values
        n_grid = len(x_malla)
        pesos = self.df_malla['peso'].values if 'peso' in self.df_malla.columns else None
        
        # Pesos de pozos
        pp_exp = self._get_pesos_pozos()
        
        # Puntos fijos/opcionales
        if self.usar_opcionales.get() and len(self.puntos_fijos) > 0:
            indices_fijos = self.puntos_fijos
            indices_opcionales = self.puntos_opcionales
            x_fijos = x_muestreo[indices_fijos]
            y_fijos = y_muestreo[indices_fijos]
            z_fijos = z_muestreo[indices_fijos]
            x_pool = x_muestreo[indices_opcionales]
            y_pool = y_muestreo[indices_opcionales]
            z_pool = z_muestreo[indices_opcionales]
            num_fijos = len(indices_fijos)
        else:
            indices_fijos = []
            indices_opcionales = list(range(len(x_muestreo)))
            x_fijos, y_fijos, z_fijos = np.array([]), np.array([]), np.array([])
            x_pool = x_muestreo
            y_pool = y_muestreo
            z_pool = z_muestreo
            num_fijos = 0
        
        max_adicionales = min(max_pts - num_fijos, len(indices_opcionales)) if num_fijos > 0 \
                          else min(max_pts, len(indices_opcionales))
        
        # Precomputar matrices si se usa Kalman
        C_gc_full = None
        C_cc_full = None
        var_value_ks = None
        R_ks = None
        
        if usar_kalman:
            var_value_ks = self.var_sill.get()
            nugget_ks = self.var_nugget.get()
            range_ks = self.var_range.get()
            R_ks = nugget_ks if nugget_ks > 0 else 1e-6
            
            model_ks = self.crear_modelo()
            coords_grid = np.column_stack([x_malla, y_malla])
            coords_cand = np.column_stack([x_muestreo, y_muestreo])
            C_gc_full = model_ks.covariance(cdist(coords_grid, coords_cand)).astype(np.float64)
            C_cc_full = model_ks.covariance(cdist(coords_cand, coords_cand)).astype(np.float64)
        
        # Limpiar tabla
        for item in self.tree_exp.get_children():
            self.tree_exp.delete(item)
        
        self.resultados_experimentacion = []
        self.exp_corriendo = True
        self.exp_progreso_bar['maximum'] = total
        self.exp_progreso_bar['value'] = 0
        
        n_corrida_base = 0
        
        for idx_comb, (n_gen, n_pop, n_par, mut_pct, sel_t, cross_t, mut_t) in enumerate(combinaciones):
            if not self.exp_corriendo:
                self.lbl_exp_progreso.config(text=f"Detenido en corrida {idx_comb}/{total}")
                break
            
            self.lbl_exp_progreso.config(
                text=f"Corrida {idx_comb + 1}/{total}: gen={n_gen}, pop={n_pop}, "
                     f"par={n_par}, mut={mut_pct}%, sel={sel_t}, cruce={cross_t}, mut={mut_t}")
            self.exp_progreso_bar['value'] = idx_comb
            self.root.update()
            
            # Validar que padres <= población
            n_par_real = min(n_par, n_pop)
            
            try:
                tiempo_ini = time.time()
                
                varianzas_corrida = [n_grid]  # varianza inicial
                num_puntos_corrida = [0]
                
                if num_fijos > 0:
                    var_fijos = self.calcular_varianza_kriging(
                        x_fijos, y_fijos, z_fijos, x_malla, y_malla, pesos)
                    varianzas_corrida.append(var_fijos)
                    num_puntos_corrida.append(num_fijos)
                
                for n_pts_adic in range(1, max_adicionales + 1):
                    if not self.exp_corriendo:
                        break
                    
                    # Definir fitness según método
                    if usar_kalman:
                        def fitness_func(ga_instance, solution, solution_idx):
                            idx_sel = np.argsort(solution)[::-1][:n_pts_adic]
                            if len(idx_sel) == 0:
                                return -1e10
                            idx_orig = [indices_opcionales[i] for i in idx_sel]
                            todos = list(indices_fijos) + idx_orig if num_fijos > 0 else idx_orig
                            v = self.calcular_varianza_kriging_simple(
                                todos, C_gc_full, C_cc_full, var_value_ks, R_ks, n_grid, pesos)
                            if pp_exp is not None:
                                mean_w = np.mean([pp_exp[i] for i in idx_orig])
                                return -v / mean_w
                            return -v
                    else:
                        def fitness_func(ga_instance, solution, solution_idx):
                            idx_sel = np.argsort(solution)[::-1][:n_pts_adic]
                            if len(idx_sel) == 0:
                                return -1e10
                            if num_fijos > 0:
                                x_s = np.concatenate([x_fijos, x_pool[idx_sel]])
                                y_s = np.concatenate([y_fijos, y_pool[idx_sel]])
                                z_s = np.concatenate([z_fijos, z_pool[idx_sel]])
                            else:
                                x_s = x_pool[idx_sel]
                                y_s = y_pool[idx_sel]
                                z_s = z_pool[idx_sel]
                            v = self.calcular_varianza_kriging(
                                x_s, y_s, z_s, x_malla, y_malla, pesos)
                            if pp_exp is not None:
                                idx_orig = [indices_opcionales[i] for i in idx_sel]
                                mean_w = np.mean([pp_exp[i] for i in idx_orig])
                                return -v / mean_w
                            return -v
                    
                    num_genes = len(x_pool)
                    
                    # Población inicial con semilla secuencial si está disponible
                    initial_population = None
                    if self.resultados_secuencial:
                        initial_population = np.random.uniform(0.0, 1.0, (n_pop, num_genes))
                        sol_sec = self.crear_solucion_inicial_desde_secuencial(
                            num_genes, n_pts_adic, indices_opcionales, num_fijos)
                        initial_population[0] = sol_sec
                        for ii in range(1, min(3, n_pop // 4)):
                            var_sol = sol_sec + np.random.normal(0, 0.1, num_genes)
                            initial_population[ii] = np.clip(var_sol, 0.0, 1.0)
                    
                    ga_instance = pygad.GA(
                        num_generations=n_gen,
                        num_parents_mating=n_par_real,
                        fitness_func=fitness_func,
                        sol_per_pop=n_pop,
                        num_genes=num_genes,
                        gene_type=float,
                        initial_population=initial_population,
                        parent_selection_type=sel_t,
                        keep_parents=min(2, n_par_real),
                        crossover_type=cross_t,
                        mutation_type=mut_t,
                        mutation_percent_genes=mut_pct,
                        random_seed=42,
                        suppress_warnings=True
                    )
                    
                    ga_instance.run()
                    
                    solution, solution_fitness, _ = ga_instance.best_solution()
                    idx_sel_pool = np.argsort(solution)[::-1][:n_pts_adic]
                    
                    if num_fijos > 0:
                        idx_reales = indices_fijos + [indices_opcionales[i] for i in idx_sel_pool]
                    else:
                        idx_reales = [indices_opcionales[i] for i in idx_sel_pool]
                    
                    # Varianza SIEMPRE reportada con kriging ordinario (real, sin pesos de pozo)
                    x_ok = x_muestreo[idx_reales]
                    y_ok = y_muestreo[idx_reales]
                    z_ok = z_muestreo[idx_reales]
                    var_ok = self.calcular_varianza_kriging(
                        x_ok, y_ok, z_ok, x_malla, y_malla, pesos)
                    
                    varianzas_corrida.append(var_ok)
                    num_puntos_corrida.append(num_fijos + n_pts_adic)
                
                tiempo_total = time.time() - tiempo_ini
                
                # Guardar resultado
                var_final = varianzas_corrida[-1] if len(varianzas_corrida) > 1 else np.inf
                
                resultado = {
                    'corrida': idx_comb + 1,
                    'metodo': metodo_str,
                    'generaciones': n_gen,
                    'poblacion': n_pop,
                    'padres': n_par_real,
                    'mutacion_pct': mut_pct,
                    'sel_padres': sel_t,
                    'cruce': cross_t,
                    'mutacion': mut_t,
                    'var_final': var_final,
                    'tiempo': tiempo_total,
                    'num_puntos': num_puntos_corrida,
                    'varianzas': varianzas_corrida
                }
                
                self.resultados_experimentacion.append(resultado)
                
                # Insertar en treeview
                self.tree_exp.insert('', 'end', values=(
                    idx_comb + 1, metodo_str, n_gen, n_pop, n_par_real, mut_pct,
                    sel_t, cross_t, mut_t,
                    f"{var_final:.4f}", f"{tiempo_total:.2f}"
                ))
                
            except Exception as e:
                # Registrar el error pero continuar con la siguiente combinación
                resultado = {
                    'corrida': idx_comb + 1,
                    'metodo': metodo_str,
                    'generaciones': n_gen,
                    'poblacion': n_pop,
                    'padres': n_par_real,
                    'mutacion_pct': mut_pct,
                    'sel_padres': sel_t,
                    'cruce': cross_t,
                    'mutacion': mut_t,
                    'var_final': np.inf,
                    'tiempo': 0,
                    'num_puntos': [0],
                    'varianzas': [n_grid],
                    'error': str(e)
                }
                self.resultados_experimentacion.append(resultado)
                
                self.tree_exp.insert('', 'end', values=(
                    idx_comb + 1, metodo_str, n_gen, n_pop, n_par_real, mut_pct,
                    sel_t, cross_t, mut_t, "ERROR", "0"
                ))
        
        self.exp_corriendo = False
        self.exp_progreso_bar['value'] = total
        
        # Resumen
        n_exitosas = sum(1 for r in self.resultados_experimentacion if r['var_final'] != np.inf)
        
        if n_exitosas > 0:
            mejor = min(self.resultados_experimentacion, key=lambda r: r['var_final'])
            self.lbl_exp_progreso.config(
                text=f"Completado: {n_exitosas}/{len(self.resultados_experimentacion)} corridas exitosas | "
                     f"Mejor var. final: {mejor['var_final']:.4f} (corrida {mejor['corrida']})")
            
            messagebox.showinfo("Experimentación completada",
                f"Corridas completadas: {n_exitosas}/{total}\n\n"
                f"Mejor configuración (corrida {mejor['corrida']}):\n"
                f"  Generaciones: {mejor['generaciones']}\n"
                f"  Población: {mejor['poblacion']}\n"
                f"  Padres: {mejor['padres']}\n"
                f"  Mutación: {mejor['mutacion_pct']}%\n"
                f"  Sel. padres: {mejor['sel_padres']}\n"
                f"  Cruce: {mejor['cruce']}\n"
                f"  Mutación: {mejor['mutacion']}\n"
                f"  Varianza final: {mejor['var_final']:.6f}\n"
                f"  Tiempo: {mejor['tiempo']:.2f}s")
        else:
            self.lbl_exp_progreso.config(text="Ninguna corrida completada exitosamente")
            messagebox.showwarning("Advertencia", "Ninguna corrida fue exitosa")
    
    def graficar_experimentacion(self):
        """Grafica las mejores corridas de la experimentación"""
        if not self.resultados_experimentacion:
            messagebox.showwarning("Advertencia", "No hay resultados de experimentación")
            return
        
        # Filtrar corridas exitosas y ordenar por varianza final
        exitosas = [r for r in self.resultados_experimentacion if r['var_final'] != np.inf]
        if not exitosas:
            messagebox.showwarning("Advertencia", "No hay corridas exitosas")
            return
        
        exitosas.sort(key=lambda r: r['var_final'])
        
        # Graficar las mejores N (máximo 10)
        n_mostrar = min(10, len(exitosas))
        
        self.fig_exp.clear()
        ax = self.fig_exp.add_subplot(111)
        
        colores = plt.cm.tab10(np.linspace(0, 1, n_mostrar))
        marcadores = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
        
        for i, res in enumerate(exitosas[:n_mostrar]):
            etiqueta = (f"#{res['corrida']} gen={res['generaciones']} "
                       f"pop={res['poblacion']} {res['sel_padres']}/{res['cruce']}/{res['mutacion']}")
            ax.plot(res['num_puntos'], res['varianzas'],
                   color=colores[i], marker=marcadores[i % len(marcadores)],
                   linewidth=1.5, markersize=4, label=etiqueta, alpha=0.8)
        
        # Agregar secuencial si existe para referencia
        if self.resultados_secuencial:
            ax.plot(self.resultados_secuencial['num_puntos'],
                   self.resultados_secuencial['varianzas'],
                   'k--', linewidth=2, alpha=0.5, label='Inclusiones sucesivas (ref.)')
        
        ax.set_xlabel('Número de puntos de muestreo', fontsize=11)
        ax.set_ylabel('Varianza normalizada (Kriging Ordinario)', fontsize=11)
        ax.set_title(f'Mejores {n_mostrar} configuraciones del AG', fontsize=13, fontweight='bold')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        
        self.canvas_exp.draw()
    
    def exportar_experimentacion(self):
        """Exporta los resultados de experimentación a Excel"""
        if not self.resultados_experimentacion:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Guardar experimentación",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if not filename:
            return
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Hoja de resumen de todas las corridas
                resumen_rows = []
                for r in self.resultados_experimentacion:
                    resumen_rows.append({
                        'Corrida': r['corrida'],
                        'Método fitness': r['metodo'],
                        'Generaciones': r['generaciones'],
                        'Población': r['poblacion'],
                        'Padres': r['padres'],
                        'Mutación %': r['mutacion_pct'],
                        'Sel. padres': r['sel_padres'],
                        'Tipo cruce': r['cruce'],
                        'Tipo mutación': r['mutacion'],
                        'Varianza final (K.O.)': r['var_final'] if r['var_final'] != np.inf else 'ERROR',
                        'Tiempo (s)': round(r['tiempo'], 2),
                        'Error': r.get('error', '')
                    })
                
                df_resumen = pd.DataFrame(resumen_rows)
                df_resumen.to_excel(writer, sheet_name='Resumen corridas', index=False)
                
                # Hoja con curvas de varianza de las mejores 10 corridas
                exitosas = [r for r in self.resultados_experimentacion if r['var_final'] != np.inf]
                exitosas.sort(key=lambda r: r['var_final'])
                
                if exitosas:
                    # Encontrar la longitud máxima de num_puntos
                    max_len = max(len(r['num_puntos']) for r in exitosas[:10])
                    
                    curvas_data = {'Num_Puntos': list(range(max_len))}
                    
                    for i, r in enumerate(exitosas[:10]):
                        col_name = (f"#{r['corrida']}_gen{r['generaciones']}_"
                                   f"pop{r['poblacion']}_{r['sel_padres']}")
                        # Rellenar con NaN si es más corta
                        vals = list(r['varianzas']) + [np.nan] * (max_len - len(r['varianzas']))
                        curvas_data[col_name] = vals[:max_len]
                    
                    # Agregar secuencial como referencia
                    if self.resultados_secuencial:
                        ref = list(self.resultados_secuencial['varianzas'])
                        ref_padded = ref + [np.nan] * (max_len - len(ref))
                        curvas_data['Inclusiones_sucesivas'] = ref_padded[:max_len]
                    
                    df_curvas = pd.DataFrame(curvas_data)
                    df_curvas.to_excel(writer, sheet_name='Curvas mejores 10', index=False)
                
                # Hoja de parámetros de experimentación
                params_exp = {
                    'Parámetro': [
                        'Método de fitness',
                        'Modelo de variograma',
                        'Transformación de datos',
                        'Máx. puntos',
                        'Generaciones (min, max, paso)',
                        'Población (min, max, paso)',
                        'Padres (min, max, paso)',
                        'Mutación % (min, max, paso)',
                        'Tipos selección padres',
                        'Tipos cruce',
                        'Tipos mutación',
                        'Total combinaciones',
                        'Corridas exitosas',
                        'Malla ponderada',
                        'Pesos de pozos'
                    ],
                    'Valor': [
                        self.exp_metodo.get(),
                        self.info_modelo(),
                        self.info_transformacion(),
                        self.exp_max_puntos.get(),
                        f"{self.exp_gen_min.get()}, {self.exp_gen_max.get()}, {self.exp_gen_paso.get()}",
                        f"{self.exp_pop_min.get()}, {self.exp_pop_max.get()}, {self.exp_pop_paso.get()}",
                        f"{self.exp_parents_min.get()}, {self.exp_parents_max.get()}, {self.exp_parents_paso.get()}",
                        f"{self.exp_mut_min.get()}, {self.exp_mut_max.get()}, {self.exp_mut_paso.get()}",
                        ', '.join(k for k, v in self.exp_parent_types.items() if v.get()),
                        ', '.join(k for k, v in self.exp_crossover_types.items() if v.get()),
                        ', '.join(k for k, v in self.exp_mutation_types.items() if v.get()),
                        len(self.resultados_experimentacion),
                        sum(1 for r in self.resultados_experimentacion if r['var_final'] != np.inf),
                        'Sí' if self.malla_ponderada else 'No',
                        f"Sí (col: {self.col_peso_pozo})" if self._get_pesos_pozos() is not None else 'No'
                    ]
                }
                df_params = pd.DataFrame(params_exp)
                df_params.to_excel(writer, sheet_name='Parámetros experimento', index=False)
            
            messagebox.showinfo("Éxito", f"Experimentación exportada a:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar: {str(e)}")
    
    def crear_tab_resultados(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Frame superior con información y botones
        frame_superior = ttk.Frame(frame)
        frame_superior.pack(fill='x', pady=5)
        
        # Labels de información
        self.lbl_tiempo_secuencial = ttk.Label(frame_superior, text="Tiempo inclusiones sucesivas: --", 
                                                font=('Arial', 10))
        self.lbl_tiempo_secuencial.pack(side='left', padx=10)
        
        self.lbl_tiempo_genetico = ttk.Label(frame_superior, text="Tiempo algoritmo genético: --", 
                                              font=('Arial', 10))
        self.lbl_tiempo_genetico.pack(side='left', padx=10)
        
        self.lbl_tiempo_kalman = ttk.Label(frame_superior, text="Tiempo filtro de Kalman: --", 
                                            font=('Arial', 10))
        self.lbl_tiempo_kalman.pack(side='left', padx=10)
        
        # Botón de exportar
        ttk.Button(frame_superior, text="Exportar resultados a Excel", 
                  command=self.exportar_resultados).pack(side='right', padx=10)
        
        # Gráfica comparativa
        self.fig_resultados = Figure(figsize=(12, 6))
        self.canvas_resultados = FigureCanvasTkAgg(self.fig_resultados, frame)
        self.canvas_resultados.get_tk_widget().pack(fill='both', expand=True)
    
    def cargar_muestreo(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de muestreo",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if filename:
            try:
                self.df_muestreo = pd.read_excel(filename)
                # Identificar columna del parámetro (columna posterior a las coordenadas x, y)
                cols = self.df_muestreo.columns.tolist()
                
                # Buscar las posiciones de x e y
                if 'x' in cols and 'y' in cols:
                    idx_x = cols.index('x')
                    idx_y = cols.index('y')
                    # Encontrar la posición mayor entre x e y
                    idx_max = max(idx_x, idx_y)
                    # La columna del parámetro es la siguiente después de las coordenadas
                    if idx_max + 1 < len(cols):
                        # Detectar columna de peso de pozo (si existe)
                        peso_pozo_keywords = ['peso_pozo', 'well_weight', 'w_pozo', 'peso_w', 
                                              'weight_well', 'pozo_peso']
                        self.col_peso_pozo = None
                        self.pesos_pozos = None
                        
                        for c in cols:
                            if c.lower().strip() in peso_pozo_keywords:
                                self.col_peso_pozo = c
                                break
                        
                        if self.col_peso_pozo is not None:
                            pw = self.df_muestreo[self.col_peso_pozo].values.astype(float)
                            if np.any(pw <= 0):
                                pw = np.where(pw <= 0, np.min(pw[pw > 0]) if np.any(pw > 0) else 1.0, pw)
                            self.pesos_pozos = pw
                            self.usar_pesos_pozos.set(True)
                        else:
                            self.pesos_pozos = np.ones(len(self.df_muestreo))
                            self.usar_pesos_pozos.set(False)
                        
                        # Detectar TODAS las columnas numéricas después de x, y
                        # EXCLUYENDO la columna de peso de pozo
                        param_cols = [c for i, c in enumerate(cols) if i > idx_max
                                      and pd.api.types.is_numeric_dtype(self.df_muestreo[c])
                                      and c != self.col_peso_pozo]
                        if not param_cols:
                            messagebox.showwarning("Advertencia", 
                                "No se encontraron columnas numéricas de parámetro")
                            return
                        self.parametros_disponibles = param_cols
                        self.parametro_col = param_cols[0]
                    else:
                        messagebox.showwarning("Advertencia", 
                            "No se encontró columna de parámetro después de las coordenadas")
                        return
                else:
                    messagebox.showerror("Error", 
                        "El archivo debe contener columnas 'x' e 'y'")
                    return
                
                # Mostrar datos en Treeview
                self.tree_datos['columns'] = cols
                self.tree_datos['show'] = 'headings'
                
                for col in cols:
                    self.tree_datos.heading(col, text=col)
                    self.tree_datos.column(col, width=100)

                self.puntos_fijos = []
                self.puntos_opcionales = list(range(len(self.df_muestreo)))
                
                # Limpiar datos anteriores
                for item in self.tree_datos.get_children():
                    self.tree_datos.delete(item)
                
                # Insertar datos
                for idx, row in self.df_muestreo.iterrows():
                    self.tree_datos.insert('', 'end', values=list(row))
                
                # Actualizar información
                params_str = ', '.join(self.parametros_disponibles)
                self.lbl_info_datos.config(
                    text=f"Archivo cargado: {len(self.df_muestreo)} puntos | "
                         f"Parámetros ({len(self.parametros_disponibles)}): {params_str}"
                )
                
                # Actualizar info de pesos de pozos
                if self.col_peso_pozo is not None:
                    pw = self.pesos_pozos
                    self.lbl_pesos_pozos.config(
                        text=f"Columna '{self.col_peso_pozo}' detectada | "
                             f"min={pw.min():.3f}, max={pw.max():.3f}, promedio={pw.mean():.3f}",
                        foreground='blue')
                else:
                    self.lbl_pesos_pozos.config(
                        text="Sin columna de pesos detectada (todos los pozos con peso=1.0)",
                        foreground='gray')
                
                # Inicializar configuración por parámetro
                self.parametros_config = {}
                for pc in self.parametros_disponibles:
                    z_raw = self.df_muestreo[pc].values.astype(float)
                    self.parametros_config[pc] = {
                        'activo': True, 'peso': 1.0,
                        'var_range': 100.0, 'var_sill': 1.0, 'var_nugget': 0.1,
                        'tipo_modelo': 'Spherical', 'matern_nu': 1.5,
                        'tipo_transformacion': 'Ninguna',
                        'z_transformado': z_raw.copy(), 'lambda_boxcox': None
                    }
                
                # Estimar variograma automáticamente para TODOS los parámetros
                self._estimar_todos_parametros(mostrar_msg=False)
                
                # Cargar primer parámetro en los campos de edición
                self._cargar_param_en_edicion(self.parametros_disponibles[0])
                self._refrescar_tabla_parametros()
                
                peso_msg = ""
                if self.col_peso_pozo is not None:
                    peso_msg = f"\n\nPesos de pozos: columna '{self.col_peso_pozo}' detectada"
                
                messagebox.showinfo("Éxito", 
                    f"Datos cargados correctamente\n"
                    f"{len(self.df_muestreo)} puntos\n"
                    f"{len(self.parametros_disponibles)} parámetro(s): {params_str}\n\n"
                    f"Variogramas estimados automáticamente."
                    f"{peso_msg}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar archivo: {str(e)}")
    
    def seleccionar_puntos_fijos(self):
        if self.df_muestreo is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos de muestreo")
            return
        
        # Ventana de selección
        ventana = tk.Toplevel(self.root)
        ventana.title("Seleccionar puntos fijos/opcionales")
        ventana.geometry("600x500")
        
        # Frame principal con scrollbar
        main_frame = ttk.Frame(ventana)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas y scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        ttk.Label(scrollable_frame, text="Marque los puntos fijos (los desmarcados serán opcionales):", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Checkboxes para cada punto
        check_vars = []
        for idx, row in self.df_muestreo.iterrows():
            var = tk.BooleanVar(value=False)
            check_vars.append(var)
            ttk.Checkbutton(scrollable_frame, 
                           text=f"{row['clave']} - ({row['x']:.2f}, {row['y']:.2f})",
                           variable=var).pack(anchor='w', padx=20, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame de botones
        frame_botones = ttk.Frame(ventana)
        frame_botones.pack(pady=10)
        
        def marcar_todos():
            for var in check_vars:
                var.set(True)
        
        def desmarcar_todos():
            for var in check_vars:
                var.set(False)
        
        def guardar_seleccion():
            self.puntos_fijos = []
            self.puntos_opcionales = []
            
            for idx, var in enumerate(check_vars):
                if var.get():
                    self.puntos_fijos.append(idx)
                else:
                    self.puntos_opcionales.append(idx)
            
            messagebox.showinfo("Éxito", 
                f"Puntos fijos: {len(self.puntos_fijos)}\nPuntos opcionales: {len(self.puntos_opcionales)}")
            ventana.destroy()
        
        ttk.Button(frame_botones, text="Marcar todos", command=marcar_todos).pack(side='left', padx=5)
        ttk.Button(frame_botones, text="Desmarcar todos", command=desmarcar_todos).pack(side='left', padx=5)
        ttk.Button(frame_botones, text="Guardar selección", command=guardar_seleccion).pack(side='left', padx=5)
        ttk.Button(frame_botones, text="Cancelar", command=ventana.destroy).pack(side='left', padx=5)
    
    def estimar_parametros(self):
        if self.df_muestreo is None or self.parametro_col is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos de muestreo")
            return
        
        # Asegurar que la transformación esté aplicada
        self.aplicar_transformacion()
        
        try:
            x = self.df_muestreo['x'].values
            y = self.df_muestreo['y'].values
            z = self.obtener_datos_transformados()
            
            # Calcular distancias entre puntos
            coords = np.column_stack([x, y])
            dists = cdist(coords, coords)
            max_dist = np.max(dists[dists > 0])
            
            # Calcular número de bins apropiado según el número de puntos
            n_puntos = len(x)
            if n_puntos < 10:
                n_bins = 3
            elif n_puntos < 20:
                n_bins = 5
            elif n_puntos < 50:
                n_bins = 10
            else:
                n_bins = 15
            
            # Crear bins manualmente
            bins = np.linspace(0, max_dist, n_bins + 1)
            
            try:
                bin_center, gamma = gs.vario_estimate((x, y), z, bin_edges=bins)
            except:
                bin_center, gamma = gs.vario_estimate((x, y), z, bin_edges=n_bins)
            
            # Estimar parámetros
            estimated_range = max_dist * 0.65
            varianza_total = np.var(z)
            
            if len(gamma) > 2:
                estimated_sill = np.mean(gamma[-min(3, len(gamma)):]) * 0.85
            else:
                estimated_sill = varianza_total * 0.7
            
            if len(gamma) > 0:
                estimated_nugget = min(gamma[0], varianza_total * 0.3)
            else:
                estimated_nugget = varianza_total * 0.15
            
            if estimated_nugget + estimated_sill > varianza_total:
                estimated_sill = varianza_total * 0.7
                estimated_nugget = varianza_total * 0.2
            
            self.var_range.set(round(estimated_range, 2))
            self.var_sill.set(round(estimated_sill, 4))
            self.var_nugget.set(round(estimated_nugget, 4))
            
            # Guardar en config del parámetro actual
            self._guardar_edicion_a_config(silencioso=True)
            
            self.recalcular_variograma()
            messagebox.showinfo("Éxito", 
                f"Parámetros estimados para '{self.parametro_actual_nombre or self.parametro_col}'")
        except Exception as e:
            messagebox.showerror("Error", f"Error al estimar parámetros: {str(e)}")
    
    def recalcular_variograma(self):
        if self.df_muestreo is None:
            return
        
        try:
            x = self.df_muestreo['x'].values
            y = self.df_muestreo['y'].values
            z = self.obtener_datos_transformados()
            
            coords = np.column_stack([x, y])
            dists = cdist(coords, coords)
            max_dist = np.max(dists[dists > 0])
            
            n_puntos = len(x)
            if n_puntos < 10:
                n_bins = 3
            elif n_puntos < 20:
                n_bins = 5
            elif n_puntos < 50:
                n_bins = 10
            else:
                n_bins = 15
            
            bins = np.linspace(0, max_dist, n_bins + 1)
            
            try:
                bin_center, gamma = gs.vario_estimate((x, y), z, bin_edges=bins)
            except:
                bin_center, gamma = gs.vario_estimate((x, y), z, bin_edges=n_bins)
            
            var_value = self.var_sill.get()
            nugget_value = self.var_nugget.get()
            range_value = self.var_range.get()
            
            model = self.crear_modelo()
            nombre_modelo = self.info_modelo()
            nombre_transf = self.info_transformacion()
            
            self.fig_vario.clear()
            ax = self.fig_vario.add_subplot(111)
            ax.scatter(bin_center, gamma, label='Variograma experimental', alpha=0.6, s=50, color='blue')
            
            h_range = np.linspace(0, max(bin_center) if len(bin_center) > 0 else range_value*2, 100)
            ax.plot(h_range, model.variogram(h_range), 'r-', 
                   label=f'Modelo: {nombre_modelo}\nRange={range_value:.2f}\nPartial sill={var_value:.4f}\nNugget={nugget_value:.4f}',
                   linewidth=2)
            ax.axhline(y=var_value + nugget_value, color='g', linestyle='--', 
                      label=f'Sill={var_value + nugget_value:.4f}', linewidth=1.5)
            ax.set_xlabel('Distancia', fontsize=11)
            ax.set_ylabel('Semivarianza', fontsize=11)
            
            titulo = f'Variograma - {nombre_modelo}'
            param_name = self.parametro_actual_nombre or self.parametro_col or ''
            if param_name:
                titulo = f'[{param_name}] {titulo}'
            if nombre_transf != "Ninguna":
                titulo += f'  |  Transf: {nombre_transf}'
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            self.canvas_vario.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular variograma: {str(e)}")
    
    def generar_malla(self):
        if self.df_muestreo is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos de muestreo")
            return
        
        try:
            x_muestreo = self.df_muestreo['x'].values
            y_muestreo = self.df_muestreo['y'].values
            
            x_min, x_max = x_muestreo.min(), x_muestreo.max()
            y_min, y_max = y_muestreo.min(), y_muestreo.max()
            
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            x_min -= x_margin
            x_max += x_margin
            y_min -= y_margin
            y_max += y_margin
            
            n_nodos = self.num_nodos_malla.get()
            
            if self.tipo_malla.get() == "rectangular":
                n_x = int(np.sqrt(n_nodos * (x_max - x_min) / (y_max - y_min)))
                n_y = int(n_nodos / n_x)
                
                x_grid = np.linspace(x_min, x_max, n_x)
                y_grid = np.linspace(y_min, y_max, n_y)
                x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
                
                x_flat = x_mesh.flatten()
                y_flat = y_mesh.flatten()
            else:
                puntos_muestreo = np.column_stack([x_muestreo, y_muestreo])
                hull = ConvexHull(puntos_muestreo)
                
                vertices = puntos_muestreo[hull.vertices]
                centroid = np.mean(vertices, axis=0)
                
                buffer_factor = 1.15
                vertices_expandidos = centroid + buffer_factor * (vertices - centroid)
                self.hull_expandido = vertices_expandidos
                
                x_min_exp = vertices_expandidos[:, 0].min()
                x_max_exp = vertices_expandidos[:, 0].max()
                y_min_exp = vertices_expandidos[:, 1].min()
                y_max_exp = vertices_expandidos[:, 1].max()
                
                n_x = int(np.sqrt(n_nodos * (x_max_exp - x_min_exp) / (y_max_exp - y_min_exp)))
                n_y = int(n_nodos / n_x)
                
                x_grid = np.linspace(x_min_exp, x_max_exp, n_x)
                y_grid = np.linspace(y_min_exp, y_max_exp, n_y)
                x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
                
                puntos_malla = np.column_stack([x_mesh.flatten(), y_mesh.flatten()])
                path = Path(vertices_expandidos)
                mask = path.contains_points(puntos_malla)
                x_flat = puntos_malla[mask, 0]
                y_flat = puntos_malla[mask, 1]
            
            self.df_malla = pd.DataFrame({
                'clave': [f'M{i+1}' for i in range(len(x_flat))],
                'x': x_flat,
                'y': y_flat,
                'peso': np.ones(len(x_flat))
            })
            
            self.malla_ponderada = False
            self.pesos_malla = np.ones(len(x_flat))
            
            self.lbl_info_malla.config(
                text=f"Malla generada: {len(self.df_malla)} nodos | Sin pesos (uniforme)",
                foreground='black'
            )
            
            messagebox.showinfo("Éxito", f"Malla generada con {len(self.df_malla)} nodos\n(sin pesos personalizados)")
            self.visualizar_malla()
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar malla: {str(e)}")
    
    def cargar_malla(self):
        """Carga malla desde archivo Excel, detectando columna de pesos si existe"""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de malla",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if filename:
            try:
                df_temp = pd.read_excel(filename)
                
                x_col = None
                y_col = None
                peso_col = None
                clave_col = None
                
                for col in df_temp.columns:
                    col_lower = col.lower().strip()
                    if col_lower == 'x' or col_lower == 'coordenada x':
                        x_col = col
                    elif col_lower == 'y' or col_lower == 'coordenada y':
                        y_col = col
                    elif col_lower == 'peso' or col_lower == 'weight' or col_lower == 'pesos':
                        peso_col = col
                    elif col_lower == 'clave' or col_lower == 'id' or col_lower == 'nombre':
                        clave_col = col
                
                if x_col is None or y_col is None:
                    messagebox.showerror("Error", 
                        "El archivo debe contener columnas 'x' e 'y' (o 'coordenada x', 'coordenada y')")
                    return
                
                n_nodos = len(df_temp)
                
                if clave_col is not None:
                    claves = df_temp[clave_col].astype(str).values
                else:
                    claves = [f'M{i+1}' for i in range(n_nodos)]
                
                if peso_col is not None:
                    pesos = df_temp[peso_col].values.astype(float)
                    
                    if np.any(pesos <= 0):
                        messagebox.showwarning("Advertencia", 
                            "Algunos pesos son <= 0. Se reemplazarán por el valor mínimo positivo.")
                        pesos = np.where(pesos <= 0, np.min(pesos[pesos > 0]), pesos)
                    
                    self.malla_ponderada = True
                    self.pesos_malla = pesos
                    
                    info_pesos = f"Con pesos: mín={pesos.min():.3f}, máx={pesos.max():.3f}, promedio={pesos.mean():.3f}"
                else:
                    pesos = np.ones(n_nodos)
                    self.malla_ponderada = False
                    self.pesos_malla = pesos
                    info_pesos = "Sin pesos (uniforme)"
                
                self.df_malla = pd.DataFrame({
                    'clave': claves,
                    'x': df_temp[x_col].values,
                    'y': df_temp[y_col].values,
                    'peso': pesos
                })
                
                color = 'blue' if self.malla_ponderada else 'black'
                self.lbl_info_malla.config(
                    text=f"Malla cargada: {n_nodos} nodos | {info_pesos}",
                    foreground=color
                )
                
                self.visualizar_malla()
                
                msg = f"Malla cargada con {n_nodos} nodos"
                if self.malla_ponderada:
                    msg += f"\n\n✓ Pesos detectados\n- Mínimo: {pesos.min():.4f}\n- Máximo: {pesos.max():.4f}\n- Promedio: {pesos.mean():.4f}"
                    msg += "\n\nLos nodos con pesos altos tendrán mayor influencia en la priorización de pozos."
                else:
                    msg += "\n\nNo se detectó columna de pesos. Se usarán pesos uniformes."
                
                messagebox.showinfo("Éxito", msg)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar malla: {str(e)}")
    
    def visualizar_malla(self):
        """Visualiza la malla con pesos (si existen) usando colores"""
        if self.df_malla is None or self.df_muestreo is None:
            return
        
        self.fig_malla.clear()
        ax = self.fig_malla.add_subplot(111)
        
        x_malla = self.df_malla['x'].values
        y_malla = self.df_malla['y'].values
        pesos = self.df_malla['peso'].values if 'peso' in self.df_malla.columns else np.ones(len(x_malla))
        
        if self.malla_ponderada and np.std(pesos) > 0:
            scatter = ax.scatter(x_malla, y_malla, c=pesos, cmap='turbo', 
                               s=30, alpha=0.7, label='Malla ponderada')
            cbar = self.fig_malla.colorbar(scatter, ax=ax, label='Peso', shrink=0.8)
            cbar.ax.tick_params(labelsize=8)
            titulo_malla = 'Malla de estimación ponderada'
        else:
            ax.scatter(x_malla, y_malla, c='blue', 
                      s=20, alpha=0.5, label='Malla de estimación')
            titulo_malla = f'Malla de estimación ({self.tipo_malla.get().capitalize()})'
        
        ax.scatter(self.df_muestreo['x'], self.df_muestreo['y'], c='red', 
                  s=50, marker='*', label='Puntos de muestreo', zorder=5)
        
        if self.tipo_malla.get() == "ajustada" and not self.malla_ponderada:
            try:
                x_muestreo = self.df_muestreo['x'].values
                y_muestreo = self.df_muestreo['y'].values
                puntos_muestreo = np.column_stack([x_muestreo, y_muestreo])
                hull = ConvexHull(puntos_muestreo)
                
                hull_points = puntos_muestreo[hull.vertices]
                hull_points_closed = np.vstack([hull_points, hull_points[0]])
                ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 
                       'orange', linewidth=2, linestyle='--', alpha=0.7, 
                       label='Área de muestreo')
                
                if self.hull_expandido is not None:
                    hull_exp_closed = np.vstack([self.hull_expandido, self.hull_expandido[0]])
                    ax.plot(hull_exp_closed[:, 0], hull_exp_closed[:, 1], 
                           'green', linewidth=2.5, alpha=0.8, 
                           label='Área ajustada (con buffer)')
                    ax.fill(hull_exp_closed[:, 0], hull_exp_closed[:, 1], 
                           'green', alpha=0.1)
            except:
                pass
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{titulo_malla} y Puntos de muestreo')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        self.canvas_malla.draw()
    
    # ================================================================
    # Métodos auxiliares: modelo de variograma y transformación
    # ================================================================
    
    def crear_modelo(self):
        """
        Crea y retorna el modelo de variograma según la selección actual.
        Centraliza la creación del modelo para evitar duplicación.
        """
        var_value = self.var_sill.get()
        nugget_value = self.var_nugget.get()
        range_value = self.var_range.get()
        tipo = self.tipo_modelo.get()
        
        params = dict(dim=2, var=var_value, len_scale=range_value, nugget=nugget_value)
        
        modelos = {
            'Spherical': gs.Spherical,
            'Exponential': gs.Exponential,
            'Gaussian': gs.Gaussian,
            'Matérn': gs.Matern,
            'Stable': gs.Stable,
            'Linear': gs.Linear,
            'Cubic': gs.Cubic,
        }
        
        clase = modelos.get(tipo, gs.Spherical)
        
        if tipo == 'Matérn':
            params['nu'] = self.matern_nu.get()
        elif tipo == 'Stable':
            # alpha=2 → Gaussiano, alpha=1 → Exponencial
            params['alpha'] = min(max(self.matern_nu.get(), 0.1), 2.0)
        
        return clase(**params)
    
    def aplicar_transformacion(self):
        """
        Aplica la transformación seleccionada a los datos del parámetro.
        Almacena los datos transformados en self.z_transformado.
        Retorna True si la transformación fue exitosa.
        """
        if self.df_muestreo is None or self.parametro_col is None:
            return False
        
        z_original = self.df_muestreo[self.parametro_col].values.astype(float)
        tipo = self.tipo_transformacion.get()
        
        try:
            if tipo == "Ninguna":
                self.z_transformado = z_original.copy()
                self.lambda_boxcox = None
                
            elif tipo == "Log (ln)":
                if np.any(z_original <= 0):
                    messagebox.showwarning("Advertencia", 
                        "Los datos contienen valores ≤ 0.\n"
                        "Se aplicará desplazamiento: ln(z + |min| + 1)")
                    offset = np.abs(z_original.min()) + 1.0
                    self.z_transformado = np.log(z_original + offset)
                else:
                    self.z_transformado = np.log(z_original)
                self.lambda_boxcox = None
                
            elif tipo == "Log10":
                if np.any(z_original <= 0):
                    messagebox.showwarning("Advertencia", 
                        "Los datos contienen valores ≤ 0.\n"
                        "Se aplicará desplazamiento: log10(z + |min| + 1)")
                    offset = np.abs(z_original.min()) + 1.0
                    self.z_transformado = np.log10(z_original + offset)
                else:
                    self.z_transformado = np.log10(z_original)
                self.lambda_boxcox = None
                
            elif tipo == "Raíz cuadrada":
                if np.any(z_original < 0):
                    messagebox.showwarning("Advertencia", 
                        "Los datos contienen valores negativos.\n"
                        "Se aplicará desplazamiento: sqrt(z + |min|)")
                    offset = np.abs(z_original.min())
                    self.z_transformado = np.sqrt(z_original + offset)
                else:
                    self.z_transformado = np.sqrt(z_original)
                self.lambda_boxcox = None
                
            elif tipo == "Box-Cox":
                if np.any(z_original <= 0):
                    offset = np.abs(z_original.min()) + 1.0
                    z_pos = z_original + offset
                else:
                    z_pos = z_original
                
                self.z_transformado, self.lambda_boxcox = boxcox(z_pos)
                
            elif tipo == "Normal Score":
                # Transformación por rangos a distribución normal estándar
                from scipy.stats import norm
                n = len(z_original)
                ranks = np.argsort(np.argsort(z_original))  # Rangos (0-indexed)
                # Fórmula de Blom para probabilidades de plotting
                prob = (ranks + 0.375) / (n + 0.25)
                self.z_transformado = norm.ppf(prob)
                self.lambda_boxcox = None
            else:
                self.z_transformado = z_original.copy()
                self.lambda_boxcox = None
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar transformación '{tipo}':\n{str(e)}")
            self.z_transformado = z_original.copy()
            self.lambda_boxcox = None
            return False
    
    def obtener_datos_transformados(self):
        """
        Retorna los datos transformados. Si no se ha aplicado transformación,
        retorna los datos originales.
        """
        if self.z_transformado is not None:
            return self.z_transformado
        elif self.df_muestreo is not None and self.parametro_col is not None:
            return self.df_muestreo[self.parametro_col].values.astype(float)
        else:
            return np.array([])
    
    def info_transformacion(self):
        """Retorna cadena descriptiva de la transformación actual"""
        tipo = self.tipo_transformacion.get()
        if tipo == "Box-Cox" and self.lambda_boxcox is not None:
            return f"Box-Cox (λ={self.lambda_boxcox:.4f})"
        return tipo
    
    def info_modelo(self):
        """Retorna cadena descriptiva del modelo actual"""
        tipo = self.tipo_modelo.get()
        if tipo == 'Matérn':
            return f"Matérn (ν={self.matern_nu.get():.2f})"
        elif tipo == 'Stable':
            return f"Stable (α={min(max(self.matern_nu.get(), 0.1), 2.0):.2f})"
        return tipo
    
    def _get_pesos_pozos(self):
        """
        Retorna el array de pesos de pozos si está habilitado, o None.
        Los pesos se usan para ponderar la reducción de varianza de cada pozo
        candidato durante la selección: un pozo con peso 2.0 se considera
        el doble de valioso que uno con peso 1.0 para la priorización.
        """
        if self.usar_pesos_pozos.get() and self.pesos_pozos is not None:
            return self.pesos_pozos
        return None
    
    def calcular_varianza_kriging(self, x_muestra, y_muestra, z_muestra, x_pred, y_pred, pesos=None):
        """
        Calcula la varianza normalizada de kriging en puntos de predicción.
        Si se proporcionan pesos, calcula varianza ponderada.
        """
        try:
            var_value = self.var_sill.get()
            nugget_value = self.var_nugget.get()
            
            sill_total = var_value + nugget_value
            norm_factor = sill_total if sill_total > 0 else 1.0
            
            model = self.crear_modelo()
            krige = gs.krige.Ordinary(model, (x_muestra, y_muestra), z_muestra)
            
            krige((x_pred, y_pred))
            varianzas = krige.krige_var
            
            if pesos is not None and self.malla_ponderada and len(pesos) == len(varianzas):
                pesos_normalizados = pesos * len(pesos) / np.sum(pesos)
                varianza_normalizada = np.sum(varianzas * pesos_normalizados) / norm_factor / 2
            else:
                varianza_normalizada = np.sum(varianzas) / norm_factor / 2
            
            return varianza_normalizada
        except Exception as e:
            return np.inf
    
    def calcular_varianza_kriging_simple(self, indices_obs, C_gc_full, C_cc_full, 
                                          var_value, R, n_grid, pesos=None):
        """
        Calcula la varianza de kriging simple usando matrices de covarianza precomputadas.
        
        Equivale al filtro de Kalman pero evaluando un subconjunto arbitrario
        de observaciones (no secuencial). Mucho más rápido que kriging ordinario
        porque opera sobre matrices ya calculadas.
        
        Args:
            indices_obs: Índices de los puntos de observación seleccionados
            C_gc_full: Matriz completa covarianza malla-candidatos (n_grid, n_candidatos)
            C_cc_full: Matriz completa covarianza candidatos-candidatos (n_candidatos, n_candidatos)
            var_value: Partial sill del variograma
            R: Ruido de observación (nugget)
            n_grid: Número de nodos de la malla
            pesos: Pesos opcionales de la malla
        
        Returns:
            Varianza normalizada (misma escala que kriging ordinario para comparabilidad)
        """
        n_obs = len(indices_obs)
        if n_obs == 0:
            return np.inf
        
        # Extraer sub-matrices para los puntos seleccionados
        C_gc_sel = C_gc_full[:, indices_obs]                                # (n_grid, n_obs)
        C_cc_sel = C_cc_full[np.ix_(indices_obs, indices_obs)]              # (n_obs, n_obs)
        
        # Agregar ruido de medición en la diagonal
        C_cc_R = C_cc_sel + R * np.eye(n_obs)
        
        try:
            # Varianza posterior = var_value - diag(C_gc @ inv(C_cc+R*I) @ C_gc.T)
            # Usando Cholesky: L @ L.T = C_cc_R
            #   inv(C_cc_R) = inv(L.T) @ inv(L)
            #   reduction = ||inv(L) @ C_gc.T||² por columna
            L = np.linalg.cholesky(C_cc_R)
            tmp = np.linalg.solve(L, C_gc_sel.T)          # (n_obs, n_grid)
            var_reduction = np.sum(tmp ** 2, axis=0)       # (n_grid,)
        except np.linalg.LinAlgError:
            # Cholesky falló, usar solver general
            try:
                alpha = np.linalg.solve(C_cc_R, C_gc_sel.T)   # (n_obs, n_grid)
                var_reduction = np.sum(C_gc_sel.T * alpha, axis=0)
            except:
                return np.inf
        
        var_posterior = np.maximum(var_value - var_reduction, 0.0)
        
        # Métrica normalizada (misma convención que kriging ordinario)
        sill_total = var_value + R
        norm_factor = sill_total if sill_total > 0 else 1.0
        
        if pesos is not None and self.malla_ponderada and len(pesos) == n_grid:
            pesos_norm = pesos * n_grid / np.sum(pesos)
            return np.dot(var_posterior, pesos_norm) / norm_factor / 2
        else:
            return np.sum(var_posterior) / norm_factor / 2
    
    def restaurar_parametros_ga(self):
        """Restaura los parámetros de PyGAD a sus valores por defecto"""
        self.ga_num_generations.set(30)
        self.ga_sol_per_pop.set(20)
        self.ga_num_parents_mating.set(4)
        self.ga_mutation_percent.set(10)
        self.ga_parent_selection.set("rank")
        self.ga_crossover_type.set("single_point")
        self.ga_mutation_type.set("random")
        self.ga_metodo_evaluacion.set("kriging_ordinario")
        messagebox.showinfo("Éxito", "Parámetros restaurados a valores por defecto")
    
    def ejecutar_secuencial(self):
        if not self.validar_datos():
            return
        
        try:
            tiempo_inicio = time.time()

            x_muestreo = self.df_muestreo['x'].values
            y_muestreo = self.df_muestreo['y'].values
            z_muestreo = self.obtener_datos_transformados()
            
            x_malla = self.df_malla['x'].values
            y_malla = self.df_malla['y'].values
            
            pesos = self.df_malla['peso'].values if 'peso' in self.df_malla.columns else None

            varianzas = []
            num_puntos = []
            indices_seleccionados_por_iteracion = []
            
            varianza_inicial = len(x_malla)
            
            varianzas.append(varianza_inicial)
            num_puntos.append(0)
                        
            if self.usar_opcionales.get() and len(self.puntos_fijos) > 0:
                indices_fijos = self.puntos_fijos
                indices_opcionales = self.puntos_opcionales
                
                x_fijos = x_muestreo[indices_fijos]
                y_fijos = y_muestreo[indices_fijos]
                z_fijos = z_muestreo[indices_fijos]
                
                num_fijos = len(indices_fijos)
                varianzas.append(self.calcular_varianza_kriging(
                    x_fijos, y_fijos, z_fijos, x_malla, y_malla, pesos))
                num_puntos.append(num_fijos)
            else:
                indices_fijos = []
                indices_opcionales = list(range(len(x_muestreo)))
                x_fijos, y_fijos, z_fijos = np.array([]), np.array([]), np.array([])
                num_fijos = 0
            
            x_pool = x_muestreo
            y_pool = y_muestreo
            z_pool = z_muestreo
            
            n_puntos_total = len(x_muestreo)
            orden_opt = []
            orden_opt.append(indices_fijos.copy() if num_fijos > 0 else [])
            
            # Obtener pesos de pozos
            pp = self._get_pesos_pozos()
            
            for i in range(len(self.puntos_fijos), n_puntos_total):
                mejor_indice = None
                mejor_score = -np.inf
                mejor_varianza_real = np.inf
                
                # Varianza actual (último valor registrado)
                varianza_actual = varianzas[-1]
                
                for j in range(n_puntos_total):
                    if j not in orden_opt[0]:
                        indices_temp = np.array(list(orden_opt[0]) + [j], dtype=int)
                        
                        x_temp = x_pool[indices_temp]
                        y_temp = y_pool[indices_temp]
                        z_temp = z_pool[indices_temp]
                        
                        try:
                            varianza = self.calcular_varianza_kriging(
                                x_temp, y_temp, z_temp, x_malla, y_malla, pesos)
                            
                            # Score = reducción de varianza × peso del pozo
                            reduccion = varianza_actual - varianza
                            peso_j = pp[j] if pp is not None else 1.0
                            score = peso_j * reduccion
                            
                            if score > mejor_score:
                                mejor_score = score
                                mejor_varianza_real = varianza
                                mejor_indice = j
                        except:
                            continue
                
                if mejor_indice is not None:
                    orden_opt[0].append(mejor_indice)
                    varianzas.append(mejor_varianza_real)
                    num_puntos.append(i + 1)
                    
                    self.root.update()
            
            tiempo_fin = time.time()
            self.tiempo_secuencial = tiempo_fin - tiempo_inicio
            
            self.resultados_secuencial = {
                'num_puntos': num_puntos,
                'varianzas': varianzas,
                'indices': orden_opt[0]
            }
            
            msg = f"Priorización inclusiones sucesivas completada\nTiempo: {self.tiempo_secuencial:.2f} segundos"
            if self.malla_ponderada:
                msg += "\n(Usando malla ponderada)"
            if pp is not None:
                msg += "\n(Usando pesos de pozos)"
            
            messagebox.showinfo("Éxito", msg)
            self.graficar_resultados()
        except Exception as e:
            messagebox.showerror("Error", f"Error en priorización inclusiones sucesivas: {str(e)}")
    
    def crear_solucion_inicial_desde_secuencial(self, num_genes, n_pts_adicionales, indices_opcionales, num_fijos):
        """
        Crea una solución inicial basada en el resultado del método secuencial.
        """
        solucion = np.random.uniform(0.0, 0.3, num_genes)
        
        n_pts_totales = num_fijos + n_pts_adicionales
        
        if self.resultados_secuencial and 'indices' in self.resultados_secuencial:
            indices_secuencial = self.resultados_secuencial['indices']
            
            if len(indices_secuencial) >= n_pts_totales:
                indices_seleccionados_secuencial = indices_secuencial[:n_pts_totales]
                
                indices_para_pool = [idx for idx in indices_seleccionados_secuencial 
                                     if idx not in (self.puntos_fijos if num_fijos > 0 else [])]
                
                indices_para_pool = indices_para_pool[:n_pts_adicionales]
                
                for idx_original in indices_para_pool:
                    if idx_original in indices_opcionales:
                        idx_en_pool = indices_opcionales.index(idx_original)
                        if idx_en_pool < num_genes:
                            solucion[idx_en_pool] = np.random.uniform(0.7, 1.0)
        
        return solucion
                
    def ejecutar_genetico(self):
        if not self.validar_datos():
            return
        
        try:
            tiempo_inicio = time.time()
            
            usar_kalman = (self.ga_metodo_evaluacion.get() == "filtro_kalman")
            
            max_pts = self.max_puntos.get()
            
            x_muestreo = self.df_muestreo['x'].values
            y_muestreo = self.df_muestreo['y'].values
            z_muestreo = self.obtener_datos_transformados()
            
            x_malla = self.df_malla['x'].values
            y_malla = self.df_malla['y'].values
            n_grid = len(x_malla)
            
            pesos = self.df_malla['peso'].values if 'peso' in self.df_malla.columns else None
            
            # Pesos de pozos para priorización
            pp = self._get_pesos_pozos()
            
            # ============================================================
            # Precomputar matrices si se usa filtro de Kalman
            # ============================================================
            if usar_kalman:
                var_value = self.var_sill.get()
                nugget_value = self.var_nugget.get()
                range_value = self.var_range.get()
                R = nugget_value if nugget_value > 0 else 1e-6
                
                model_ks = self.crear_modelo()
                
                coords_grid = np.column_stack([x_malla, y_malla])
                coords_cand = np.column_stack([x_muestreo, y_muestreo])
                
                # Covarianza cruzada malla-candidatos
                C_gc_full = model_ks.covariance(cdist(coords_grid, coords_cand)).astype(np.float64)
                # Covarianza candidatos-candidatos
                C_cc_full = model_ks.covariance(cdist(coords_cand, coords_cand)).astype(np.float64)
                
                print("✓ Matrices de covarianza precomputadas para fitness con Kalman/Kriging Simple")

            varianzas = []
            num_puntos = []
            indices_seleccionados_por_iteracion = []
            
            varianza_inicial = n_grid
            
            varianzas.append(varianza_inicial)
            num_puntos.append(0)
                        
            if self.usar_opcionales.get() and len(self.puntos_fijos) > 0:
                indices_fijos = self.puntos_fijos
                indices_opcionales = self.puntos_opcionales
                
                x_fijos = x_muestreo[indices_fijos]
                y_fijos = y_muestreo[indices_fijos]
                z_fijos = z_muestreo[indices_fijos]
                
                x_pool = x_muestreo[indices_opcionales]
                y_pool = y_muestreo[indices_opcionales]
                z_pool = z_muestreo[indices_opcionales]
                
                num_fijos = len(indices_fijos)
                # Varianza de puntos fijos siempre con kriging ordinario
                varianzas.append(self.calcular_varianza_kriging(
                    x_fijos, y_fijos, z_fijos, x_malla, y_malla, pesos))
                num_puntos.append(num_fijos)
            else:
                indices_fijos = []
                indices_opcionales = list(range(len(x_muestreo)))
                x_fijos, y_fijos, z_fijos = np.array([]), np.array([]), np.array([])
                x_pool = x_muestreo
                y_pool = y_muestreo
                z_pool = z_muestreo
                num_fijos = 0
            
            indices_seleccionados_por_iteracion.append(indices_fijos.copy() if num_fijos > 0 else [])
            
            max_adicionales = min(max_pts - num_fijos, len(indices_opcionales)) if num_fijos > 0 else min(max_pts, len(indices_opcionales))
            
            if self.resultados_secuencial:
                print("✓ Usando resultado del método secuencial como punto de partida para el algoritmo genético")
            
            for n_pts_adicionales in range(1, max_adicionales + 1):
                
                # ===========================================================
                # Definir función de fitness según método seleccionado
                # ===========================================================
                if usar_kalman:
                    # Fitness con kriging simple (matrices precomputadas)
                    def fitness_func(ga_instance, solution, solution_idx):
                        indices_pool_sel = np.argsort(solution)[::-1][:n_pts_adicionales]
                        
                        if len(indices_pool_sel) == 0:
                            return -1e10
                        
                        indices_originales = [indices_opcionales[i] for i in indices_pool_sel]
                        
                        if num_fijos > 0:
                            todos_indices = list(indices_fijos) + indices_originales
                        else:
                            todos_indices = indices_originales
                        
                        varianza = self.calcular_varianza_kriging_simple(
                            todos_indices, C_gc_full, C_cc_full,
                            var_value, R, n_grid, pesos)
                        
                        # Ponderar por peso promedio de los pozos opcionales seleccionados
                        if pp is not None:
                            mean_w = np.mean([pp[i] for i in indices_originales])
                            return -varianza / mean_w
                        return -varianza
                else:
                    # Fitness con kriging ordinario
                    def fitness_func(ga_instance, solution, solution_idx):
                        indices_seleccionados = np.argsort(solution)[::-1][:n_pts_adicionales]
                        
                        if len(indices_seleccionados) == 0:
                            return -1e10
                        
                        if num_fijos > 0:
                            x_sel = np.concatenate([x_fijos, x_pool[indices_seleccionados]])
                            y_sel = np.concatenate([y_fijos, y_pool[indices_seleccionados]])
                            z_sel = np.concatenate([z_fijos, z_pool[indices_seleccionados]])
                        else:
                            x_sel = x_pool[indices_seleccionados]
                            y_sel = y_pool[indices_seleccionados]
                            z_sel = z_pool[indices_seleccionados]
                        
                        varianza = self.calcular_varianza_kriging(
                            x_sel, y_sel, z_sel, x_malla, y_malla, pesos)
                        
                        # Ponderar por peso promedio de los pozos opcionales seleccionados
                        if pp is not None:
                            idx_orig = [indices_opcionales[i] for i in indices_seleccionados]
                            mean_w = np.mean([pp[i] for i in idx_orig])
                            return -varianza / mean_w
                        return -varianza
                
                num_genes = len(x_pool)
                
                initial_population = None
                sol_per_pop = self.ga_sol_per_pop.get()
                
                if self.resultados_secuencial:
                    initial_population = np.random.uniform(0.0, 1.0, (sol_per_pop, num_genes))
                    
                    solucion_secuencial = self.crear_solucion_inicial_desde_secuencial(
                        num_genes, n_pts_adicionales, indices_opcionales, num_fijos
                    )
                    initial_population[0] = solucion_secuencial
                    
                    num_individuos_basados = min(3, sol_per_pop // 4)
                    for i in range(1, num_individuos_basados):
                        variacion = solucion_secuencial + np.random.normal(0, 0.1, num_genes)
                        variacion = np.clip(variacion, 0.0, 1.0)
                        initial_population[i] = variacion
                
                ga_instance = pygad.GA(
                    num_generations=self.ga_num_generations.get(),
                    num_parents_mating=self.ga_num_parents_mating.get(),
                    fitness_func=fitness_func,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    gene_type=float,
                    initial_population=initial_population,
                    parent_selection_type=self.ga_parent_selection.get(),
                    keep_parents=2,
                    crossover_type=self.ga_crossover_type.get(),
                    mutation_type=self.ga_mutation_type.get(),
                    mutation_percent_genes=self.ga_mutation_percent.get(),
                    random_seed=42,
                    suppress_warnings=True
                )
                
                ga_instance.run()
                
                # Obtener mejor solución
                solution, solution_fitness, solution_idx = ga_instance.best_solution()
                
                indices_seleccionados_pool = np.argsort(solution)[::-1][:n_pts_adicionales]
                
                if num_fijos > 0:
                    indices_reales = indices_fijos + [indices_opcionales[i] for i in indices_seleccionados_pool]
                else:
                    indices_reales = [indices_opcionales[i] for i in indices_seleccionados_pool]
                
                # ===========================================================
                # Varianza reportada: SIEMPRE recalculada con kriging ordinario
                # (El fitness puede estar ponderado por pesos de pozos)
                # ===========================================================
                x_sel_ok = x_muestreo[indices_reales]
                y_sel_ok = y_muestreo[indices_reales]
                z_sel_ok = z_muestreo[indices_reales]
                mejor_varianza = self.calcular_varianza_kriging(
                    x_sel_ok, y_sel_ok, z_sel_ok, x_malla, y_malla, pesos)
                
                varianzas.append(mejor_varianza)
                num_puntos.append(num_fijos + n_pts_adicionales)
                indices_seleccionados_por_iteracion.append(indices_reales)
                
                self.root.update()
            
            tiempo_fin = time.time()
            self.tiempo_genetico = tiempo_fin - tiempo_inicio
            
            # Registrar qué método se usó para el fitness
            metodo_fitness = "Filtro de Kalman (Kriging Simple)" if usar_kalman else "Kriging Ordinario"
            
            self.resultados_genetico = {
                'num_puntos': num_puntos,
                'varianzas': varianzas,
                'indices_por_iteracion': indices_seleccionados_por_iteracion,
                'metodo_fitness': metodo_fitness
            }
            
            msg = (f"Algoritmo genético completado\n"
                   f"Tiempo: {self.tiempo_genetico:.2f} segundos\n"
                   f"Fitness evaluado con: {metodo_fitness}\n"
                   f"Varianzas reportadas con: Kriging Ordinario")
            if self.malla_ponderada:
                msg += "\n\n(Usando malla ponderada)"
            
            messagebox.showinfo("Éxito", msg)
            self.graficar_resultados()
        except Exception as e:
            messagebox.showerror("Error", f"Error en algoritmo genético: {str(e)}")
    
    # ================================================================
    # FILTRO DE KALMAN - Priorización secuencial
    # ================================================================
    def ejecutar_kalman(self):
        """
        Priorización de pozos mediante filtro de Kalman.
        
        Fundamento teórico:
        -------------------
        En geoestadística, el campo espacial Z(x) se descompone como:
            Z(x) = S(x) + ε(x)
        donde S(x) es el proceso estructurado (con covarianza dada por el
        variograma) y ε(x) es ruido de medición (nugget).
        
        El filtro de Kalman trata esto como un problema de estimación de estado:
        - Estado: valores de S en los nodos de la malla (a priori, covarianza = C_gg)
        - Observaciones: mediciones en los pozos candidatos, con ruido R = nugget
        - Modelo de observación: z_k = S(x_k) + ε_k
        
        Al asimilar una observación en el candidato k, se actualiza la covarianza
        mediante la fórmula de Kalman (actualización de rango 1):
            C_posterior = C_prior - (C_prior * h_k * h_k^T * C_prior) / (h_k^T * C_prior * h_k + R)
        
        Implementación en dos fases:
        ----------------------------
        FASE 1 (rápida): El filtro de Kalman determina el ORDEN óptimo de
                         selección de pozos mediante actualizaciones rank-1
                         de la covarianza (equivalente a kriging simple).
        
        FASE 2 (evaluación): Una vez determinado el orden, se recalcula la
                             varianza con KRIGING ORDINARIO para que la métrica
                             sea directamente comparable con los otros métodos.
        
        Esto combina la eficiencia computacional del filtro de Kalman para
        la optimización con la consistencia del kriging ordinario para la
        evaluación de resultados.
        """
        if not self.validar_datos():
            return
        
        try:
            tiempo_inicio = time.time()
            
            x_muestreo = self.df_muestreo['x'].values
            y_muestreo = self.df_muestreo['y'].values
            z_muestreo = self.obtener_datos_transformados()
            
            x_malla = self.df_malla['x'].values
            y_malla = self.df_malla['y'].values
            n_grid = len(x_malla)
            n_candidates = len(x_muestreo)
            
            pesos = self.df_malla['peso'].values if 'peso' in self.df_malla.columns else None
            
            # Parámetros del variograma
            var_value = self.var_sill.get()
            nugget_value = self.var_nugget.get()
            range_value = self.var_range.get()
            
            model = self.crear_modelo()
            R = nugget_value if nugget_value > 0 else 1e-6
            
            # ===========================================================
            # FASE 1: Determinar orden óptimo con filtro de Kalman
            # ===========================================================
            # Construir matrices de covarianza (kriging simple / Kalman)
            # model.covariance(h) retorna la covarianza estructurada
            # (partial sill), con C(0) = var. El nugget se maneja como R.
            # ===========================================================
            
            coords_grid = np.column_stack([x_malla, y_malla])
            coords_cand = np.column_stack([x_muestreo, y_muestreo])
            
            # Varianza a priori en cada nodo = partial sill
            var_grid = np.full(n_grid, var_value, dtype=np.float64)
            
            # Covarianza cruzada malla-candidatos
            dist_gc = cdist(coords_grid, coords_cand)
            C_gc = model.covariance(dist_gc).astype(np.float64)
            
            # Covarianza candidatos-candidatos
            dist_cc = cdist(coords_cand, coords_cand)
            C_cc = model.covariance(dist_cc).astype(np.float64)
            
            # Pesos normalizados para criterio de selección de Kalman
            if pesos is not None and self.malla_ponderada:
                pesos_norm = (pesos * n_grid / np.sum(pesos)).astype(np.float64)
            else:
                pesos_norm = None
            
            sill_total = var_value + nugget_value
            norm_factor_kalman = sill_total if sill_total > 0 else 1.0
            
            # Actualización rank-1 de Kalman
            def kalman_update(k):
                """
                Asimila la observación en el candidato k.
                Actualiza in-place: var_grid, C_gc, C_cc
                """
                nonlocal var_grid, C_gc, C_cc
                
                c_gk = C_gc[:, k].copy()
                c_ck = C_cc[:, k].copy()
                sigma_kk = C_cc[k, k]
                
                S = sigma_kk + R
                if S <= 1e-12:
                    return
                
                inv_S = 1.0 / S
                
                var_grid -= (c_gk ** 2) * inv_S
                np.maximum(var_grid, 0.0, out=var_grid)
                
                C_gc -= np.outer(c_gk, c_ck) * inv_S
                C_cc -= np.outer(c_ck, c_ck) * inv_S
            
            # Manejo de puntos fijos/opcionales
            if self.usar_opcionales.get() and len(self.puntos_fijos) > 0:
                indices_fijos = list(self.puntos_fijos)
                indices_opcionales = list(self.puntos_opcionales)
                num_fijos = len(indices_fijos)
            else:
                indices_fijos = []
                indices_opcionales = list(range(n_candidates))
                num_fijos = 0
            
            selected = []
            
            # Asimilar puntos fijos
            if num_fijos > 0:
                for idx in indices_fijos:
                    kalman_update(idx)
                    selected.append(idx)
            
            # Selección greedy: en cada paso elegir el candidato
            # que maximiza la reducción de traza ponderada × peso del pozo
            disponibles = [i for i in range(n_candidates) if i not in selected]
            
            # Obtener pesos de pozos
            pp = self._get_pesos_pozos()
            
            for step in range(len(disponibles)):
                mejor_idx = None
                mejor_score = -np.inf
                
                for k in disponibles:
                    c_gk = C_gc[:, k]
                    sigma_kk = C_cc[k, k]
                    S = sigma_kk + R
                    
                    if S <= 1e-12:
                        continue
                    
                    reduccion_por_nodo = (c_gk ** 2) / S
                    
                    if pesos_norm is not None:
                        reduccion_total = np.dot(reduccion_por_nodo, pesos_norm)
                    else:
                        reduccion_total = np.sum(reduccion_por_nodo)
                    
                    # Ponderar por peso del pozo candidato
                    peso_k = pp[k] if pp is not None else 1.0
                    score = peso_k * reduccion_total
                    
                    if score > mejor_score:
                        mejor_score = score
                        mejor_idx = k
                
                if mejor_idx is not None and mejor_score > 1e-15:
                    kalman_update(mejor_idx)
                    selected.append(mejor_idx)
                    disponibles.remove(mejor_idx)
                else:
                    # Ya no hay reducción significativa; agregar restantes
                    # en cualquier orden (su aporte es despreciable)
                    for k in disponibles:
                        selected.append(k)
                    break
                
                self.root.update()
            
            tiempo_kalman_orden = time.time() - tiempo_inicio
            
            # ===========================================================
            # FASE 2: Recalcular varianzas con KRIGING ORDINARIO
            # Usando el orden determinado por Kalman, se evalúa la varianza
            # con la misma función que usa el método de inclusiones
            # sucesivas, garantizando comparabilidad directa.
            # ===========================================================
            
            varianzas = []
            num_puntos = []
            
            # Varianza inicial (sin puntos) - misma convención que secuencial
            varianza_inicial = n_grid
            varianzas.append(varianza_inicial)
            num_puntos.append(0)
            
            # Reconstruir secuencia acumulativa y evaluar con kriging ordinario
            if num_fijos > 0:
                # Evaluar con los puntos fijos
                x_sel = x_muestreo[indices_fijos]
                y_sel = y_muestreo[indices_fijos]
                z_sel = z_muestreo[indices_fijos]
                
                var_ok = self.calcular_varianza_kriging(
                    x_sel, y_sel, z_sel, x_malla, y_malla, pesos)
                varianzas.append(var_ok)
                num_puntos.append(num_fijos)
            
            # Evaluar cada punto adicional en el orden de Kalman
            indices_acumulados = list(indices_fijos) if num_fijos > 0 else []
            
            # Los puntos opcionales en el orden determinado por Kalman
            puntos_opcionales_ordenados = [idx for idx in selected if idx not in indices_fijos]
            
            for i, idx in enumerate(puntos_opcionales_ordenados):
                indices_acumulados.append(idx)
                
                x_sel = x_muestreo[indices_acumulados]
                y_sel = y_muestreo[indices_acumulados]
                z_sel = z_muestreo[indices_acumulados]
                
                var_ok = self.calcular_varianza_kriging(
                    x_sel, y_sel, z_sel, x_malla, y_malla, pesos)
                varianzas.append(var_ok)
                num_puntos.append(len(indices_acumulados))
                
                self.root.update()
            
            tiempo_fin = time.time()
            self.tiempo_kalman = tiempo_fin - tiempo_inicio
            
            self.resultados_kalman = {
                'num_puntos': num_puntos,
                'varianzas': varianzas,
                'indices': selected
            }
            
            msg = (f"Filtro de Kalman completado\n"
                   f"Tiempo total: {self.tiempo_kalman:.2f} segundos\n"
                   f"  · Fase 1 (orden con Kalman): {tiempo_kalman_orden:.2f}s\n"
                   f"  · Fase 2 (evaluación con Kriging Ord.): {self.tiempo_kalman - tiempo_kalman_orden:.2f}s\n"
                   f"Puntos priorizados: {len(selected)}")
            if self.malla_ponderada:
                msg += "\n\n(Usando malla ponderada)"
            
            messagebox.showinfo("Éxito", msg)
            self.graficar_resultados()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en filtro de Kalman: {str(e)}")
    
    def ejecutar_ambos(self):
        """Ejecuta secuencial + genético (compatibilidad)"""
        self.ejecutar_secuencial()
        if self.resultados_secuencial:
            self.ejecutar_genetico()
    
    def ejecutar_todos(self):
        """Ejecuta los tres métodos: secuencial, filtro de Kalman y genético"""
        self.ejecutar_secuencial()
        self.ejecutar_kalman()
        if self.resultados_secuencial:
            self.ejecutar_genetico()
    
    def validar_datos(self):
        if self.df_muestreo is None:
            messagebox.showwarning("Advertencia", "Cargue los datos de muestreo")
            return False
        if self.df_malla is None:
            messagebox.showwarning("Advertencia", "Genere o cargue la malla de estimación")
            return False
        
        if len(self.puntos_fijos) + len(self.puntos_opcionales) == 0:
            messagebox.showerror("Error", 
                "Debe clasificar los puntos de muestreo.\n\n"
                "Use el botón 'Seleccionar Puntos Fijos/Opcionales' en la pestaña Datos\n"
                "para marcar cuáles puntos son fijos (obligatorios) y cuáles opcionales.\n\n"
                "Nota: Puede dejar todos como opcionales (sin puntos fijos) si lo desea.")
            return False
        
        return True
    
    def graficar_resultados(self):
        self.fig_resultados.clear()
        ax = self.fig_resultados.add_subplot(111)
        
        titulo_extra = " (Malla ponderada)" if self.malla_ponderada else ""
        
        if self.resultados_secuencial:
            ax.plot(self.resultados_secuencial['num_puntos'], 
                   self.resultados_secuencial['varianzas'],
                   'b-o', label=f'Inclusiones sucesivas - Kriging Ord. ({self.tiempo_secuencial:.2f}s)', 
                   linewidth=2, markersize=5)
            self.lbl_tiempo_secuencial.config(
                text=f"Tiempo inc. sucesivas: {self.tiempo_secuencial:.2f}s"
            )
        
        if self.resultados_kalman:
            ax.plot(self.resultados_kalman['num_puntos'], 
                   self.resultados_kalman['varianzas'],
                   'g-^', label=f'Filtro de Kalman ({self.tiempo_kalman:.2f}s)', 
                   linewidth=2, markersize=5)
            self.lbl_tiempo_kalman.config(
                text=f"Tiempo Kalman: {self.tiempo_kalman:.2f}s"
            )
        
        if self.resultados_genetico:
            # Mostrar método de fitness en la leyenda
            metodo_str = ""
            if 'metodo_fitness' in self.resultados_genetico:
                if "Kalman" in self.resultados_genetico['metodo_fitness']:
                    metodo_str = " [fitness: Kalman]"
                else:
                    metodo_str = " [fitness: K.O.]"
            
            ax.plot(self.resultados_genetico['num_puntos'], 
                   self.resultados_genetico['varianzas'],
                   'r-s', label=f'Algoritmo genético{metodo_str} ({self.tiempo_genetico:.2f}s)', 
                   linewidth=2, markersize=5)
            self.lbl_tiempo_genetico.config(
                text=f"Tiempo genético: {self.tiempo_genetico:.2f}s"
            )
        
        ax.set_xlabel('Número de puntos de muestreo', fontsize=12)
        ax.set_ylabel('Varianza normalizada' + (' ponderada' if self.malla_ponderada else ''), fontsize=12)
        ax.set_title(f'Comparación de metodologías de selección de puntos{titulo_extra}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        self.canvas_resultados.draw()
    
    def exportar_resultados(self):
        if self.resultados_secuencial is None and self.resultados_genetico is None and self.resultados_kalman is None:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Guardar resultados",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if filename:
            try:
                # Calcular varianza máxima
                varianza_max = None
                if self.resultados_secuencial and len(self.resultados_secuencial['varianzas']) > 0:
                    varianza_max = self.resultados_secuencial['varianzas'][0]
                elif self.resultados_kalman and len(self.resultados_kalman['varianzas']) > 0:
                    varianza_max = self.resultados_kalman['varianzas'][0]
                elif self.resultados_genetico and len(self.resultados_genetico['varianzas']) > 0:
                    varianza_max = self.resultados_genetico['varianzas'][0]
                
                if varianza_max is None or varianza_max == 0:
                    varianza_max = 1.0
                
                num_fijos = len(self.puntos_fijos) if len(self.puntos_fijos) > 0 else 0
                
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Hoja de resumen
                    resumen_data = {
                        'Metodología': [],
                        'Tiempo (segundos)': [],
                        'Puntos evaluados': [],
                        'Malla ponderada': []
                    }
                    
                    if self.resultados_secuencial:
                        resumen_data['Metodología'].append('Priorización inclusiones sucesivas (Kriging Ord.)')
                        resumen_data['Tiempo (segundos)'].append(self.tiempo_secuencial)
                        resumen_data['Puntos evaluados'].append(len(self.resultados_secuencial['num_puntos'])-1)
                        resumen_data['Malla ponderada'].append('Sí' if self.malla_ponderada else 'No')
                    
                    if self.resultados_kalman:
                        resumen_data['Metodología'].append('Filtro de Kalman')
                        resumen_data['Tiempo (segundos)'].append(self.tiempo_kalman)
                        resumen_data['Puntos evaluados'].append(len(self.resultados_kalman['num_puntos'])-1)
                        resumen_data['Malla ponderada'].append('Sí' if self.malla_ponderada else 'No')
                    
                    if self.resultados_genetico:
                        metodo_ga = self.resultados_genetico.get('metodo_fitness', 'Kriging Ordinario')
                        resumen_data['Metodología'].append(f'Algoritmo genético (fitness: {metodo_ga})')
                        resumen_data['Tiempo (segundos)'].append(self.tiempo_genetico)
                        resumen_data['Puntos evaluados'].append(len(self.resultados_genetico['num_puntos'])-1)
                        resumen_data['Malla ponderada'].append('Sí' if self.malla_ponderada else 'No')
                    
                    df_resumen = pd.DataFrame(resumen_data)
                    df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
                    
                    # Hoja de secuencial
                    if self.resultados_secuencial:
                        varianzas = np.array(self.resultados_secuencial['varianzas'])
                        reduccion = (1 - varianzas / varianza_max) * 100
                        
                        claves_acumuladas = []
                        claves_acumuladas.append('')
                        
                        num_puntos_list = self.resultados_secuencial['num_puntos']
                        indices_list = self.resultados_secuencial['indices']
                        
                        if num_fijos > 0 and len(num_puntos_list) > 1:
                            claves_fijos = []
                            for i in range(num_fijos):
                                if i < len(indices_list):
                                    idx = indices_list[i]
                                    clave = self.df_muestreo.iloc[idx]['clave']
                                    claves_fijos.append(str(clave))
                            claves_acumuladas.append(', '.join(claves_fijos))
                            
                            for i in range(num_fijos, len(indices_list)):
                                idx = indices_list[i]
                                clave = self.df_muestreo.iloc[idx]['clave']
                                claves_previas = claves_acumuladas[-1]
                                claves_acumuladas.append(claves_previas + ', ' + str(clave))
                        else:
                            for i, idx in enumerate(indices_list):
                                if i == 0:
                                    clave = self.df_muestreo.iloc[idx]['clave']
                                    claves_acumuladas.append(str(clave))
                                else:
                                    clave = self.df_muestreo.iloc[idx]['clave']
                                    claves_previas = claves_acumuladas[-1]
                                    claves_acumuladas.append(claves_previas + ', ' + str(clave))
                        
                        n_datos = len(self.resultados_secuencial['num_puntos'])
                        if len(claves_acumuladas) != n_datos:
                            while len(claves_acumuladas) < n_datos:
                                claves_acumuladas.append('')
                            claves_acumuladas = claves_acumuladas[:n_datos]
                        
                        df_sec = pd.DataFrame({
                            'Num_Puntos': self.resultados_secuencial['num_puntos'],
                            'Claves_Seleccionadas': claves_acumuladas,
                            'Varianza_Normalizada': varianzas,
                            'Reduccion_Varianza_%': reduccion
                        })
                        df_sec.to_excel(writer, sheet_name='Inclusiones sucesivas', index=False)
                    
                    # ====================================================
                    # Hoja de filtro de Kalman
                    # ====================================================
                    if self.resultados_kalman:
                        varianzas_k = np.array(self.resultados_kalman['varianzas'])
                        reduccion_k = (1 - varianzas_k / varianza_max) * 100
                        
                        claves_acumuladas_k = []
                        claves_acumuladas_k.append('')  # Para 0 puntos
                        
                        indices_list_k = self.resultados_kalman['indices']
                        
                        if num_fijos > 0 and len(indices_list_k) >= num_fijos:
                            # Claves de puntos fijos
                            claves_fijos_k = []
                            for i in range(num_fijos):
                                idx = indices_list_k[i]
                                clave = self.df_muestreo.iloc[idx]['clave']
                                claves_fijos_k.append(str(clave))
                            claves_acumuladas_k.append(', '.join(claves_fijos_k))
                            
                            # Puntos opcionales uno a uno
                            for i in range(num_fijos, len(indices_list_k)):
                                idx = indices_list_k[i]
                                clave = self.df_muestreo.iloc[idx]['clave']
                                claves_previas = claves_acumuladas_k[-1]
                                claves_acumuladas_k.append(claves_previas + ', ' + str(clave))
                        else:
                            for i, idx in enumerate(indices_list_k):
                                clave = self.df_muestreo.iloc[idx]['clave']
                                if i == 0:
                                    claves_acumuladas_k.append(str(clave))
                                else:
                                    claves_previas = claves_acumuladas_k[-1]
                                    claves_acumuladas_k.append(claves_previas + ', ' + str(clave))
                        
                        n_datos_k = len(self.resultados_kalman['num_puntos'])
                        if len(claves_acumuladas_k) != n_datos_k:
                            while len(claves_acumuladas_k) < n_datos_k:
                                claves_acumuladas_k.append('')
                            claves_acumuladas_k = claves_acumuladas_k[:n_datos_k]
                        
                        df_kal = pd.DataFrame({
                            'Num_Puntos': self.resultados_kalman['num_puntos'],
                            'Claves_Seleccionadas': claves_acumuladas_k,
                            'Varianza_Normalizada': varianzas_k,
                            'Reduccion_Varianza_%': reduccion_k
                        })
                        df_kal.to_excel(writer, sheet_name='Filtro de Kalman', index=False)
                    
                    # Hoja de genético
                    if self.resultados_genetico:
                        varianzas = np.array(self.resultados_genetico['varianzas'])
                        reduccion = (1 - varianzas / varianza_max) * 100
                        
                        claves_por_iteracion = []
                        if num_fijos > 0:
                            claves_por_iteracion.append('')
                        
                        if 'indices_por_iteracion' in self.resultados_genetico:
                            for indices_lista in self.resultados_genetico['indices_por_iteracion']:
                                if len(indices_lista) == 0:
                                    claves_por_iteracion.append('')
                                else:
                                    claves = []
                                    for idx in indices_lista:
                                        if idx < len(self.df_muestreo):
                                            clave = self.df_muestreo.iloc[idx]['clave']
                                            claves.append(str(clave))
                                        else:
                                            idx_malla = idx - len(self.df_muestreo)
                                            claves.append(f'Malla-{idx_malla + 1}')
                                    claves_por_iteracion.append(', '.join(claves))
                        
                        n_datos = len(self.resultados_genetico['num_puntos'])
                        if len(claves_por_iteracion) != n_datos:
                            while len(claves_por_iteracion) < n_datos:
                                claves_por_iteracion.append('')
                            claves_por_iteracion = claves_por_iteracion[:n_datos]
                        
                        df_gen = pd.DataFrame({
                            'Num_Puntos': self.resultados_genetico['num_puntos'],
                            'Claves_Seleccionadas': claves_por_iteracion,
                            'Varianza_Normalizada': varianzas,
                            'Reduccion_Varianza_%': reduccion
                        })
                        df_gen.to_excel(writer, sheet_name='Genetico', index=False)
                    
                    # Hoja de parámetros del variograma
                    params_data = {
                        'Parámetro': ['Modelo', 'Transformación de datos', 
                                      'Range', 'Partial sill', 'Nugget', 'Sill', 'Varianza máxima'],
                        'Valor': [
                            self.info_modelo(),
                            self.info_transformacion(),
                            self.var_range.get(),
                            self.var_sill.get(),
                            self.var_nugget.get(),
                            self.var_sill.get() + self.var_nugget.get(),
                            varianza_max
                        ]
                    }
                    df_params = pd.DataFrame(params_data)
                    df_params.to_excel(writer, sheet_name='Parametros_Variograma', index=False)
                    
                    # Hoja de parámetros de PyGAD
                    if self.resultados_genetico:
                        metodo_fitness = self.resultados_genetico.get('metodo_fitness', 'Kriging Ordinario')
                        pygad_params = {
                            'Parámetro': [
                                'Número de generaciones',
                                'Tamaño de población',
                                'Padres para apareamiento',
                                '% Mutación',
                                'Tipo de selección',
                                'Tipo de cruce',
                                'Tipo de mutación',
                                'Método de evaluación del fitness',
                                'Varianzas reportadas con',
                                'Usa solución secuencial como semilla'
                            ],
                            'Valor': [
                                self.ga_num_generations.get(),
                                self.ga_sol_per_pop.get(),
                                self.ga_num_parents_mating.get(),
                                self.ga_mutation_percent.get(),
                                self.ga_parent_selection.get(),
                                self.ga_crossover_type.get(),
                                self.ga_mutation_type.get(),
                                metodo_fitness,
                                'Kriging Ordinario',
                                'Sí' if self.resultados_secuencial else 'No'
                            ]
                        }
                        df_pygad = pd.DataFrame(pygad_params)
                        df_pygad.to_excel(writer, sheet_name='Parametros_PyGAD', index=False)
                    
                    # Hoja de información de la malla
                    if self.df_malla is not None:
                        malla_info = {
                            'Parámetro': [
                                'Número de nodos',
                                'Malla ponderada',
                                'Peso mínimo',
                                'Peso máximo',
                                'Peso promedio',
                                'Desviación estándar pesos'
                            ],
                            'Valor': [
                                len(self.df_malla),
                                'Sí' if self.malla_ponderada else 'No',
                                self.df_malla['peso'].min() if 'peso' in self.df_malla.columns else 1.0,
                                self.df_malla['peso'].max() if 'peso' in self.df_malla.columns else 1.0,
                                self.df_malla['peso'].mean() if 'peso' in self.df_malla.columns else 1.0,
                                self.df_malla['peso'].std() if 'peso' in self.df_malla.columns else 0.0
                            ]
                        }
                        df_malla_info = pd.DataFrame(malla_info)
                        df_malla_info.to_excel(writer, sheet_name='Info_Malla', index=False)
                        
                        self.df_malla.to_excel(writer, sheet_name='Malla_Completa', index=False)
                    
                    # Hoja de pesos de pozos
                    pp = self._get_pesos_pozos()
                    pesos_info = {
                        'Parámetro': [
                            'Pesos de pozos habilitados',
                            'Columna de pesos',
                            'Peso mínimo',
                            'Peso máximo',
                            'Peso promedio'
                        ],
                        'Valor': [
                            'Sí' if pp is not None else 'No',
                            self.col_peso_pozo if self.col_peso_pozo else 'N/A',
                            f"{self.pesos_pozos.min():.4f}" if self.pesos_pozos is not None else 'N/A',
                            f"{self.pesos_pozos.max():.4f}" if self.pesos_pozos is not None else 'N/A',
                            f"{self.pesos_pozos.mean():.4f}" if self.pesos_pozos is not None else 'N/A'
                        ]
                    }
                    df_pesos_info = pd.DataFrame(pesos_info)
                    df_pesos_info.to_excel(writer, sheet_name='Pesos_Pozos', index=False)
                    
                    # Si hay pesos, exportar tabla de pozos con pesos
                    if self.pesos_pozos is not None and self.df_muestreo is not None:
                        df_pozos_pesos = pd.DataFrame({
                            'clave': self.df_muestreo['clave'].values if 'clave' in self.df_muestreo.columns else range(len(self.pesos_pozos)),
                            'x': self.df_muestreo['x'].values,
                            'y': self.df_muestreo['y'].values,
                            'peso_pozo': self.pesos_pozos
                        })
                        df_pozos_pesos.to_excel(writer, sheet_name='Pesos_Pozos_Detalle', index=False)
                
                messagebox.showinfo("Éxito", f"Resultados exportados exitosamente a:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al exportar resultados: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GeostatApp(root)
    root.mainloop()
