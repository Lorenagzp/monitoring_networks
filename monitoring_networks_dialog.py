# -*- coding: utf-8 -*-
"""
/***************************************************************************
 MonitoringNetworks
 A QGIS plugin
 Optimize the selection of monitoring wells to maximize information and mininize costs.

 # This script implements the main dialog interface for the Monitoring Networks QGIS plugin.
 # The window consists of tabbed dialogs guiding users through:
 # 1 - input data selection, 
 # 2 - geostatistical analysis,
 # 3 - estimation grid creation,
 # 4 - well prioritization via Kalman-based optimization,
 # 5 - visualization of results in a map.

# There can be indicators of prior importance of certain:
# * wells or
# * parameters or
# * areas of interest.
# These should be indicated as weights for the optimization.

# This script was written almost entirely using AI in Cursor software.

Plugin Builder used: http://g-sherman.github.io/Qgis-Plugin-Builder/ 

 Thanks to James in Norway, for theaching me how to translate QGIS plugins <3
 https://jamesinnorway.wordpress.com/2014/04/08/translation-of-qgis-plugins/
                              -------------------
        begin                : 2026-03-14
        By                   : Lorena Gonzalez

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/



"""

# Functions for the user interface of the plugin.

from qgis.PyQt import uic
from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer, QObject, QEvent, QVariant, QTranslator, QCoreApplication
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                                QTabWidget, QWidget, QComboBox, QListWidget,
                                QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox,
                                QLabel, QGroupBox, QCheckBox,
                                QProgressBar, QTextEdit, QMessageBox, QFileDialog,
                                QAbstractItemView, QHeaderView, QGridLayout, QScrollArea,
                                QApplication,
                                QSizePolicy)
from qgis.core import (QgsProject, QgsVectorLayer, QgsMapLayerProxyModel,
                      QgsFieldProxyModel, QgsFeature, QgsGeometry, QgsPointXY,
                      QgsField, QgsFields, QgsWkbTypes, QgsVectorDataProvider)
from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox
from PyQt5.QtGui import QColor
import numpy as np
import gstools as gs
from scipy import stats
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .monitoring_networks_analysis import (
    extract_layer_coordinates,
    extract_point_records_from_layer,
    align_coordinates_with_transform,
    align_point_ids_with_transform,
    align_well_weights_with_transform,
    resolve_well_weight_field,
    compute_descriptive_stats,
    run_ordinary_kriging_cross_validation,
    compute_variance_reduction_curve,
    compute_total_variance_percent,
    OptimizationInput,
    ParameterInput,
    ordinary_kriging_interpolation,
    write_excel_sheets,
    read_excel_table_rows,
    STAT_KEYS,
)

STATS_TRANSFORM_COL = 0
STATS_VALUE_COL_OFFSET = 1

STATS_VALUE_HEADERS = (
    'Count', 'Min', 'Max', 'Mean', 'Median',
    'Std Dev', 'Variance', 'Asymmetry', 'Kurtosis',
)

VAR_COL_PARAM = 0
VAR_COL_WEIGHT = 1
VAR_COL_TRANSFORM = 2
VAR_COL_MODEL = 3
VAR_COL_NUGGET = 4
VAR_COL_SILL = 5
VAR_COL_RANGE = 6
VAR_COL_R2 = 7

VARIOGRAM_MODEL_TYPES = [
    'spherical', 'exponential', 'gaussian', 'stable', 'matern',
]

CV_SUMMARY_KEYS = ('min', 'max', 'mean', 'mae', 'mse', 'rmse')

CV_SUMMARY_FORMATS = {
    'min': '{:.3f}',
    'max': '{:.3f}',
    'mean': '{:.3f}',
    'mae': '{:.3f}',
    'mse': '{:.3f}',
    'rmse': '{:.3f}',
}

CV_COL_ID = 0
CV_COL_INCLUDED = 1
CV_COL_MEASURED = 2
CV_COL_PREDICTED = 3
CV_COL_ERROR = 4
CV_COL_SE = 5
CV_COL_STD_ERROR = 6

# Internal combo item data for tab 4 multi-parameter optimization (not translated).
MN_COMBINED_PARAMETERS_KEY = '__mn_combined_parameters__'

def stats_value_header_labels():
    """Column headers for the tab 2 basic statistics table."""
    return [
        QCoreApplication.translate("Tab 2", header)
        for header in STATS_VALUE_HEADERS
    ]

def cv_summary_header_labels(context_name):
    """Column headers for CV summary tables (tabs 2 and 5)."""
    return [
        QCoreApplication.translate(context_name, "Min error"),
        QCoreApplication.translate(context_name, "Max error"),
        QCoreApplication.translate(context_name, "Mean error"),
        QCoreApplication.translate(context_name, "MAE"),
        QCoreApplication.translate(context_name, "MSE"),
        QCoreApplication.translate(context_name, "RMSE"),
    ]


def cv_results_header_labels(context_name):
    """Column headers for per-point CV tables (tabs 2 and 5)."""
    return [
        QCoreApplication.translate(context_name, "ID"),
        QCoreApplication.translate(context_name, "Included?"),
        QCoreApplication.translate(context_name, "Measured"),
        QCoreApplication.translate(context_name, "Predicted"),
        QCoreApplication.translate(context_name, "Error"),
        QCoreApplication.translate(context_name, "SE"),
        QCoreApplication.translate(context_name, "Standardized Error"),
    ]


def resolve_total_variance_series(results):
    """
    Total variance (%) at each optimization step from stored optimize results.

    Uses ``total_variance_percent`` when present (new runs); otherwise derives
    it from ``normalized_variances`` for backward compatibility.
    """
    if not results:
        return np.asarray([], dtype=float)
    cached = results.get('total_variance_percent')
    if cached is not None:
        return np.asarray(cached, dtype=float)
    return compute_total_variance_percent(
        results.get('normalized_variances', [])
    )


def format_total_variance_pct(value, decimals=2, missing='—'):
    """Format total variance (%) for UI tables and attribute display."""
    if value is None:
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not np.isfinite(number):
        return missing
    return f"{number:.{decimals}f}"


class MonitoringNetworksDialog(QDialog):
    """Main dialog of the plugin"""
    
    
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle(QCoreApplication.translate("Main Window", "Monitoring Networks Priorization"))
        self.setMinimumSize(750, 750)
        self._cached_layer = None
        self._cached_attribute = None
        self._cached_raw_values = None
        self._cached_coordinates = None
        self._cached_point_ids = None
        self._cached_null_count = 0
        self.stats_map_colorbar = None
        self.results_all_interp_colorbar = None
        self.results_sel_interp_colorbar = None
        self.results_interp_color_limits = None
        self._syncing_variogram = False
        self.init_ui()
        self.current_grid_points = None  # Store grid generated in tab 3 (array of points)
        self.variogram_models_by_attribute = {}  # Dictionary to store models by attribute
        self.variance_results = {}  # Variance reduction curves keyed by parameter

        initial_layer = self.input_data_layer.currentLayer()
        if initial_layer is not None:
            self.on_layer_changed(initial_layer)
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        
        # Tab 1: Data Selection
        self.tab_data = QTabWidget()
        self.setup_data_tab()
        self.tabs.addTab(
            self.tab_data, 
            "1. " + QCoreApplication.translate("Main Window", "Input Data")
        )
        
        # Tab 2: Geostatistical Analysis
        self.tab_variogram = QTabWidget()
        self.setup_variogram_tab()
        self.tabs.addTab(
            self.tab_variogram, 
            "2. " + QCoreApplication.translate("Main Window", "Geostatistics")
        )

        # Tab 3: Estimation Grid
        self.tab_grid = QTabWidget()
        self.setup_grid_tab()
        self.tabs.addTab(
            self.tab_grid, 
            "3. " + QCoreApplication.translate("Main Window", "Estimation Grid")
        )
        
        # Tab 4: Prioritization
        self.tab_prioritization = QTabWidget()
        self.setup_prioritization_tab()
        self.tabs.addTab(
            self.tab_prioritization, 
            "4. " + QCoreApplication.translate("Main Window", "Priorization")
        )
        
        # Tab 5: Results
        self.tab_results = QTabWidget()
        self.setup_results_tab()
        self.tabs.addTab(
            self.tab_results, 
            "5. " + QCoreApplication.translate("Main Window", "Map")
        )
        
        layout.addWidget(self.tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton(QCoreApplication.translate("Main Window", "← Previous"))
        self.prev_btn.clicked.connect(self.previous_tab)
        button_layout.addWidget(self.prev_btn)
        
        button_layout.addStretch()
        
        self.next_btn = QPushButton(QCoreApplication.translate("Main Window", "Next →"))
        self.next_btn.clicked.connect(self.next_tab)
        button_layout.addWidget(self.next_btn)
        
        
        self.close_btn = QPushButton(QCoreApplication.translate("Main Window", "Close"))
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def setup_data_tab(self):
        """Configures the data selection tab"""
        
        # Layout for the selection of data source
        data_tab_layout = QVBoxLayout()

        # Section to select the point layer to use
        layer_group = QGroupBox()
        layer_layout = QVBoxLayout()
        layer_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 1", "Point layer:")
        ))
   
        self.input_data_layer = QgsMapLayerComboBox() # Combo box to select layer of points from the TOC
        self.input_data_layer.setFilters(QgsMapLayerProxyModel.PointLayer) # Filter to show only point layers
        self.input_data_layer.layerChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.input_data_layer)


        # Layer info text
        self.layer_info = QTextEdit()
        self.layer_info.setMaximumHeight(100)
        self.layer_info.setReadOnly(True)
        layer_layout.addWidget(self.layer_info)
        
        layer_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 1", "Layer attribute values:")
        ))
        self.layer_fields_table = QTableWidget()
        self.layer_fields_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layer_fields_table.setAlternatingRowColors(True)
        self.layer_fields_table.horizontalHeader().setStretchLastSection(True)
        self.layer_fields_table.setMinimumHeight(140)
        self.layer_fields_table.setMaximumHeight(280)
        layer_layout.addWidget(self.layer_fields_table)

        layer_group.setLayout(layer_layout)
        data_tab_layout.addWidget(layer_group)
        
        # Attribute selector section
        attr_group = QGroupBox(QCoreApplication.translate("Tab 1", "Selection of Attributes for Analysis"))
        attr_layout = QVBoxLayout()
        attr_layout.addWidget(QLabel(QCoreApplication.translate("Tab 1", "Attributes available to use as parameters of the monitoring network:")))
        
        # List of selectable attributes
        self.selected_data_parameters = QListWidget()
        self.selected_data_parameters.setSelectionMode(QAbstractItemView.MultiSelection) #Allow multiple attribute selection
        self.selected_data_parameters.itemSelectionChanged.connect(self.on_attribute_changed)
        attr_layout.addWidget(self.selected_data_parameters)
        
        attr_group.setLayout(attr_layout)
        data_tab_layout.addWidget(attr_group)

        # Assign the layout to the data tab
        self.tab_data.setLayout(data_tab_layout)
        
    
    def setup_variogram_tab(self):
        """Setup the variogram tab"""

        # Layout for the variogram tab
        variogram_tab_layout = QVBoxLayout()

        # Attribute selector for tab 2 (options from tab 1 multi-selection)
        attr_select_layout = QHBoxLayout()
        attr_select_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 2", "Parameter:")
        ))
        self.selected_attr_combo = QComboBox()
        self.selected_attr_combo.addItem(
            QCoreApplication.translate("Tab 2", "No parameter selected")
        )
        self.selected_attr_combo.setEnabled(False)
        self.selected_attr_combo.currentIndexChanged.connect(
            self._on_tab2_attribute_changed
        )
        attr_select_layout.addWidget(self.selected_attr_combo)
        variogram_tab_layout.addLayout(attr_select_layout)

        # Calculate geostatistics button
        self.calc_geostats_btn = QPushButton(
            QCoreApplication.translate("Tab 2", "Calculate geostatistics")
        )
        self.calc_geostats_btn.clicked.connect(self.calculate_geostatistics)
        variogram_tab_layout.addWidget(self.calc_geostats_btn) #, 1, 1


        # Create QScrollArea to allow scrolling
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        #scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container widget for scrollable content
        scroll_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Basic statistics of the selected attribute
        stats_group = QGroupBox(QCoreApplication.translate("Tab 2", "Basic statistics"))
        stats_layout = QVBoxLayout()

        self.stats_table = QTableWidget()
        table_headers = [QCoreApplication.translate(
            "Tab 2", 'Transformation'
        )] + stats_value_header_labels()
        self.stats_table.setColumnCount(len(table_headers))
        self.stats_table.setRowCount(1)
        self.stats_table.setHorizontalHeaderLabels(table_headers)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.resizeColumnsToContents()
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setFixedHeight(80)

        self.stats_transform_combo = QComboBox()
        self.stats_transform_combo.addItems([
            QCoreApplication.translate("Tab 2","None"), 
            QCoreApplication.translate("Tab 2","Logarithmic")
        ])
        self.stats_transform_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_attribute_statistics(force_reextract=False)
        )
        self.stats_table.setCellWidget(
            0, STATS_TRANSFORM_COL, self.stats_transform_combo
        )

        for col in range(STATS_VALUE_COL_OFFSET, self.stats_table.columnCount()):
            self.stats_table.setItem(0, col, QTableWidgetItem(""))
        stats_layout.addWidget(self.stats_table)

        # Informative label to select attributes
        self.stats_info_label = QLabel(QCoreApplication.translate(
            "Tab 2", "Select an attribute to view its statistics.")
        )
        self.stats_info_label.setStyleSheet("color: gray; font-style: italic;")
        stats_layout.addWidget(self.stats_info_label) 

        # Histogram plot
        PLOT_HEIGHT = 250
        plots_layout = QHBoxLayout()
        self.stats_hist_figure = Figure(figsize=(4, 3))
        self.stats_hist_canvas = FigureCanvas(self.stats_hist_figure)
        self.stats_hist_canvas.setMinimumHeight(PLOT_HEIGHT)
        self.stats_hist_ax = self.stats_hist_figure.add_subplot(111)
        plots_layout.addWidget(self.stats_hist_canvas) 

        # Spatial distribution plot
        self.stats_map_figure = Figure(figsize=(4, 3))
        self.stats_map_canvas = FigureCanvas(self.stats_map_figure)
        self.stats_map_ax = self.stats_map_figure.add_subplot(111)
        plots_layout.addWidget(self.stats_map_canvas)
        stats_layout.addLayout(plots_layout)
        self._clear_stats_plots()

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # 2. Mathematical models
        params_group = QGroupBox(QCoreApplication.translate("Tab 2", "Variogram Parameters"))
        params_layout = QVBoxLayout()
        
        self.var_params_table = QTableWidget()
        self.var_params_table.setColumnCount(8)
        self.var_params_table.setHorizontalHeaderLabels([
            QCoreApplication.translate("Tab 2", "Parameter"), 
            QCoreApplication.translate("Tab 2", "Weight"),  
            QCoreApplication.translate("Tab 2", "Transform"), 
            QCoreApplication.translate("Tab 2", "Model"),
            QCoreApplication.translate("Tab 2", "Nugget"),
            QCoreApplication.translate("Tab 2", "Sill"), 
            QCoreApplication.translate("Tab 2", "Range"),
            QCoreApplication.translate("Tab 2", "R²")
        ])
        #self.var_params_table.setFixedHeight(126)
        # Connect table changes to update the variogram
        self.var_params_table.itemChanged.connect(self.on_params_table_changed)
        self.var_params_table.resizeColumnsToContents()
        self.var_params_table.horizontalHeader().setStretchLastSection(True)
        params_layout.addWidget(self.var_params_table)

        # Progress bar
        progress_layout = QVBoxLayout()
        self.var_progress_status_label = QLabel(QCoreApplication.translate("Tab 2", "State: Ready"))
        self.var_progress_status_label.setStyleSheet("color: gray; font-style: italic;")
        self.var_progress_status_label.setVisible(True)
        progress_layout.addWidget(self.var_progress_status_label)

        self.var_progress_bar = QProgressBar()
        self.var_progress_bar.setVisible(True)
        self.var_progress_bar.setValue(0)
        progress_layout.addWidget(self.var_progress_bar)

        params_layout.addLayout(progress_layout)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # 3: Variogram
        variogram_widget_group = QGroupBox()
        variogram_widget_layout = QVBoxLayout()
        self.variogram_widget = VariogramWidget(self, self)
        self.variogram_widget.params_changed.connect(
            self._on_variogram_widget_params_changed
        )
        variogram_widget_layout.addWidget(self.variogram_widget)
        variogram_widget_group.setLayout(variogram_widget_layout)
        layout.addWidget(variogram_widget_group)

        # Leave-one-out cross-validation (ordinary kriging with fitted variogram).
        # Refreshes when geostatistics runs or when variogram parameters change.
        cv_group = QGroupBox(
            QCoreApplication.translate("Tab 2", "Cross-Validation")
        )
        cv_layout = QVBoxLayout()

        cv_summary_label = QLabel(
            QCoreApplication.translate("Tab 2", "Cross-Validation Summary")
        )
        cv_layout.addWidget(cv_summary_label)

        self.cv_summary_table = QTableWidget()
        self.cv_summary_table.setColumnCount(len(CV_SUMMARY_KEYS))
        self.cv_summary_table.setRowCount(1)
        self.cv_summary_table.setHorizontalHeaderLabels(
            cv_summary_header_labels("Tab 2")
        )
        self.cv_summary_table.verticalHeader().setVisible(False)
        self.cv_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cv_summary_table.setFixedHeight(70)
        self.cv_summary_table.horizontalHeader().setStretchLastSection(True)
        cv_layout.addWidget(self.cv_summary_table)

        cv_results_label = QLabel(
            QCoreApplication.translate("Tab 2", "Cross-Validation Results")
        )
        cv_layout.addWidget(cv_results_label)

        self.cv_results_table = QTableWidget()
        self.cv_results_table.setColumnCount(7)
        self.cv_results_table.setHorizontalHeaderLabels(
            cv_results_header_labels("Tab 2")
        )
        self.cv_results_table.verticalHeader().setVisible(True)
        self.cv_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cv_results_table.setMinimumHeight(160)
        self.cv_results_table.setMaximumHeight(220)
        self.cv_results_table.horizontalHeader().setStretchLastSection(True)
        cv_layout.addWidget(self.cv_results_table)

        self.cv_info_label = QLabel(
            QCoreApplication.translate(
                "Tab 2",
                "Run geostatistics to compute leave-one-out cross-validation.",
            )
        )
        self.cv_info_label.setStyleSheet("color: gray; font-style: italic;")
        cv_layout.addWidget(self.cv_info_label)

        self.cv_figure = Figure(figsize=(6, 4))
        self.cv_canvas = FigureCanvas(self.cv_figure)
        self.cv_canvas.setMinimumHeight(PLOT_HEIGHT)
        self.cv_ax = self.cv_figure.add_subplot(111)
        cv_layout.addWidget(self.cv_canvas)

        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group)
        self._clear_cross_validation()

        # Add stretch at the end so the content doesn't expand more than necessary
        #layout.addStretch()
        
        # Configure the container widget with the layout
        scroll_widget.setLayout(layout)
        
        # Add the container widget to the scroll area
        scroll_area.setWidget(scroll_widget)
        
        # Add the scroll area to the main layout
        variogram_tab_layout.addWidget(scroll_area)

        # Assign the layout to the variogram tab
        self.tab_variogram.setLayout(variogram_tab_layout)
            
    def setup_grid_tab(self):
        """Configures the estimation grid tab"""

        layout = QVBoxLayout()
        
        # Generate grid
        gen_grid_group = QGroupBox(
            QCoreApplication.translate("Tab 3", "Calculate the estimation grid")
        )
        gen_grid_layout = QGridLayout()

        # Node spacing
        gen_grid_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 3", "Spacing between nodes (map units):")
        ), 0, 0)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(10, 10000)
        
        # Default spacing: (max_dist / 2) / 10
        layer = getattr(self, 'input_data_layer', None)
        max_dist = None
        if layer and hasattr(layer, 'currentLayer') and layer.currentLayer() is not None:
            current_layer = layer.currentLayer()
            try:
                if current_layer is not None:
                    features = list(current_layer.getFeatures())
                    coords = np.array([[f.geometry().asPoint().x(), f.geometry().asPoint().y()] for f in features if f.geometry().isMultipart() == False and f.geometry().isEmpty() == False])
                    if len(coords) >= 2:
                        from scipy.spatial.distance import pdist
                        max_dist = np.max(pdist(coords))
            except Exception:
                pass
        
        if max_dist is not None and max_dist > 0:
            spacing_default = (max_dist / 2) / 10
        else:
            spacing_default = 100.0
        
        self.spacing_spin.setValue(spacing_default)
        self.spacing_spin.setDecimals(2)
        self.spacing_spin.setSingleStep(1.0)
        self.spacing_spin.valueChanged.connect(self.on_spacing_changed)
        gen_grid_layout.addWidget(self.spacing_spin, 0, 1)
        
        # Label to display estimated total points
        gen_grid_layout.addWidget(QLabel(QCoreApplication.translate("Tab 3", "Estimated total points:")), 1, 0)
        self.estimated_points_label = QLabel(QCoreApplication.translate("Tab 3", "Calculating..."))
        self.estimated_points_label.setStyleSheet("color: gray; font-style: italic;")
        gen_grid_layout.addWidget(self.estimated_points_label, 1, 1)
        
        # Buffer
        gen_grid_layout.addWidget(QLabel(QCoreApplication.translate("Tab 3", "Buffer (map units):")), 2, 0)
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0, 10000)
        self.buffer_spin.setValue(100)
        self.buffer_spin.setDecimals(0)
        self.buffer_spin.valueChanged.connect(self.update_estimated_points)
        gen_grid_layout.addWidget(self.buffer_spin, 2, 1)
        
        # Calculate grid button
        self.preview_grid_btn = QPushButton(QCoreApplication.translate("Tab 3", "Calculate grid"))
        self.preview_grid_btn.clicked.connect(self.preview_grid)
        gen_grid_layout.addWidget(self.preview_grid_btn, 3, 1) #, 1, 1

        gen_grid_group.setLayout(gen_grid_layout)
        layout.addWidget(gen_grid_group)

        # Option to upload the grid from a file
        load_grid_group = QGroupBox(QCoreApplication.translate("Tab 3", "Or Upload the estimation grid from *.XLSX file [Optional]"))
        load_grid_layout = QGridLayout()

        # Button to upload grid from file
        load_grid_layout.addWidget(QLabel(QCoreApplication.translate("Tab 3", "The file needs to contain the next data columns: ID, X, Y, weight")))
        self.load_grid_btn = QPushButton(QCoreApplication.translate("Tab 3", "Select *.XLSX file"))
        self.load_grid_btn.clicked.connect(self.load_grid_as_layer)
        load_grid_layout.addWidget(self.load_grid_btn, 2, 1)

        load_grid_group.setLayout(load_grid_layout)
        layout.addWidget(load_grid_group)
        
        # Canvas for visualization
        self.grid_figure = Figure(figsize=(8, 6))
        self.grid_canvas = FigureCanvas(self.grid_figure)
        layout.addWidget(self.grid_canvas)
        
        # Button to save grid as temporary layer
        #btn_grid_layout = QHBoxLayout()
        self.save_grid_btn = QPushButton(QCoreApplication.translate("Tab 3", "Save Grid as Temporary Layer"))
        self.save_grid_btn.clicked.connect(self.save_grid_as_layer,1,0)
        #btn_grid_layout.addWidget(self.save_grid_btn)
        #btn_grid_layout.addStretch()
        #layout.addLayout(btn_grid_layout) #needed?
        layout.addWidget(self.save_grid_btn) #,3, 0, 1, 2
        
        # Load the layout elements to the Grid Tab
        self.tab_grid.setLayout(layout)
        
        # Calculate initial point estimate (after all widgets are created)
        # Use QTimer to ensure it runs after the interface is ready
        from qgis.PyQt.QtCore import QTimer
        QTimer.singleShot(100, self.update_estimated_points)

    def setup_prioritization_tab(self):
        """Configures the prioritization tab"""

        priorization_tab_layout = QVBoxLayout()

        # Select parameter to configure monitoring network
        mn_param_select_group = QGroupBox(
            QCoreApplication.translate("Tab 4", "Configure monitoring network")
        )
        mn_param_select_layout = QVBoxLayout()
        mn_param_select_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 4", "Select parameter to optimize:")
        ))
        self.mn_param_select_combo = QComboBox()
        self.mn_param_select_combo.addItem(
            QCoreApplication.translate("Tab 4", "No parameter selected")
        )
        self.mn_param_select_combo.setEnabled(False)
        self.mn_param_select_combo.currentIndexChanged.connect(
            self._on_mn_parameter_changed
        )
        mn_param_select_layout.addWidget(self.mn_param_select_combo)

        self.mn_use_well_weights = QCheckBox(
            QCoreApplication.translate(
                "Tab 4", "Use personalized well weight in the optimization"
            )
        )
        self.mn_use_well_weights.setEnabled(False)
        self.mn_use_well_weights.stateChanged.connect(
            self._update_mn_well_weight_info
        )
        mn_param_select_layout.addWidget(self.mn_use_well_weights)

        self.mn_well_weight_info_label = QLabel()
        self.mn_well_weight_info_label.setStyleSheet(
            "color: gray; font-style: italic;"
        )
        self.mn_well_weight_info_label.setWordWrap(True)
        mn_param_select_layout.addWidget(self.mn_well_weight_info_label)

        self.mn_use_grid_weights = QCheckBox(
            QCoreApplication.translate(
                "Tab 4", "Use estimation grid node weights in the optimization"
            )
        )
        self.mn_use_grid_weights.setEnabled(False)
        self.mn_use_grid_weights.stateChanged.connect(
            self._update_mn_grid_weight_info
        )
        mn_param_select_layout.addWidget(self.mn_use_grid_weights)

        self.mn_grid_weight_info_label = QLabel()
        self.mn_grid_weight_info_label.setStyleSheet(
            "color: gray; font-style: italic;"
        )
        self.mn_grid_weight_info_label.setWordWrap(True)
        mn_param_select_layout.addWidget(self.mn_grid_weight_info_label)

        mn_param_select_group.setLayout(mn_param_select_layout)
        priorization_tab_layout.addWidget(mn_param_select_group)

        # Create QScrollArea to allow scrolling
        mn_scroll_area = QScrollArea()
        mn_scroll_area.setWidgetResizable(True)
        mn_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container widget for scrollable content
        mn_scroll_widget = QWidget()
        mn_layout = QVBoxLayout()
        mn_layout.setContentsMargins(10, 10, 10, 10)

        # Simple-kriging variance reduction curve (greedy well ordering)
        variance_group = QGroupBox(
            QCoreApplication.translate("Tab 4", "Variance Reduction")
        )
        variance_layout = QVBoxLayout()

        optimize_btn_row = QHBoxLayout()
        self.mn_optimize_btn = QPushButton(
            QCoreApplication.translate("Tab 4", "Optimize")
        )
        self.mn_optimize_btn.clicked.connect(self.optimize_monitoring_network)
        optimize_btn_row.addWidget(self.mn_optimize_btn)
        self.mn_download_prioritization_btn = QPushButton(
            QCoreApplication.translate("Tab 4", "Download prioritization")
        )
        self.mn_download_prioritization_btn.clicked.connect(
            self.download_prioritization
        )
        optimize_btn_row.addWidget(self.mn_download_prioritization_btn)
        optimize_btn_row.addStretch()
        variance_layout.addLayout(optimize_btn_row)

        self.variance_info_label = QLabel(
            QCoreApplication.translate(
                "Tab 4",
                "Generate an estimation grid (tab 3) and fit a variogram "
                "(tab 2), then click Optimize.",
            )
        )
        self.variance_info_label.setStyleSheet("color: gray; font-style: italic;")
        self.variance_info_label.setWordWrap(True)
        variance_layout.addWidget(self.variance_info_label)

        MN_PLOT_HEIGHT = 280
        self.variance_figure = Figure(figsize=(6, 4))
        self.variance_canvas = FigureCanvas(self.variance_figure)
        self.variance_canvas.setMinimumHeight(MN_PLOT_HEIGHT)
        self.variance_ax = self.variance_figure.add_subplot(111)
        variance_layout.addWidget(self.variance_canvas)

        order_label = QLabel(
            QCoreApplication.translate("Tab 4", "Optimization order")
        )
        variance_layout.addWidget(order_label)

        self.mn_prioritization_order_table = QTableWidget()
        self.mn_prioritization_order_table.setColumnCount(4)
        self.mn_prioritization_order_table.setHorizontalHeaderLabels([
            QCoreApplication.translate("Tab 4", "Rank"),
            QCoreApplication.translate("Tab 4", "Well ID"),
            QCoreApplication.translate("Tab 4", "Well weight"),
            QCoreApplication.translate("Tab 4", "Total variance (%)"),
        ])
        self.mn_prioritization_order_table.verticalHeader().setVisible(True)
        self.mn_prioritization_order_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.mn_prioritization_order_table.setMinimumHeight(160)
        self.mn_prioritization_order_table.setMaximumHeight(280)
        self.mn_prioritization_order_table.horizontalHeader().setStretchLastSection(
            True
        )
        variance_layout.addWidget(self.mn_prioritization_order_table)

        variance_group.setLayout(variance_layout)
        mn_layout.addWidget(variance_group)
        self._clear_variance_reduction_plot()

        # Configure the container widget with the layout
        mn_scroll_widget.setLayout(mn_layout)
        
        # Add the container widget to the scroll area
        mn_scroll_area.setWidget(mn_scroll_widget)
        
        # Add the scroll area to the main layout
        priorization_tab_layout.addWidget(mn_scroll_area)

        # Assign the layout to the prioritization tab
        self.tab_prioritization.setLayout(priorization_tab_layout)
        self._update_mn_well_weight_info()
        self._update_mn_grid_weight_info()


    def setup_results_tab(self):
        """Configures the results tab (OK maps and CV for all vs selected wells)."""
        results_layout = QVBoxLayout()

        interp_group = QGroupBox(
            QCoreApplication.translate("Tab 5", "Ordinary Kriging Interpolation")
        )
        interp_layout = QVBoxLayout()

        wells_row = QHBoxLayout()
        wells_row.addWidget(QLabel(
            QCoreApplication.translate("Tab 5", "Number of monitoring wells:")
        ))
        self.results_well_spin = QSpinBox()
        self.results_well_spin.setMinimum(3)
        self.results_well_spin.setMaximum(3)
        self.results_well_spin.setValue(3)
        self.results_well_spin.setEnabled(False)
        self.results_well_spin.valueChanged.connect(
            self._refresh_ok_interpolation_selected_wells
        )
        wells_row.addWidget(self.results_well_spin)
        self.save_selected_wells_btn = QPushButton(
            QCoreApplication.translate("Tab 5", "Download selected wells as layer")
        )
        self.save_selected_wells_btn.clicked.connect(
            self.save_selected_wells_as_layer
        )
        wells_row.addWidget(self.save_selected_wells_btn)
        wells_row.addStretch()
        interp_layout.addLayout(wells_row)

        plots_row = QHBoxLayout()
        RESULTS_PLOT_HEIGHT = 280

        all_column = QVBoxLayout()
        all_column.addWidget(QLabel(
            QCoreApplication.translate("Tab 5", "All wells")
        ))
        self.results_all_interp_figure = Figure(figsize=(5, 4))
        self.results_all_interp_canvas = FigureCanvas(self.results_all_interp_figure)
        self.results_all_interp_canvas.setMinimumHeight(RESULTS_PLOT_HEIGHT)
        self.results_all_interp_ax = self.results_all_interp_figure.add_subplot(111)
        all_column.addWidget(self.results_all_interp_canvas)
        (
            all_cv_widget,
            self.results_all_cv_summary_table,
            self.results_all_cv_results_table,
            self.results_all_cv_info_label,
        ) = self._build_cv_tables_widget("Tab 5")
        all_column.addWidget(all_cv_widget)
        plots_row.addLayout(all_column, stretch=1)

        sel_column = QVBoxLayout()
        sel_column.addWidget(QLabel(
            QCoreApplication.translate("Tab 5", "Selected wells")
        ))
        self.results_sel_interp_figure = Figure(figsize=(5, 4))
        self.results_sel_interp_canvas = FigureCanvas(self.results_sel_interp_figure)
        self.results_sel_interp_canvas.setMinimumHeight(RESULTS_PLOT_HEIGHT)
        self.results_sel_interp_ax = self.results_sel_interp_figure.add_subplot(111)
        sel_column.addWidget(self.results_sel_interp_canvas)
        (
            sel_cv_widget,
            self.results_sel_cv_summary_table,
            self.results_sel_cv_results_table,
            self.results_sel_cv_info_label,
        ) = self._build_cv_tables_widget("Tab 5")
        sel_column.addWidget(sel_cv_widget)
        plots_row.addLayout(sel_column, stretch=1)

        interp_layout.addLayout(plots_row)
        interp_group.setLayout(interp_layout)
        results_layout.addWidget(interp_group)

        self.tab_results.setLayout(results_layout)
        self._clear_ok_interpolation_plots()


    def _build_cv_tables_widget(self, context_name):
        """Creates summary/results CV tables and info label for tab 5 columns."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        summary_label = QLabel(
            QCoreApplication.translate(context_name, "Cross-Validation Summary")
        )
        layout.addWidget(summary_label)

        summary_table = QTableWidget()
        summary_table.setColumnCount(len(CV_SUMMARY_KEYS))
        summary_table.setRowCount(1)
        summary_table.setHorizontalHeaderLabels(
            cv_summary_header_labels(context_name)
        )
        summary_table.verticalHeader().setVisible(False)
        summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        summary_table.setFixedHeight(70)
        summary_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(summary_table)

        results_label = QLabel(
            QCoreApplication.translate(context_name, "Cross-Validation Results")
        )
        layout.addWidget(results_label)

        results_table = QTableWidget()
        results_table.setColumnCount(7)
        results_table.setHorizontalHeaderLabels(
            cv_results_header_labels(context_name)
        )
        results_table.verticalHeader().setVisible(True)
        results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        results_table.setMinimumHeight(140)
        results_table.setMaximumHeight(200)
        results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(results_table)

        info_label = QLabel(
            QCoreApplication.translate(
                context_name,
                "Run Kalman optimization on tab 4 to compute cross-validation.",
            )
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        container.setLayout(layout)
        return container, summary_table, results_table, info_label


    def _count_layer_wells(self, attr_name):
        """Returns the number of valid measurement points in the input layer."""
        layer = self.input_data_layer.currentLayer()
        if not layer or not attr_name:
            return 0
        _, coordinates, values, _, _ = extract_point_records_from_layer(
            layer, attr_name
        )
        if coordinates is None or values is None:
            return 0
        return int(values.size)

    def _update_results_well_spinbox(self):
        """Sets the well-count spin box range from 3 to the layer point count."""
        if not hasattr(self, 'results_well_spin'):
            return

        attr_name = self._current_mn_parameter()
        count_attr = attr_name
        if self._is_combined_mn_parameter(attr_name):
            selected = self._selected_analysis_parameters()
            count_attr = selected[0] if selected else None
        n_wells = self._count_layer_wells(count_attr) if count_attr else 0

        self.results_well_spin.blockSignals(True)
        if n_wells < 3:
            self.results_well_spin.setEnabled(False)
            self.results_well_spin.setMinimum(3)
            self.results_well_spin.setMaximum(3)
            self.results_well_spin.setValue(3)
        else:
            self.results_well_spin.setEnabled(True)
            self.results_well_spin.setMinimum(3)
            self.results_well_spin.setMaximum(n_wells)
            current = self.results_well_spin.value()
            if current > n_wells:
                self.results_well_spin.setValue(n_wells)
            elif current < 3:
                self.results_well_spin.setValue(3)
        self.results_well_spin.blockSignals(False)

    def _clear_results_cv_tables(self, summary_table, results_table, info_label, message):
        """Resets tab 5 cross-validation tables and info label."""
        if summary_table is None or results_table is None:
            return

        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        for col in range(summary_table.columnCount()):
            item = summary_table.item(0, col)
            if item is None:
                item = QTableWidgetItem("")
                summary_table.setItem(0, col, item)
            else:
                item.setText("")
            item.setFlags(read_only)

        results_table.setRowCount(0)
        if info_label is not None:
            info_label.setText(message)
            info_label.setStyleSheet("color: gray; font-style: italic;")

    def _populate_cv_summary_table(self, summary_table, summary):
        """Fills a one-row CV summary table (tabs 2 and 5)."""
        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        for col, key in enumerate(CV_SUMMARY_KEYS):
            value = summary.get(key, np.nan)
            text = "—" if np.isnan(value) else CV_SUMMARY_FORMATS[key].format(value)
            item = summary_table.item(0, col)
            if item is None:
                item = QTableWidgetItem(text)
                summary_table.setItem(0, col, item)
            else:
                item.setText(text)
            item.setFlags(read_only)
        summary_table.resizeColumnsToContents()

    def _populate_cv_results_table(
        self,
        results_table,
        point_ids,
        cv_rows,
        context_name="Tab 2",
        highlight_selected=False,
    ):
        """Fills a per-point CV table (tabs 2 and 5)."""
        yes_text = QCoreApplication.translate(context_name, "Yes")
        no_text = QCoreApplication.translate(context_name, "No")
        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        selected_row_color = QColor('#d5f5e3')
        results_table.setRowCount(len(cv_rows))

        for row, (point_id, row_data) in enumerate(zip(point_ids, cv_rows)):
            included = row_data.get('included', True)
            cells = [
                str(point_id),
                yes_text if included else no_text,
                f"{row_data['measured']:.3f}",
                f"{row_data['predicted']:.3f}",
                f"{row_data['error']:.3f}",
                f"{row_data['se']:.3f}",
                f"{row_data['standardized_error']:.3f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(read_only)
                if highlight_selected and included:
                    item.setBackground(selected_row_color)
                results_table.setItem(row, col, item)

        results_table.resizeColumnsToContents()

    def _clear_ok_interpolation_plot_side(
        self, figure, canvas, colorbar_attr, message=...
    ):
        """Resets one tab 5 interpolation figure."""
        if figure is None or canvas is None:
            return

        figure.clear()
        ax = figure.add_subplot(111)
        colorbar = getattr(self, colorbar_attr, None)
        if colorbar is not None:
            setattr(self, colorbar_attr, None)

        if message is None:
            canvas.draw()
            return ax

        placeholder = (
            QCoreApplication.translate("Tab 5", "No interpolation map available")
            if message is ...
            else message
        )
        ax.text(
            0.5,
            0.5,
            placeholder,
            ha='center',
            va='center',
            transform=ax.transAxes,
            color='gray',
        )
        ax.set_xticks([])
        ax.set_yticks([])
        canvas.draw()
        return ax

    def _clear_ok_interpolation_plots(self, message=...):
        """Resets both tab 5 interpolation maps and CV tables."""
        self.results_interp_color_limits = None
        default_cv_message = (
            QCoreApplication.translate(
                "Tab 5",
                "Run Kalman optimization on tab 4 to compute cross-validation.",
            )
            if message is ...
            else message
        )
        self._clear_ok_interpolation_plot_side(
            getattr(self, 'results_all_interp_figure', None),
            getattr(self, 'results_all_interp_canvas', None),
            'results_all_interp_colorbar',
            message,
        )
        self._clear_ok_interpolation_plot_side(
            getattr(self, 'results_sel_interp_figure', None),
            getattr(self, 'results_sel_interp_canvas', None),
            'results_sel_interp_colorbar',
            message,
        )
        self.results_all_interp_ax = (
            self.results_all_interp_figure.axes[0]
            if hasattr(self, 'results_all_interp_figure')
            and self.results_all_interp_figure.axes
            else None
        )
        self.results_sel_interp_ax = (
            self.results_sel_interp_figure.axes[0]
            if hasattr(self, 'results_sel_interp_figure')
            and self.results_sel_interp_figure.axes
            else None
        )
        self._clear_results_cv_tables(
            getattr(self, 'results_all_cv_summary_table', None),
            getattr(self, 'results_all_cv_results_table', None),
            getattr(self, 'results_all_cv_info_label', None),
            default_cv_message,
        )
        self._clear_results_cv_tables(
            getattr(self, 'results_sel_cv_summary_table', None),
            getattr(self, 'results_sel_cv_results_table', None),
            getattr(self, 'results_sel_cv_info_label', None),
            default_cv_message,
        )

    def _resolve_cv_parameter_name(self, attr_name):
        """Map tab 4 combined mode key to the reference parameter used for CV."""
        if self._is_combined_mn_parameter(attr_name):
            selected = self._selected_analysis_parameters()
            return selected[0] if selected else None
        return attr_name

    def _tab5_reference_parameter(self):
        """Parameter whose variogram and values drive tab 5 maps and CV."""
        return self._resolve_cv_parameter_name(self._current_mn_parameter())

    def _compute_parameter_cross_validation(self, attr_name, network_indices=None):
        """
        Leave-one-out ordinary-kriging CV for one parameter.

        Loads wells from the input layer with the same transform and variogram
        state stored on tab 2, so tab 2 and tab 5 use identical inputs.
        """
        resolved_name = self._resolve_cv_parameter_name(attr_name)
        if not resolved_name:
            return None, None, QCoreApplication.translate(
                "Tab 2", "Select a parameter for cross-validation."
            )

        (
            coordinates,
            point_ids,
            _well_weights,
            parameters,
            error,
        ) = self._load_optimization_parameters([resolved_name])
        if error:
            return None, None, error
        if not parameters:
            return None, None, QCoreApplication.translate(
                "Tab 2", "No valid data for cross-validation."
            )

        state = self._get_variogram_state(resolved_name)
        if not state:
            return None, None, QCoreApplication.translate(
                "Tab 2",
                "Run geostatistics to fit a variogram for this parameter first.",
            )

        param = parameters[0]
        try:
            cv_result = run_ordinary_kriging_cross_validation(
                coordinates,
                param.values,
                state.get('model_type', 'spherical'),
                state.get('nugget', 0),
                state.get('sill', 0),
                state.get('range', 0),
                network_indices=network_indices,
            )
        except Exception:
            return None, None, QCoreApplication.translate(
                "Tab 2", "Cross-validation could not be completed."
            )

        if cv_result is None:
            return None, None, QCoreApplication.translate(
                "Tab 2",
                "At least three points are required for cross-validation.",
            )

        return cv_result, point_ids, None

    def _refresh_tab2_cross_validation(self, attr_name=None):
        """Recomputes and displays tab 2 CV for the given parameter."""
        if not hasattr(self, 'cv_summary_table'):
            return
        if attr_name is None:
            attr_name = self._current_analysis_attribute()
        if not attr_name:
            self._clear_cross_validation()
            return

        cv_result, point_ids, error = self._compute_parameter_cross_validation(
            attr_name
        )
        if error or cv_result is None:
            self._clear_cross_validation()
            self.cv_info_label.setText(error or QCoreApplication.translate(
                "Tab 2", "Cross-validation could not be completed."
            ))
            self.cv_info_label.setStyleSheet("color: red; font-style: italic;")
            return

        rows = cv_result['rows']
        self._populate_cv_summary_table(self.cv_summary_table, cv_result['summary'])
        self._populate_cv_results_table(
            self.cv_results_table,
            point_ids,
            rows,
            context_name="Tab 2",
        )

        measured = np.asarray([row['measured'] for row in rows], dtype=float)
        predicted = np.asarray([row['predicted'] for row in rows], dtype=float)
        valid = np.isfinite(predicted)
        self._update_cross_validation_plot(
            measured[valid], predicted[valid], attr_name
        )

        self.cv_info_label.setText(
            QCoreApplication.translate(
                "Tab 2",
                "Leave-one-out cross-validation for «{param}» ({n} points).",
            ).format(param=attr_name, n=int(np.sum(valid)))
        )
        self.cv_info_label.setStyleSheet("color: gray; font-style: italic;")

    def _refresh_tab5_views_after_variogram_change(self, attr_name):
        """Refresh tab 5 maps/CV when variogram inputs change for that parameter."""
        if attr_name not in getattr(self, 'variance_results', {}):
            return
        if attr_name != self._tab5_reference_parameter():
            return
        self._refresh_ok_interpolation_plot()

    def _get_ok_interpolation_context(self):
        """
        Shared validation for tab 5 interpolation/CV.

        Returns (context_dict, error_message). context_dict is None on error.
        """
        attr_name = self._current_mn_parameter()
        if not attr_name:
            return None, QCoreApplication.translate(
                "Tab 5", "Select a parameter on tab 4."
            )

        if attr_name not in self.variance_results:
            return None, QCoreApplication.translate(
                "Tab 5", "Run Kalman optimization on tab 4 first."
            )

        if (
            not hasattr(self, 'current_grid_points')
            or self.current_grid_points is None
            or len(self.current_grid_points) == 0
        ):
            return None, QCoreApplication.translate(
                "Tab 5", "Generate an estimation grid on tab 3 first."
            )

        param_data = self._get_mn_parameter_data(attr_name)
        if param_data is None:
            return None, QCoreApplication.translate(
                "Tab 5",
                "Run geostatistics on tab 2 for this parameter first.",
            )

        return {
            'attr_name': attr_name,
            'param_data': param_data,
            'state': param_data['state'],
            'coordinates': param_data['coordinates'],
            'values': param_data['values'],
            'point_ids': param_data.get('point_ids'),
            'grid': np.asarray(self.current_grid_points, dtype=float),
            'selection_order': self.variance_results[attr_name]['selection_order'],
        }, None

    def _compute_ok_estimates(self, context, well_indices):
        """Runs ordinary kriging on the grid for the given well indices."""
        coordinates = context['coordinates']
        values = context['values']
        state = context['state']
        grid = context['grid']
        indices = np.asarray(well_indices, dtype=int)

        return ordinary_kriging_interpolation(
            coordinates[indices],
            values[indices],
            grid,
            state.get('model_type', 'spherical'),
            state.get('nugget', 0),
            state.get('sill', 0),
            state.get('range', 0),
        )

    def _ok_interpolation_color_limits(self, estimates):
        """Returns (vmin, vmax) for tab 5 maps from grid estimates."""
        valid = np.asarray(estimates, dtype=float)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return None
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmin == vmax:
            pad = max(abs(vmin) * 0.05, 0.5)
            vmin -= pad
            vmax += pad
        return vmin, vmax

    def _ensure_ok_interpolation_color_limits(self, context):
        """
        Shared color ramp limits for tab 5 maps.

        Uses the all-wells interpolation range so the selected-wells map stays
        comparable to the left panel without redrawing it.
        """
        if self.results_interp_color_limits is not None:
            return self.results_interp_color_limits

        n_wells = context['coordinates'].shape[0]
        try:
            all_estimates = self._compute_ok_estimates(
                context, np.arange(n_wells)
            )
        except Exception:
            return None
        if all_estimates is None:
            return None

        self.results_interp_color_limits = self._ok_interpolation_color_limits(
            all_estimates
        )
        return self.results_interp_color_limits

    def _draw_ok_interpolation_map(
        self,
        figure,
        canvas,
        colorbar_attr,
        ax,
        context,
        estimates,
        selected_indices,
        title,
        color_limits=None,
    ):
        """Draws an OK interpolation map with selected/unselected well markers."""
        coordinates = context['coordinates']
        attr_name = context['attr_name']
        selected_indices = np.asarray(selected_indices, dtype=int)
        n_selected = selected_indices.size

        gx = context['grid'][:, 0]
        gy = context['grid'][:, 1]
        ax.clear()

        if color_limits is None:
            color_limits = self._ok_interpolation_color_limits(estimates)
        vmin = vmax = None
        if color_limits is not None:
            vmin, vmax = color_limits

        try:
            import matplotlib.tri as mtri
            triangulation = mtri.Triangulation(gx, gy)
            if vmin is not None and vmax is not None:
                levels = np.linspace(vmin, vmax, 21)
                contour = ax.tricontourf(
                    triangulation,
                    estimates,
                    levels=levels,
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                contour = ax.tricontourf(
                    triangulation,
                    estimates,
                    levels=20,
                    cmap='viridis',
                )
            colorbar = figure.colorbar(contour, ax=ax, label=attr_name)
        except Exception:
            scatter = ax.scatter(
                gx,
                gy,
                c=estimates,
                cmap='viridis',
                s=12,
                alpha=0.9,
                vmin=vmin,
                vmax=vmax,
            )
            colorbar = figure.colorbar(scatter, ax=ax, label=attr_name)
        setattr(self, colorbar_attr, colorbar)

        all_selected = n_selected == coordinates.shape[0]
        if not all_selected:
            unused_mask = np.ones(coordinates.shape[0], dtype=bool)
            unused_mask[selected_indices] = False
            if np.any(unused_mask):
                ax.scatter(
                    coordinates[unused_mask, 0],
                    coordinates[unused_mask, 1],
                    c='lightgray',
                    s=28,
                    edgecolors='k',
                    linewidths=0.3,
                    label=QCoreApplication.translate("Tab 5", "Not used"),
                    zorder=4,
                )

        sel_coords = coordinates[selected_indices]
        well_label = (
            QCoreApplication.translate("Tab 5", "All wells")
            if all_selected
            else QCoreApplication.translate("Tab 5", "Selected wells")
        )
        ax.scatter(
            sel_coords[:, 0],
            sel_coords[:, 1],
            c='red',
            s=55,
            edgecolors='k',
            linewidths=0.5,
            label=well_label,
            zorder=5,
        )

        point_ids = context.get('point_ids')
        if point_ids is not None:
            self._annotate_well_id_labels(ax, coordinates, point_ids)

        ax.set_xlabel(QCoreApplication.translate("Tab 5", "X"))
        ax.set_ylabel(QCoreApplication.translate("Tab 5", "Y"))
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        figure.subplots_adjust(right=0.88)
        canvas.draw()

    def _update_results_cross_validation(
        self,
        context,
        network_indices,
        summary_table,
        results_table,
        info_label,
    ):
        """Computes and displays CV tables for a well network."""
        cv_result, point_ids, error = self._compute_parameter_cross_validation(
            context['attr_name'],
            network_indices=network_indices,
        )

        default_message = error or QCoreApplication.translate(
            "Tab 5",
            "Cross-validation could not be computed.",
        )
        if cv_result is None:
            self._clear_results_cv_tables(
                summary_table, results_table, info_label, default_message
            )
            info_label.setStyleSheet("color: red; font-style: italic;")
            return

        if point_ids is None:
            point_ids = context.get('point_ids')
        if point_ids is None:
            point_ids = [str(i) for i in range(context['values'].size)]

        self._populate_cv_summary_table(summary_table, cv_result['summary'])
        self._populate_cv_results_table(
            results_table,
            point_ids,
            cv_result['rows'],
            context_name="Tab 5",
            highlight_selected=(network_indices is not None),
        )

        n_total = context['values'].size
        if network_indices is None:
            info_label.setText(
                QCoreApplication.translate(
                    "Tab 5",
                    "Leave-one-out cross-validation for «{param}» ({n} points).",
                ).format(param=context['attr_name'], n=n_total)
            )
        else:
            info_label.setText(
                QCoreApplication.translate(
                    "Tab 5",
                    "Cross-validation for «{param}» using {n_network} network "
                    "wells ({n_total} points).",
                ).format(
                    param=context['attr_name'],
                    n_network=len(network_indices),
                    n_total=n_total,
                )
            )
        info_label.setStyleSheet("color: gray; font-style: italic;")

    def _refresh_ok_interpolation_all_wells(self):
        """Builds the all-wells OK map and CV (not tied to the well-count spinbox)."""
        if not hasattr(self, 'results_all_interp_ax'):
            return

        context, error = self._get_ok_interpolation_context()
        if context is None:
            self._clear_ok_interpolation_plot_side(
                self.results_all_interp_figure,
                self.results_all_interp_canvas,
                'results_all_interp_colorbar',
                error,
            )
            self._clear_results_cv_tables(
                self.results_all_cv_summary_table,
                self.results_all_cv_results_table,
                self.results_all_cv_info_label,
                error,
            )
            self.results_all_cv_info_label.setStyleSheet(
                "color: gray; font-style: italic;"
            )
            return

        n_wells = context['coordinates'].shape[0]
        all_indices = np.arange(n_wells)

        try:
            estimates = self._compute_ok_estimates(context, all_indices)
        except Exception:
            estimates = None

        if estimates is None or estimates.size != context['grid'].shape[0]:
            message = QCoreApplication.translate(
                "Tab 5", "Interpolation could not be computed."
            )
            self._clear_ok_interpolation_plot_side(
                self.results_all_interp_figure,
                self.results_all_interp_canvas,
                'results_all_interp_colorbar',
                message,
            )
            self._clear_results_cv_tables(
                self.results_all_cv_summary_table,
                self.results_all_cv_results_table,
                self.results_all_cv_info_label,
                message,
            )
            return

        self._clear_ok_interpolation_plot_side(
            self.results_all_interp_figure,
            self.results_all_interp_canvas,
            'results_all_interp_colorbar',
            message=None,
        )
        self.results_all_interp_ax = self.results_all_interp_figure.axes[0]
        color_limits = self._ok_interpolation_color_limits(estimates)
        self.results_interp_color_limits = color_limits
        title = QCoreApplication.translate(
            "Tab 5",
            "Ordinary Kriging – {param} (all {n} wells)",
        ).format(param=context['attr_name'], n=n_wells)
        self._draw_ok_interpolation_map(
            self.results_all_interp_figure,
            self.results_all_interp_canvas,
            'results_all_interp_colorbar',
            self.results_all_interp_ax,
            context,
            estimates,
            all_indices,
            title,
            color_limits=color_limits,
        )
        self._update_results_cross_validation(
            context,
            network_indices=None,
            summary_table=self.results_all_cv_summary_table,
            results_table=self.results_all_cv_results_table,
            info_label=self.results_all_cv_info_label,
        )

    def _refresh_ok_interpolation_selected_wells(self):
        """Builds the selected-wells OK map and CV (responds to the spinbox)."""
        if not hasattr(self, 'results_sel_interp_ax'):
            return

        context, error = self._get_ok_interpolation_context()
        if context is None:
            self._clear_ok_interpolation_plot_side(
                self.results_sel_interp_figure,
                self.results_sel_interp_canvas,
                'results_sel_interp_colorbar',
                error,
            )
            self._clear_results_cv_tables(
                self.results_sel_cv_summary_table,
                self.results_sel_cv_results_table,
                self.results_sel_cv_info_label,
                error,
            )
            self.results_sel_cv_info_label.setStyleSheet(
                "color: gray; font-style: italic;"
            )
            return

        n_requested = self.results_well_spin.value()
        selection_order = context['selection_order']
        n_use = min(n_requested, len(selection_order))
        if n_use < 1:
            self._clear_ok_interpolation_plot_side(
                self.results_sel_interp_figure,
                self.results_sel_interp_canvas,
                'results_sel_interp_colorbar',
            )
            return

        indices = selection_order[:n_use]

        try:
            estimates = self._compute_ok_estimates(context, indices)
        except Exception:
            estimates = None

        if estimates is None or estimates.size != context['grid'].shape[0]:
            message = QCoreApplication.translate(
                "Tab 5", "Interpolation could not be computed."
            )
            self._clear_ok_interpolation_plot_side(
                self.results_sel_interp_figure,
                self.results_sel_interp_canvas,
                'results_sel_interp_colorbar',
                message,
            )
            self._clear_results_cv_tables(
                self.results_sel_cv_summary_table,
                self.results_sel_cv_results_table,
                self.results_sel_cv_info_label,
                message,
            )
            return

        self._clear_ok_interpolation_plot_side(
            self.results_sel_interp_figure,
            self.results_sel_interp_canvas,
            'results_sel_interp_colorbar',
            message=None,
        )
        self.results_sel_interp_ax = self.results_sel_interp_figure.axes[0]
        color_limits = self._ensure_ok_interpolation_color_limits(context)
        title = QCoreApplication.translate(
            "Tab 5",
            "Ordinary Kriging – {param} ({n} wells)",
        ).format(param=context['attr_name'], n=n_use)
        self._draw_ok_interpolation_map(
            self.results_sel_interp_figure,
            self.results_sel_interp_canvas,
            'results_sel_interp_colorbar',
            self.results_sel_interp_ax,
            context,
            estimates,
            indices,
            title,
            color_limits=color_limits,
        )
        self._update_results_cross_validation(
            context,
            network_indices=indices,
            summary_table=self.results_sel_cv_summary_table,
            results_table=self.results_sel_cv_results_table,
            info_label=self.results_sel_cv_info_label,
        )

    def _refresh_ok_interpolation_plots(
        self, refresh_all=True, refresh_selected=True
    ):
        """Refreshes tab 5 maps/CV; the all-wells side skips spinbox-only updates."""
        if refresh_all:
            self._refresh_ok_interpolation_all_wells()
        if refresh_selected:
            self._refresh_ok_interpolation_selected_wells()

    def _refresh_ok_interpolation_plot(self):
        """Refreshes both tab 5 interpolation panels."""
        self._refresh_ok_interpolation_plots(
            refresh_all=True, refresh_selected=True
        )

    def save_selected_wells_as_layer(self):
        """Adds the Kalman-selected wells as a temporary point layer in QGIS."""
        context, error = self._get_ok_interpolation_context()
        if context is None:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 5", "Download selected wells as layer"),
                error,
            )
            return

        layer = self.input_data_layer.currentLayer()
        if not layer:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 5", "Download selected wells as layer"),
                QCoreApplication.translate("Tab 5", "No input point layer is selected."),
            )
            return

        n_requested = self.results_well_spin.value()
        selection_order = np.asarray(context['selection_order'], dtype=int)
        n_use = min(n_requested, selection_order.size)
        if n_use < 1:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 5", "Download selected wells as layer"),
                QCoreApplication.translate(
                    "Tab 5", "No selected wells are available to export."
                ),
            )
            return

        attr_name = context['attr_name']
        display_name = self._mn_parameter_display_name(attr_name)
        coordinates = context['coordinates']
        values = context['values']
        point_ids = context.get('point_ids')
        selected_indices = selection_order[:n_use]

        results = self.variance_results[attr_name]
        total_variance = resolve_total_variance_series(results)

        try:
            crs = layer.crs()
            layer_name = QCoreApplication.translate(
                "Tab 5", "Selected Monitoring Wells - {param}"
            ).format(param=display_name)
            temp_layer = QgsVectorLayer(
                f"Point?crs={crs.authid()}",
                layer_name,
                "memory",
            )

            fields = QgsFields()
            fields.append(QgsField('Well_ID', QVariant.String))
            fields.append(QgsField('Prioritization_Rank', QVariant.Int))
            fields.append(QgsField('Parameter', QVariant.String))
            fields.append(QgsField('Measured', QVariant.Double))
            fields.append(QgsField('Total_Variance_pct', QVariant.Double))
            fields.append(QgsField('Coord_X', QVariant.Double))
            fields.append(QgsField('Coord_Y', QVariant.Double))

            temp_layer.dataProvider().addAttributes(fields)
            temp_layer.updateFields()

            features = []
            for rank, well_idx in enumerate(selected_indices, start=1):
                well_idx = int(well_idx)
                coord = coordinates[well_idx]
                well_id = (
                    str(point_ids[well_idx])
                    if point_ids is not None
                    else str(well_idx)
                )
                total_var = (
                    float(total_variance[rank])
                    if rank < total_variance.size
                    else None
                )

                feature = QgsFeature()
                feature.setGeometry(
                    QgsGeometry.fromPointXY(QgsPointXY(float(coord[0]), float(coord[1])))
                )
                feature.setAttributes([
                    well_id,
                    rank,
                    display_name,
                    float(values[well_idx]),
                    total_var,
                    float(coord[0]),
                    float(coord[1]),
                ])
                features.append(feature)

            temp_layer.dataProvider().addFeatures(features)
            temp_layer.updateExtents()
            QgsProject.instance().addMapLayer(temp_layer)

            QMessageBox.information(
                self,
                QCoreApplication.translate("Tab 5", "Download selected wells as layer"),
                QCoreApplication.translate(
                    "Tab 5",
                    "Saved {n} selected well(s) as a temporary layer for «{param}».",
                ).format(n=len(features), param=display_name),
            )
        except Exception as exc:
            import traceback
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 5", "Download selected wells as layer"),
                QCoreApplication.translate(
                    "Tab 5",
                    "Could not create the wells layer:\n{error}\n{details}",
                ).format(error=str(exc), details=traceback.format_exc()),
            )

    def previous_tab(self):
        """Go to previous tab"""
        current = self.tabs.currentIndex()
        if current > 0:
            self.tabs.setCurrentIndex(current - 1)
    
    def next_tab(self):
        """Go to the next tab."""
        current = self.tabs.currentIndex()
        if current < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(current + 1)

    @staticmethod
    def _format_layer_field_value(value):
        """Formats a QGIS attribute value for display in tab 1."""
        if value is None or (isinstance(value, QVariant) and value.isNull()):
            return ""
        if isinstance(value, QVariant):
            value = value.value()
        if value is None:
            return ""
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return ""
            text = f"{value:.6g}"
            return text
        return str(value)

    def _refresh_layer_fields_table(self, layer, max_rows=500):
        """Fills the tab 1 table with attribute values from the selected layer."""
        if not hasattr(self, 'layer_fields_table'):
            return

        self.layer_fields_table.clear()
        self.layer_fields_table.setRowCount(0)
        self.layer_fields_table.setColumnCount(0)

        if not layer:
            return

        fields = layer.fields()
        field_names = [field.name() for field in fields]
        if not field_names:
            return

        features = []
        for feature in layer.getFeatures():
            features.append(feature)
            if len(features) >= max_rows:
                break

        self.layer_fields_table.setColumnCount(len(field_names))
        self.layer_fields_table.setRowCount(len(features))
        self.layer_fields_table.setHorizontalHeaderLabels(field_names)

        for row, feature in enumerate(features):
            for col, field_name in enumerate(field_names):
                text = self._format_layer_field_value(feature[field_name])
                self.layer_fields_table.setItem(
                    row, col, QTableWidgetItem(text)
                )

        self.layer_fields_table.resizeColumnsToContents()

    def _clear_variogram_widget(self):
        """Resets the tab 2 variogram plot and parameters."""
        if not hasattr(self, 'variogram_widget'):
            return

        self._syncing_variogram = True
        try:
            self.variogram_widget.clear()
        finally:
            self._syncing_variogram = False

        if hasattr(self, 'var_progress_bar'):
            self.var_progress_bar.setValue(0)
        if hasattr(self, 'var_progress_status_label'):
            self.var_progress_status_label.setText(
                QCoreApplication.translate("Tab 2", "State: Ready")
            )

    def on_layer_changed(self, layer):
        """Update the layer information and the numeric attributes."""
        self._cached_layer = None
        self._cached_attribute = None
        self._cached_raw_values = None
        self._cached_coordinates = None
        self._cached_point_ids = None
        self._cached_null_count = 0
        self.variogram_models_by_attribute.clear()
        self.variance_results = {}
        self._sync_var_params_table_from_store()
        self.clear_stats_table()
        self._clear_variogram_widget()
        self._sync_attr_name_combo()
        self._clear_variance_reduction_plot()
        self._update_results_well_spinbox()
        self._clear_ok_interpolation_plots()
        self._update_mn_well_weight_info()
        self._update_mn_grid_weight_info()
        self.selected_data_parameters.clear()

        if not layer:
            self.layer_info.clear()
            self._refresh_layer_fields_table(None)
            return

        self._refresh_layer_fields_table(layer)
        info = QCoreApplication.translate(
            "Tab 1",
            "Name: {name}\n"
            "Number of features: {count}\n"
            "Coordinate system: {crs}",
        ).format(
            name=layer.name(),
            count=layer.featureCount(),
            crs=layer.crs().authid(),
        )
        feature_count = layer.featureCount()
        if feature_count > 500:
            info += "\n" + QCoreApplication.translate(
                "Tab 1", "Table shows the first 500 features."
            )
        self.layer_info.setText(info)
        for field in layer.fields():
            if field.type() in (QVariant.Int, QVariant.Double, QVariant.LongLong):
                self.selected_data_parameters.addItem(field.name())

    def _current_analysis_attribute(self):
        """Returns the attribute selected on tab 2, or None."""
        if not hasattr(self, 'selected_attr_combo') or not self.selected_attr_combo.isEnabled():
            return None
        return self.selected_attr_combo.currentText()

    def _sync_attr_name_combo(self):
        """Refreshes tab 2 combo from tab 1 selected attributes."""
        if not hasattr(self, 'selected_attr_combo'):
            return

        previous = self.selected_attr_combo.currentText()
        selected = [
            item.text()
            for item in self.selected_data_parameters.selectedItems()
        ]

        self.selected_attr_combo.blockSignals(True)
        self.selected_attr_combo.clear()
        if not selected:
            self.selected_attr_combo.addItem(
                QCoreApplication.translate("Tab 2", "No parameter selected")
            )
            self.selected_attr_combo.setEnabled(False)
        else:
            self.selected_attr_combo.addItems(selected)
            self.selected_attr_combo.setEnabled(True)
            restore_idx = self.selected_attr_combo.findText(previous)
            if restore_idx >= 0 and previous in selected:
                self.selected_attr_combo.setCurrentIndex(restore_idx)
            else:
                self.selected_attr_combo.setCurrentIndex(0)
        self.selected_attr_combo.blockSignals(False)
        self._sync_mn_param_select_combo()

    def _sync_mn_param_select_combo(self):
        """Refreshes tab 4 parameter combo from tab 1 selected attributes."""
        if not hasattr(self, 'mn_param_select_combo'):
            return

        previous = self.mn_param_select_combo.currentText()
        previous_data = self.mn_param_select_combo.currentData()
        selected = [
            item.text()
            for item in self.selected_data_parameters.selectedItems()
        ]

        self.mn_param_select_combo.blockSignals(True)
        self.mn_param_select_combo.clear()
        if not selected:
            self.mn_param_select_combo.addItem(
                QCoreApplication.translate("Tab 4", "No parameter selected")
            )
            self.mn_param_select_combo.setEnabled(False)
        else:
            for param in selected:
                self.mn_param_select_combo.addItem(param, param)
            if len(selected) >= 2:
                combined_label = QCoreApplication.translate(
                    "Tab 4", "Parameters combined"
                )
                self.mn_param_select_combo.addItem(
                    combined_label, MN_COMBINED_PARAMETERS_KEY
                )
            self.mn_param_select_combo.setEnabled(True)
            restore_idx = -1
            if previous_data is not None:
                restore_idx = self.mn_param_select_combo.findData(previous_data)
            if restore_idx < 0 and previous in selected:
                restore_idx = self.mn_param_select_combo.findText(previous)
            if restore_idx >= 0:
                self.mn_param_select_combo.setCurrentIndex(restore_idx)
            else:
                self.mn_param_select_combo.setCurrentIndex(0)
        self.mn_param_select_combo.blockSignals(False)
        self._update_results_well_spinbox()
        self._update_mn_well_weight_info()
        self._update_mn_grid_weight_info()
        self._refresh_ok_interpolation_plot()

    def _get_current_grid_weights(self):
        """Returns per-node grid weights aligned with current_grid_points, or None."""
        grid_info = getattr(self, 'current_grid_info', None)
        if not grid_info or 'weights' not in grid_info:
            return None
        weights = np.asarray(grid_info['weights'], dtype=float).ravel()
        points = getattr(self, 'current_grid_points', None)
        if points is None or weights.size != len(points):
            return None
        return weights

    def _grid_weights_available_for_optimization(self):
        """
        Grid weights can be used when the grid was imported with a weight column.
        """
        grid_info = getattr(self, 'current_grid_info', None)
        if not grid_info or grid_info.get('grid_type') != 'imported_from_xlsx':
            return False
        return self._get_current_grid_weights() is not None

    def _update_mn_grid_weight_info(self):
        """Updates grid-weight checkbox state and info label."""
        if not hasattr(self, 'mn_grid_weight_info_label'):
            return

        weights = self._get_current_grid_weights()
        available = self._grid_weights_available_for_optimization()

        self.mn_use_grid_weights.setEnabled(available)
        if not available:
            self.mn_use_grid_weights.blockSignals(True)
            self.mn_use_grid_weights.setChecked(False)
            self.mn_use_grid_weights.blockSignals(False)
            self.mn_grid_weight_info_label.setText(
                QCoreApplication.translate(
                    "Tab 4",
                    "Import an estimation grid with a weight column on tab 3 "
                    "to enable grid weighting.",
                )
            )
            return

        if self.mn_use_grid_weights.isChecked():
            self.mn_grid_weight_info_label.setText(
                QCoreApplication.translate(
                    "Tab 4",
                    "Using grid node weights (min={min_w:.3f}, max={max_w:.3f}, "
                    "mean={mean_w:.3f}).",
                ).format(
                    min_w=float(np.min(weights)),
                    max_w=float(np.max(weights)),
                    mean_w=float(np.mean(weights)),
                )
            )
        else:
            self.mn_grid_weight_info_label.setText(
                QCoreApplication.translate(
                    "Tab 4",
                    "Grid weights available (min={min_w:.3f}, max={max_w:.3f}); "
                    "not applied.",
                ).format(
                    min_w=float(np.min(weights)),
                    max_w=float(np.max(weights)),
                )
            )

    def _update_mn_well_weight_info(self):
        """Updates well-weight checkbox state and the detected-field info label."""
        if not hasattr(self, 'mn_well_weight_info_label'):
            return

        layer = self.input_data_layer.currentLayer()
        attr_name = self._current_mn_parameter()
        weight_field = None
        lookup_attr = attr_name
        if self._is_combined_mn_parameter(attr_name):
            selected = self._selected_analysis_parameters()
            lookup_attr = selected[0] if selected else None
        if layer and lookup_attr:
            weight_field = resolve_well_weight_field(layer, lookup_attr)

        has_field = weight_field is not None
        self.mn_use_well_weights.setEnabled(has_field)
        if not has_field:
            self.mn_use_well_weights.blockSignals(True)
            self.mn_use_well_weights.setChecked(False)
            self.mn_use_well_weights.blockSignals(False)
            self.mn_well_weight_info_label.setText(
                QCoreApplication.translate(
                    "Tab 4",
                    "No well weight field detected in the input layer.",
                )
            )
            return

        if self.mn_use_well_weights.isChecked():
            self.mn_well_weight_info_label.setText(
                QCoreApplication.translate(
                    "Tab 4",
                    "Using column «{field}» as individual well weight.",
                ).format(field=weight_field)
            )
        else:
            self.mn_well_weight_info_label.setText(
                QCoreApplication.translate(
                    "Tab 4",
                    "Weight column detected: «{field}» (not applied).",
                ).format(field=weight_field)
            )

    def _on_mn_parameter_changed(self, _index):
        """Refresh tab 4/5 views when the optimized parameter changes."""
        self._update_results_well_spinbox()
        self._update_mn_well_weight_info()
        self._update_mn_grid_weight_info()

        param_key = self._current_mn_parameter()
        if param_key and param_key in self.variance_results:
            results = self.variance_results[param_key]
            self._update_variance_reduction_plot(
                results,
                self._mn_parameter_display_name(param_key),
                param_key=param_key,
            )
            self._update_prioritization_order_table(
                results, param_key=param_key
            )

        self._refresh_ok_interpolation_plot()

    def _mn_combined_parameters_label(self):
        return QCoreApplication.translate("Tab 4", "Parameters combined")

    def _is_combined_mn_parameter(self, attr_name=None):
        if attr_name is None:
            if not hasattr(self, 'mn_param_select_combo'):
                return False
            attr_name = self.mn_param_select_combo.currentData()
        return attr_name == MN_COMBINED_PARAMETERS_KEY

    def _mn_parameter_display_name(self, attr_name):
        if self._is_combined_mn_parameter(attr_name):
            return self._mn_combined_parameters_label()
        return attr_name

    def _selected_analysis_parameters(self):
        if not hasattr(self, 'selected_data_parameters'):
            return []
        return [
            item.text()
            for item in self.selected_data_parameters.selectedItems()
        ]

    def _current_mn_parameter(self):
        """Returns tab-4 parameter key (name or MN_COMBINED_PARAMETERS_KEY), or None."""
        if (
            not hasattr(self, 'mn_param_select_combo')
            or not self.mn_param_select_combo.isEnabled()
        ):
            return None
        data = self.mn_param_select_combo.currentData()
        if data == MN_COMBINED_PARAMETERS_KEY:
            return MN_COMBINED_PARAMETERS_KEY
        if data:
            return str(data)
        text = self.mn_param_select_combo.currentText()
        if text == QCoreApplication.translate("Tab 4", "No parameter selected"):
            return None
        return text

    def _load_optimization_parameters(self, param_names):
        """
        Loads aligned well geometry and per-parameter variogram inputs.

        Returns:
            (coordinates, point_ids, well_weights, parameters, error_message)
            Any leading field is None when loading fails.
        """
        layer = self.input_data_layer.currentLayer()
        if not layer:
            return None, None, None, None, QCoreApplication.translate(
                "Tab 4", "Select an input point layer first."
            )

        if not param_names:
            return None, None, None, None, QCoreApplication.translate(
                "Tab 4", "Select a parameter to optimize."
            )

        parameters = []
        reference_coords = None
        point_ids = None
        well_weights = None

        for name in param_names:
            state = self._get_variogram_state(name)
            if not state:
                return None, None, None, None, QCoreApplication.translate(
                    "Tab 4",
                    "Run geostatistics on tab 2 for «{param}» first.",
                ).format(param=name)

            (
                raw_point_ids,
                coordinates,
                raw_values,
                _,
                raw_well_weights,
            ) = extract_point_records_from_layer(layer, name)
            if coordinates is None or raw_values is None:
                return None, None, None, None, QCoreApplication.translate(
                    "Tab 4",
                    "No valid data found for «{param}».",
                ).format(param=name)

            transform = 'log' if state.get('log_transform') else 'none'
            coordinates, values, _, error = align_coordinates_with_transform(
                coordinates, raw_values, transform
            )
            if error or coordinates is None or values is None:
                return None, None, None, None, error or QCoreApplication.translate(
                    "Tab 4",
                    "Could not align data for «{param}».",
                ).format(param=name)

            aligned_point_ids = align_point_ids_with_transform(
                raw_point_ids, raw_values, transform
            )

            if reference_coords is None:
                reference_coords = coordinates
                point_ids = aligned_point_ids
            elif (
                coordinates.shape != reference_coords.shape
                or not np.allclose(coordinates, reference_coords)
            ):
                return None, None, None, None, QCoreApplication.translate(
                    "Tab 4",
                    "Combined optimization requires the same valid wells for "
                    "every parameter (check null values and log transforms).",
                )

            if raw_well_weights is not None:
                aligned_well_weights = align_well_weights_with_transform(
                    raw_well_weights, raw_values, transform
                )
                if well_weights is None:
                    well_weights = aligned_well_weights
                elif not np.allclose(aligned_well_weights, well_weights):
                    return None, None, None, None, QCoreApplication.translate(
                        "Tab 4",
                        "Well weight values must align for every parameter.",
                    )

            parameters.append(
                ParameterInput(
                    name=name,
                    values=values,
                    model_type=state.get('model_type', 'spherical'),
                    nugget=float(state.get('nugget', 0)),
                    sill=float(state.get('sill', 0)),
                    range_val=float(state.get('range', 0)),
                    weight=float(state.get('weight', 1.0)),
                )
            )

        return reference_coords, point_ids, well_weights, parameters, None

    def _build_optimization_input(
        self,
        param_names,
        grid_coordinates,
        grid_weights=None,
        apply_well_weights=False,
    ):
        """
        Builds an OptimizationInput for one or more parameters.

        ``grid_coordinates`` is passed through unchanged (full tab-3 grid).

        Returns:
            (OptimizationInput, error_message). Input is None when validation fails.
        """
        (
            coordinates,
            point_ids,
            well_weights,
            parameters,
            error,
        ) = self._load_optimization_parameters(param_names)
        if error:
            return None, error

        if apply_well_weights:
            if well_weights is None:
                return None, QCoreApplication.translate(
                    "Tab 4",
                    "No well weight field is available for this layer.",
                )
        else:
            well_weights = None

        try:
            optimization_input = OptimizationInput(
                coordinates=coordinates,
                point_ids=point_ids,
                well_weights=well_weights,
                grid_coordinates=grid_coordinates,
                grid_weights=grid_weights,
                parameters=parameters,
            )
        except ValueError:
            return None, QCoreApplication.translate(
                "Tab 4",
                "Optimization input data are inconsistent.",
            )

        return optimization_input, None

    def _clear_variance_reduction_plot(self):
        """Resets the variance reduction plot and info label."""
        if not hasattr(self, 'variance_ax'):
            return

        self.variance_ax.clear()
        self.variance_ax.text(
            0.5,
            0.5,
            QCoreApplication.translate("Tab 4", "No variance reduction data"),
            ha='center',
            va='center',
            transform=self.variance_ax.transAxes,
            color='gray',
        )
        self.variance_ax.set_xticks([])
        self.variance_ax.set_yticks([])
        self.variance_canvas.draw()

        if hasattr(self, 'variance_info_label'):
            self.variance_info_label.setText(
                QCoreApplication.translate(
                    "Tab 4",
                    "Generate an estimation grid (tab 3) and fit a variogram "
                    "(tab 2), then click Optimize.",
                )
            )
            self.variance_info_label.setStyleSheet(
                "color: gray; font-style: italic;"
            )

        self._clear_prioritization_order_table()

    def _clear_prioritization_order_table(self):
        """Clears the tab 4 optimization-order table."""
        if not hasattr(self, 'mn_prioritization_order_table'):
            return
        self.mn_prioritization_order_table.setRowCount(0)

    def _update_prioritization_order_table(self, results, param_key=None):
        """Fills tab 4 optimization-order table (total variance % per well rank)."""
        if not hasattr(self, 'mn_prioritization_order_table'):
            return

        self._clear_prioritization_order_table()

        selection_order = results.get('selection_order')
        if selection_order is None:
            return

        if param_key is None:
            param_key = self._current_mn_parameter()

        point_ids = None
        well_weights = None
        if param_key is not None:
            param_data = self._get_mn_parameter_data(param_key)
            if param_data is not None:
                point_ids = param_data.get('point_ids')
                well_weights = param_data.get('well_weights')

        selection_order = np.asarray(selection_order, dtype=int)
        total_variance = resolve_total_variance_series(results)
        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        self.mn_prioritization_order_table.setRowCount(selection_order.size)

        for row, well_idx in enumerate(selection_order):
            rank = row + 1
            well_idx = int(well_idx)
            well_id = (
                str(point_ids[well_idx])
                if point_ids is not None
                else str(well_idx)
            )
            if well_weights is not None and well_idx < well_weights.size:
                weight_text = f"{float(well_weights[well_idx]):.3f}"
            else:
                weight_text = "1.000"

            if rank < total_variance.size:
                total_var_text = format_total_variance_pct(total_variance[rank])
            else:
                total_var_text = format_total_variance_pct(None)

            cells = [
                str(rank),
                well_id,
                weight_text,
                total_var_text,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(read_only)
                self.mn_prioritization_order_table.setItem(row, col, item)

        self.mn_prioritization_order_table.resizeColumnsToContents()

    def _update_variance_reduction_plot(self, results, display_name, param_key=None):
        """Plots remaining total variance (%) vs number of wells along Kalman order."""
        ax = self.variance_ax
        ax.clear()

        n_points = np.asarray(results['n_points'], dtype=float)
        total_variance = resolve_total_variance_series(results)

        ax.plot(
            n_points,
            total_variance,
            marker='^',
            color='#27ae60',
            linewidth=2,
            markersize=5,
            label=QCoreApplication.translate("Tab 4", "Prioritized wells"),
        )

        final_pct = float(total_variance[-1]) if total_variance.size else 0.0
        max_reduction_pct = 100.0 - final_pct
        if max_reduction_pct > 0:
            level_90 = 100.0 - 0.9 * max_reduction_pct
            level_95 = 100.0 - 0.95 * max_reduction_pct
            ax.axhline(
                level_90,
                linestyle='--',
                color='#e67e22',
                linewidth=1.2,
                label=QCoreApplication.translate(
                    "Tab 4", "90% of maximum reduction"
                ),
            )
            ax.axhline(
                level_95,
                linestyle='--',
                color='#c0392b',
                linewidth=1.2,
                label=QCoreApplication.translate(
                    "Tab 4", "95% of maximum reduction"
                ),
            )

        ax.set_xlabel(
            QCoreApplication.translate("Tab 4", "Number of monitoring points")
        )
        ax.set_ylabel(
            QCoreApplication.translate("Tab 4", "Total variance (%)")
        )
        ax.set_title(
            QCoreApplication.translate(
                "Tab 4", "Priorization order of wells– {param}"
            ).format(param=display_name)
        )
        ax.set_xlim(left=0)
        if total_variance.size:
            y_min = float(np.min(total_variance))
            y_max = float(np.max(total_variance))
            span = y_max - y_min
            margin = max(0.02 * span, 1.0) if span > 0 else 2.0
            ax.set_ylim(
                max(0.0, y_min - margin),
                min(100.0, y_max + margin),
            )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        if param_key is None:
            param_key = self._current_mn_parameter()

        selection_order = results.get('selection_order')
        point_ids = None
        if param_key is not None:
            param_data = self._get_mn_parameter_data(param_key)
            if param_data is not None:
                point_ids = param_data.get('point_ids')

        if selection_order is not None and point_ids is not None:
            for step, well_idx in enumerate(selection_order, start=1):
                if step >= len(n_points) or step >= total_variance.size:
                    break
                ax.annotate(
                    str(point_ids[well_idx]),
                    (n_points[step], total_variance[step]),
                    textcoords='offset points',
                    xytext=(4, 4),
                    fontsize=7,
                    ha='left',
                    va='bottom',
                )

        self.variance_figure.tight_layout()
        self.variance_canvas.draw()

    def _file_dialog_parent(self):
        """Parent widget for file dialogs (main window avoids Windows native crashes)."""
        if getattr(self, 'iface', None):
            main_window = self.iface.mainWindow()
            if main_window is not None:
                return main_window
        return self

    def _file_dialog_options(self):
        """Qt-only file dialogs are more stable than native dialogs inside QGIS."""
        return QFileDialog.DontUseNativeDialog

    def _prompt_save_xlsx(self, title, default_name):
        """Prompts for an .xlsx output path, or returns None if cancelled."""
        file_path, _ = QFileDialog.getSaveFileName(
            self._file_dialog_parent(),
            title,
            default_name,
            "Excel Files (*.xlsx)",
            options=self._file_dialog_options(),
        )
        if not file_path:
            return None
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'
        return file_path

    def _prompt_open_xlsx(self, title):
        """Prompts for an .xlsx input path, or returns None if cancelled."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._file_dialog_parent(),
            title,
            "",
            "Excel Files (*.xlsx *.xls)",
            options=self._file_dialog_options(),
        )
        return file_path or None

    def download_prioritization(self):
        """Exports Kalman prioritization and variogram settings to Excel."""
        attr_name = self._current_mn_parameter()
        if not attr_name:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Download prioritization"),
                QCoreApplication.translate(
                    "Tab 4", "Select a parameter to optimize."
                ),
            )
            return

        if attr_name not in self.variance_results:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Download prioritization"),
                QCoreApplication.translate(
                    "Tab 4",
                    "Run Optimize first to generate prioritization results.",
                ),
            )
            return

        default_name = (
            "prioritization_combined.xlsx"
            if self._is_combined_mn_parameter(attr_name)
            else f"prioritization_{attr_name}.xlsx"
        )
        file_path = self._prompt_save_xlsx(
            QCoreApplication.translate("Tab 4", "Save prioritization"),
            default_name,
        )
        if not file_path:
            return

        results = self.variance_results[attr_name]
        selection_order = np.asarray(results['selection_order'], dtype=int)
        total_variance = resolve_total_variance_series(results)

        if self._is_combined_mn_parameter(attr_name):
            (
                _coordinates,
                point_ids,
                _well_weights,
                _parameters,
                error,
            ) = self._load_optimization_parameters(
                self._selected_analysis_parameters()
            )
            if error:
                point_ids = None
            parameter_label = self._mn_combined_parameters_label()
        else:
            param_data = self._get_mn_parameter_data(attr_name)
            point_ids = param_data.get('point_ids') if param_data else None
            parameter_label = attr_name

        prioritization_rows = []
        for rank, well_idx in enumerate(selection_order, start=1):
            if rank >= total_variance.size:
                break
            well_id = (
                str(point_ids[well_idx])
                if point_ids is not None
                else str(well_idx)
            )
            prioritization_rows.append({
                'Rank': rank,
                'Well_ID': well_id,
                'Parameter': parameter_label,
                'Total_Variance_pct': float(total_variance[rank]),
            })

        prioritization_columns = [
            'Rank',
            'Well_ID',
            'Parameter',
            'Total_Variance_pct',
        ]

        variogram_rows = []
        for param_name in sorted(self.variogram_models_by_attribute.keys()):
            state = self._get_variogram_state(param_name)
            if not state:
                continue
            variogram_rows.append({
                'Parameter': param_name,
                'Weight': float(state.get('weight', 1.0)),
                'Transform': self._transform_display_text(
                    state.get('log_transform', False)
                ),
                'Model': state.get('model_type', ''),
                'Nugget': float(state.get('nugget', 0)),
                'Sill': float(state.get('sill', 0)),
                'Range': float(state.get('range', 0)),
                'R2': float(state.get('r2', 0)),
            })
        variogram_columns = [
            'Parameter',
            'Weight',
            'Transform',
            'Model',
            'Nugget',
            'Sill',
            'Range',
            'R2',
        ]

        try:
            # XlsxWriter export avoids openpyxl style-init crashes on some QGIS builds.
            write_excel_sheets(
                file_path,
                [
                    ('Prioritization', prioritization_columns, prioritization_rows),
                    ('Variogram_Settings', variogram_columns, variogram_rows),
                ],
            )
            QMessageBox.information(
                self,
                QCoreApplication.translate("Tab 4", "Download prioritization"),
                QCoreApplication.translate(
                    "Tab 4",
                    "Prioritization saved to:\n{path}",
                ).format(path=file_path),
            )
        except ImportError:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Download prioritization"),
                QCoreApplication.translate(
                    "Tab 4",
                    "XlsxWriter is required to export Excel files. "
                    "Install it in the QGIS Python environment "
                    "(pip install XlsxWriter).",
                ),
            )
        except Exception as exc:
            import traceback
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Download prioritization"),
                QCoreApplication.translate(
                    "Tab 4",
                    "Could not save the Excel file:\n{error}\n{details}",
                ).format(error=str(exc), details=traceback.format_exc()),
            )

    def _plot_wells_on_map_axes(
        self,
        ax,
        coordinates,
        label=None,
        point_ids=None,
        annotate_ids=False,
        zorder=6,
    ):
        """Draws monitoring-well markers on a matplotlib map axes."""
        if coordinates is None:
            return

        coords = np.asarray(coordinates, dtype=float)
        if coords.size == 0 or coords.shape[0] == 0:
            return

        if label is None:
            label = QCoreApplication.translate("Tab 3", "Wells")

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c='red',
            marker='s',
            label=label,
            s=20,
            alpha=0.95,
            edgecolors='black',
            linewidths=0.2,
            zorder=zorder,
        )

        if annotate_ids and point_ids is not None:
            self._annotate_well_id_labels(ax, coords, point_ids)

    def _annotate_well_id_labels(self, ax, coordinates, point_ids, indices=None):
        """Draws well ID text next to each point on a spatial axes."""
        if point_ids is None or coordinates is None:
            return

        coords = np.asarray(coordinates, dtype=float)
        if indices is None:
            indices = range(coords.shape[0])

        for idx in indices:
            ax.annotate(
                str(point_ids[idx]),
                (coords[idx, 0], coords[idx, 1]),
                textcoords='offset points',
                xytext=(4, 4),
                fontsize=7,
                ha='left',
                va='bottom',
            )

    def _get_mn_parameter_data(self, attr_name):
        """
        Loads aligned coordinates, values and variogram state for tab 4/5.

        For combined mode, returns the first selected parameter as the spatial
        reference (same well geometry); OK maps on tab 5 use its variogram.
        """
        if self._is_combined_mn_parameter(attr_name):
            param_names = self._selected_analysis_parameters()
            if len(param_names) < 2:
                return None
        else:
            param_names = [attr_name]

        (
            coordinates,
            point_ids,
            well_weights,
            parameters,
            error,
        ) = self._load_optimization_parameters(param_names)
        if error or not parameters:
            return None

        first_param = parameters[0]
        state = self._get_variogram_state(first_param.name)
        if not state:
            return None

        return {
            'coordinates': coordinates,
            'values': first_param.values,
            'point_ids': point_ids,
            'well_weights': well_weights,
            'state': state,
            'reference_parameter': first_param.name,
        }

    def optimize_monitoring_network(self):
        """
        Prioritizes wells with the Kalman filter.

        Phase 1 determines the selection order via rank-1 covariance updates;
        phase 2 evaluates normalized ordinary-kriging variance along that order.
        The full estimation grid from tab 3 is used.
        Single- and multi-parameter runs both use compute_variance_reduction_curve.
        """
        attr_name = self._current_mn_parameter()
        if not attr_name:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Optimize"),
                QCoreApplication.translate(
                    "Tab 4", "Select a parameter to optimize."
                ),
            )
            return

        if (
            not hasattr(self, 'current_grid_points')
            or self.current_grid_points is None
            or len(self.current_grid_points) == 0
        ):
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Optimize"),
                QCoreApplication.translate(
                    "Tab 4",
                    "Generate an estimation grid on tab 3 before optimizing.",
                ),
            )
            return

        use_well_weights = (
            hasattr(self, 'mn_use_well_weights')
            and self.mn_use_well_weights.isEnabled()
            and self.mn_use_well_weights.isChecked()
        )
        use_grid_weights = (
            hasattr(self, 'mn_use_grid_weights')
            and self.mn_use_grid_weights.isEnabled()
            and self.mn_use_grid_weights.isChecked()
        )
        grid_weights = (
            self._get_current_grid_weights() if use_grid_weights else None
        )
        if use_grid_weights and grid_weights is None:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Optimize"),
                QCoreApplication.translate(
                    "Tab 4",
                    "Grid weights are not available for the current estimation grid.",
                ),
            )
            return

        self.mn_optimize_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            if self._is_combined_mn_parameter(attr_name):
                param_names = self._selected_analysis_parameters()
                if len(param_names) < 2:
                    QMessageBox.warning(
                        self,
                        QCoreApplication.translate("Tab 4", "Optimize"),
                        QCoreApplication.translate(
                            "Tab 4",
                            "Select at least two parameters on tab 1 "
                            "for combined optimization.",
                        ),
                    )
                    return
                display_name = self._mn_combined_parameters_label()
            else:
                param_names = [attr_name]
                display_name = attr_name

            # Full estimation grid from tab 3.
            optimization_input, input_error = self._build_optimization_input(
                param_names,
                self.current_grid_points,
                grid_weights=grid_weights,
                apply_well_weights=use_well_weights,
            )
            if optimization_input is None:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 4", "Optimize"),
                    input_error,
                )
                return

            results = compute_variance_reduction_curve(optimization_input)
        except Exception as exc:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Optimize"),
                QCoreApplication.translate(
                    "Tab 4",
                    "Variance reduction could not be computed:\n{error}",
                ).format(error=str(exc)),
            )
            return
        finally:
            self.mn_optimize_btn.setEnabled(True)

        if results is None:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 4", "Optimize"),
                QCoreApplication.translate(
                    "Tab 4",
                    "At least one well and one grid node are required.",
                ),
            )
            return

        self.variance_results[attr_name] = results
        self.prioritization_grid_points = self.current_grid_points.copy()
        self._update_variance_reduction_plot(
            results, display_name, param_key=attr_name
        )
        self._update_prioritization_order_table(results, param_key=attr_name)

        n_wells = int(results['n_points'][-1])
        grid_nodes = results['grid_node_count']
        final_reduction = float(results['variance_reduction'][-1])
        weight_note = ""
        if use_well_weights:
            weight_note += " " + QCoreApplication.translate(
                "Tab 4", "(using personalized well weights)"
            )
        if use_grid_weights:
            weight_note += " " + QCoreApplication.translate(
                "Tab 4", "(using weighted estimation grid)"
            )
        if results.get('combined_mode'):
            param_weights = results.get('parameter_weights', {})
            weight_parts = [
                f"{name}={weight:.2f}"
                for name, weight in sorted(param_weights.items())
            ]
            weight_note += " " + QCoreApplication.translate(
                "Tab 4",
                "(parameter weights: {weights})",
            ).format(weights=", ".join(weight_parts))

        self.variance_info_label.setText(
            QCoreApplication.translate(
                "Tab 4",
                "Kalman ordering for «{param}»: {n} wells, "
                "{grid} grid nodes, {red:.1f}% variance reduction.{weights}",
            ).format(
                param=display_name,
                n=n_wells,
                grid=grid_nodes,
                red=final_reduction,
                weights=weight_note,
            )
        )
        self.variance_info_label.setStyleSheet("color: gray; font-style: italic;")

        self._update_results_well_spinbox()
        self._refresh_ok_interpolation_plot()

    def _on_tab2_attribute_changed(self, _index):
        """Restore variogram view when switching the active parameter on tab 2."""
        if not self.selected_attr_combo.isEnabled():
            return
        attr = self._current_analysis_attribute()
        if not attr or not self._get_variogram_state(attr):
            return
        widget_data = getattr(self.variogram_widget, 'current_data', None)
        if widget_data and widget_data.get('attribute') == attr:
            self._apply_store_to_widget(attr)
            self.variogram_widget.update_plot(emit_signal=False)

    def calculate_geostatistics(self):
        """Calculate geostatistics"""
        self.refresh_attribute_statistics(force_reextract=True)

    def _variogram_data_fingerprint(self, attr_name, log_transform, values):
        values = np.asarray(values, dtype=float)
        return (attr_name, bool(log_transform), int(values.size), float(values.sum()))

    def _transform_display_text(self, log_transform):
        if log_transform:
            return QCoreApplication.translate("Tab 2", "Logarithmic")
        return QCoreApplication.translate("Tab 2", "None")

    def _create_var_model_combo(self, model_type='spherical', row=None):
        combo = WheelIgnoringComboBox()
        combo.addItems(VARIOGRAM_MODEL_TYPES)
        idx = combo.findText(model_type)
        combo.blockSignals(True)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        combo.currentIndexChanged.connect(
            lambda _index, r=row: self._on_var_params_model_changed(r)
        )
        return combo

    def _on_var_params_model_changed(self, row):
        """Refit variogram when the model type changes in var_params_table."""
        if self._syncing_variogram:
            return

        param_item = self.var_params_table.item(row, VAR_COL_PARAM)
        combo = self.var_params_table.cellWidget(row, VAR_COL_MODEL)
        if not param_item or combo is None:
            return

        attribute = param_item.text()
        model_type = combo.currentText()
        state = dict(self._get_variogram_state(attribute) or {})
        state['model_type'] = model_type
        self._set_variogram_state(attribute, state)

        if attribute != self._current_analysis_attribute():
            return

        widget_data = getattr(self.variogram_widget, 'current_data', None)
        if not widget_data or widget_data.get('attribute') != attribute:
            return

        self._apply_store_to_widget(attribute)
        self.variogram_widget.auto_fit()
        self._refresh_tab2_cross_validation(attribute)
        self._refresh_tab5_views_after_variogram_change(attribute)

    def _get_coordinates_and_transformed_values_for_analysis(self):
        """
        Returns coordinates, transformed values and point IDs for tab 2 analysis.

        Reuses cached layer extraction and applies the current transform choice.
        """
        if (
            self._cached_coordinates is None
            or self._cached_raw_values is None
            or self._cached_point_ids is None
        ):
            return None, None, None

        transform = (
            'log' if self.stats_transform_combo.currentIndex() == 1 else 'none'
        )
        coordinates, values, _, error = align_coordinates_with_transform(
            self._cached_coordinates,
            self._cached_raw_values,
            transform,
        )
        if error or coordinates is None:
            return None, None, None

        point_ids = align_point_ids_with_transform(
            self._cached_point_ids,
            self._cached_raw_values,
            transform,
        )
        return coordinates, values, point_ids

    def _get_variogram_state(self, attr_name):
        return self.variogram_models_by_attribute.get(attr_name)

    def _set_variogram_state(self, attr_name, state):
        self.variogram_models_by_attribute[attr_name] = state

    def _apply_store_to_widget(self, attr_name):
        """Push stored variogram parameters to the widget for the active attribute."""
        state = self._get_variogram_state(attr_name)
        if not state or not hasattr(self, 'variogram_widget'):
            return
        self._syncing_variogram = True
        try:
            self.variogram_widget.set_parameters(
                model_type=state.get('model_type'),
                nugget=state.get('nugget'),
                sill=state.get('sill'),
                range_val=state.get('range'),
                r2=state.get('r2'),
            )
        finally:
            self._syncing_variogram = False

    def _write_var_params_table_row(self, row, attr_name, state):
        """Write one row of var_params_table from stored state."""
        if not state:
            return
        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        editable = read_only | Qt.ItemIsEditable

        def set_cell(col, text, flags=read_only):
            item = self.var_params_table.item(row, col)
            if item is None:
                item = QTableWidgetItem(str(text))
                self.var_params_table.setItem(row, col, item)
            else:
                item.setText(str(text))
            item.setFlags(flags)

        set_cell(VAR_COL_PARAM, attr_name)
        set_cell(VAR_COL_WEIGHT, f"{state.get('weight', 1.0):.2f}", editable)
        set_cell(
            VAR_COL_TRANSFORM,
            self._transform_display_text(state.get('log_transform', False)),
        )
        self.var_params_table.removeCellWidget(row, VAR_COL_MODEL)
        model_combo = self._create_var_model_combo(
            state.get('model_type', 'spherical'),
            row=row,
        )
        self.var_params_table.setCellWidget(row, VAR_COL_MODEL, model_combo)
        set_cell(VAR_COL_NUGGET, f"{state.get('nugget', 0):.3f}", editable)
        set_cell(VAR_COL_SILL, f"{state.get('sill', 0):.3f}", editable)
        set_cell(VAR_COL_RANGE, f"{state.get('range', 0):.2f}", editable)
        set_cell(VAR_COL_R2, f"{state.get('r2', 0):.3f}")

    def _sync_var_params_table_from_store(self, attributes=None):
        """Refresh var_params_table from variogram_models_by_attribute."""
        if not hasattr(self, 'var_params_table'):
            return
        if attributes is None:
            attributes = list(self.variogram_models_by_attribute.keys())
        self._syncing_variogram = True
        self.var_params_table.blockSignals(True)
        try:
            self.var_params_table.setRowCount(len(attributes))
            for row, attr_name in enumerate(attributes):
                self._write_var_params_table_row(
                    row, attr_name, self._get_variogram_state(attr_name)
                )
            self.var_params_table.resizeColumnsToContents()
        finally:
            self.var_params_table.blockSignals(False)
            self._syncing_variogram = False

    def _on_variogram_widget_params_changed(self, params):
        """Save fitted/adjusted widget parameters and sync the params table."""
        if self._syncing_variogram:
            return
        attr = self._current_analysis_attribute()
        if not attr:
            return

        log_transform = self.stats_transform_combo.currentIndex() == 1
        existing = self._get_variogram_state(attr) or {}
        values = None
        if self.variogram_widget.current_data:
            values = self.variogram_widget.current_data.get('values')

        fingerprint = existing.get('data_fingerprint')
        if values is not None:
            fingerprint = self._variogram_data_fingerprint(
                attr, log_transform, values
            )

        state = {
            'model_type': params.get('model', 'spherical'),
            'nugget': float(params.get('nugget', 0)),
            'sill': float(params.get('sill', 0)),
            'range': float(params.get('range', 0)),
            'r2': float(self.variogram_widget.r2_label.text() or 0),
            'weight': float(existing.get('weight', 1.0)),
            'log_transform': log_transform,
            'data_fingerprint': fingerprint,
        }
        self._set_variogram_state(attr, state)
        self._sync_var_params_table_from_store()
        self._refresh_tab2_cross_validation(attr)
        self._refresh_tab5_views_after_variogram_change(attr)

    def _update_store_from_table_cell(self, attr_name, col, text):
        """Update variogram_models_by_attribute from an edited table cell."""
        state = dict(self._get_variogram_state(attr_name))
        if col == VAR_COL_WEIGHT:
            state['weight'] = float(text)
        elif col == VAR_COL_NUGGET:
            state['nugget'] = float(text)
        elif col == VAR_COL_SILL:
            state['sill'] = float(text)
        elif col == VAR_COL_RANGE:
            state['range'] = float(text)
        else:
            return
        self._set_variogram_state(attr_name, state)

    def on_params_table_changed(self, item):
        """Sync manual table edits back to store and the variogram widget."""
        if self._syncing_variogram:
            return

        row = item.row()
        col = item.column()
        if col in (
            VAR_COL_PARAM,
            VAR_COL_TRANSFORM,
            VAR_COL_MODEL,
            VAR_COL_R2,
        ):
            return

        param_item = self.var_params_table.item(row, VAR_COL_PARAM)
        if not param_item:
            return
        attribute = param_item.text()
        if attribute not in self.variogram_models_by_attribute:
            return

        self._syncing_variogram = True
        try:
            self._update_store_from_table_cell(attribute, col, item.text())
            if attribute == self._current_analysis_attribute():
                self._apply_store_to_widget(attribute)
                self.variogram_widget.update_plot(emit_signal=False)
            if col in (VAR_COL_NUGGET, VAR_COL_SILL, VAR_COL_RANGE):
                if attribute == self._current_analysis_attribute():
                    self._refresh_tab2_cross_validation(attribute)
                self._refresh_tab5_views_after_variogram_change(attribute)
        except (ValueError, TypeError):
            pass
        finally:
            self._syncing_variogram = False

    def on_attribute_changed(self):
        """Handles the selection of attributes in the data selection tab (supports multi-selection)."""
        self._sync_attr_name_combo()
        self._clear_variogram_widget()
        if not self.selected_data_parameters.selectedItems():
            self.clear_stats_table()

    def clear_stats_table(self):
        """Clears statistics table and resets the info label."""
        for col in range(STATS_VALUE_COL_OFFSET, self.stats_table.columnCount()):
            item = self.stats_table.item(0, col)
            if item is None:
                item = QTableWidgetItem("")
                self.stats_table.setItem(0, col, item)
            else:
                item.setText("")
        self.stats_info_label.setText(
            QCoreApplication.translate(
                "Tab 2", "Select an attribute to view its statistics."
            )
        )
        self.stats_info_label.setStyleSheet("color: gray; font-style: italic;")
        self._clear_stats_plots()
        self._clear_cross_validation()

    def _clear_cross_validation(self):
        """Resets cross-validation tables, info label and scatter plot."""
        if not hasattr(self, 'cv_summary_table'):
            return

        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        for col in range(self.cv_summary_table.columnCount()):
            item = self.cv_summary_table.item(0, col)
            if item is None:
                item = QTableWidgetItem("")
                self.cv_summary_table.setItem(0, col, item)
            else:
                item.setText("")
            item.setFlags(read_only)

        self.cv_results_table.setRowCount(0)
        self.cv_info_label.setText(
            QCoreApplication.translate(
                "Tab 2",
                "Run geostatistics to compute leave-one-out cross-validation.",
            )
        )
        self.cv_info_label.setStyleSheet("color: gray; font-style: italic;")

        self.cv_ax.clear()
        self.cv_ax.text(
            0.5,
            0.5,
            QCoreApplication.translate("Tab 2", "No cross-validation data"),
            ha='center',
            va='center',
            transform=self.cv_ax.transAxes,
            color='gray',
        )
        self.cv_ax.set_xticks([])
        self.cv_ax.set_yticks([])
        self.cv_canvas.draw()

    def _update_cross_validation_plot(self, measured, predicted, attr_name):
        """Measured vs predicted scatter with a 1:1 reference line."""
        measured = np.asarray(measured, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        ax = self.cv_ax
        ax.clear()

        ax.scatter(
            measured,
            predicted,
            alpha=0.7,
            color='#4a90d9',
            edgecolors='k',
            linewidths=0.5,
            label=QCoreApplication.translate("Tab 2", "Points"),
        )

        if measured.size > 0:
            lo = float(min(measured.min(), predicted.min()))
            hi = float(max(measured.max(), predicted.max()))
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            ax.plot(
                [lo, hi],
                [lo, hi],
                linestyle='--',
                color='#c0392b',
                linewidth=1.5,
                label=QCoreApplication.translate("Tab 2", "1:1 line"),
            )

        ax.set_xlabel(
            QCoreApplication.translate("Tab 2", "Measured Value")
        )
        ax.set_ylabel(
            QCoreApplication.translate("Tab 2", "Predicted Value")
        )
        ax.set_title(
            QCoreApplication.translate(
                "Tab 2", "Cross-Validation – {param}"
            ).format(param=attr_name)
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        self.cv_figure.tight_layout()
        self.cv_canvas.draw()

    def _reset_stats_map_axes(self):
        """Recreates the map axes (avoids colorbar.remove() issues on refresh)."""
        self.stats_map_figure.clear()
        self.stats_map_ax = self.stats_map_figure.add_subplot(111)
        self.stats_map_colorbar = None

    def _clear_stats_plots(self):
        """Clears histogram and spatial map placeholders."""
        self.stats_hist_ax.clear()
        no_data = QCoreApplication.translate("Tab 2", "No data")
        self.stats_hist_ax.text(
            0.5, 0.5, no_data,
            ha='center', va='center',
            transform=self.stats_hist_ax.transAxes, color='gray',
        )
        self.stats_hist_ax.set_xticks([])
        self.stats_hist_ax.set_yticks([])
        self.stats_hist_canvas.draw()

        self._reset_stats_map_axes()
        self.stats_map_ax.text(
            0.5, 0.5, no_data,
            ha='center', va='center',
            transform=self.stats_map_ax.transAxes, color='gray',
        )
        self.stats_map_ax.set_xticks([])
        self.stats_map_ax.set_yticks([])
        self.stats_map_canvas.draw()

    def _update_stats_plots(self, coordinates, values, attr_name):
        """Updates histogram and spatial scatter for the statistics data."""
        values = np.asarray(values, dtype=float)
        ax = self.stats_hist_ax
        ax.clear()

        ax.hist(
            values, bins='auto', color='#4a90d9', edgecolor='white', alpha=0.85,
        )

        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        ax.axvline(
            mean_val, color='#c0392b', linestyle='--', linewidth=1.5,
            label=QCoreApplication.translate(
                "Tab 2", "Mean: {value:.4f}"
            ).format(value=mean_val),
        )
        ax.axvline(
            median_val, color='#d35400', linestyle='-', linewidth=1.5,
            label=QCoreApplication.translate(
                "Tab 2", "Median: {value:.4f}"
            ).format(value=median_val),
        )

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax > ymin else 1.0
        rug_base = ymin + y_range * 0.02
        rug_span = y_range * 0.06
        jitter = np.random.default_rng(0).uniform(0, rug_span, size=values.size)
        ax.scatter(
            values,
            rug_base + jitter,
            s=14,
            c='#2c3e50',
            alpha=0.55,
            edgecolors='none',
            zorder=5,
            label=QCoreApplication.translate("Tab 2", "Data"),
        )
        ax.set_ylim(ymin, ymax + rug_span * 1.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(
            QCoreApplication.translate(
                "Tab 2", "Histogram: {param}"
            ).format(param=attr_name)
        )
        ax.set_xlabel(QCoreApplication.translate("Tab 2", "Value"))
        ax.set_ylabel(QCoreApplication.translate("Tab 2", "Frequency"))
        self.stats_hist_figure.tight_layout()
        self.stats_hist_canvas.draw()

        self._reset_stats_map_axes()
        scatter = self.stats_map_ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            cmap='viridis',
            s=30,
            edgecolors='black',
            linewidths=0.3,
        )
        self.stats_map_colorbar = self.stats_map_figure.colorbar(
            scatter, ax=self.stats_map_ax, label=attr_name, fraction=0.046,
        )
        self.stats_map_ax.set_title(
            QCoreApplication.translate("Tab 2", "Spatial distribution")
        )
        self.stats_map_ax.set_xlabel(QCoreApplication.translate("Tab 2", "X"))
        self.stats_map_ax.set_ylabel(QCoreApplication.translate("Tab 2", "Y"))
        self.stats_map_figure.tight_layout()
        self.stats_map_canvas.draw()

    def _populate_stats_table(self, stats_dict):
        """Fills the single statistics row from a stats dictionary."""
        for col, key in enumerate(STAT_KEYS):
            table_col = col + STATS_VALUE_COL_OFFSET
            value = stats_dict[key]
            if key == 'count':
                text = str(int(value))
            elif np.isnan(value):
                text = "—"
            else:
                text = f"{value:.2f}"
            item = self.stats_table.item(0, table_col)
            if item is None:
                item = QTableWidgetItem(text)
                self.stats_table.setItem(0, table_col, item)
            else:
                item.setText(text)

    def refresh_attribute_statistics(self, force_reextract=False):
        """Recalculates and displays statistics for the selected attribute."""
        layer = self.input_data_layer.currentLayer()
        attr_name = self._current_analysis_attribute()

        if not layer or attr_name is None:
            self.clear_stats_table()
            return
        use_cache = (
            not force_reextract
            and self._cached_layer is layer
            and self._cached_attribute == attr_name
            and self._cached_raw_values is not None
            and self._cached_coordinates is not None
            and self._cached_point_ids is not None
        )

        if use_cache:
            raw_values = self._cached_raw_values
            coordinates = self._cached_coordinates
            point_ids = self._cached_point_ids
            null_count = self._cached_null_count
        else:
            point_ids, coordinates, raw_values, null_count, _ = (
                extract_point_records_from_layer(layer, attr_name)
            )
            self._cached_layer = layer
            self._cached_attribute = attr_name
            self._cached_raw_values = raw_values
            self._cached_coordinates = coordinates
            self._cached_point_ids = point_ids
            self._cached_null_count = null_count

        if (
            raw_values is None
            or coordinates is None
            or point_ids is None
            or len(raw_values) == 0
        ):
            self.clear_stats_table()
            self.stats_info_label.setText(
                QCoreApplication.translate(
                    "Tab 2",
                    "No valid data for the selected attribute.",
                )
            )
            self.stats_info_label.setStyleSheet("color: red; font-style: italic;")
            return

        transform = 'log' if self.stats_transform_combo.currentIndex() == 1 else 'none'
        coordinates, transformed, excluded_non_positive, error = (
            align_coordinates_with_transform(coordinates, raw_values, transform)
        )

        if error:
            self.clear_stats_table()
            self.stats_info_label.setText(
                QCoreApplication.translate("Tab 2", error)
            )
            self.stats_info_label.setStyleSheet("color: red; font-style: italic;")
            return

        stats_dict = compute_descriptive_stats(transformed)
        if stats_dict is None:
            self.clear_stats_table()
            self.stats_info_label.setText(
                QCoreApplication.translate(
                    "Tab 2",
                    "No valid data after applying the transformation.",
                )
            )
            self.stats_info_label.setStyleSheet("color: red; font-style: italic;")
            return

        self._populate_stats_table(stats_dict)
        self._update_stats_plots(coordinates, transformed, attr_name)

        info_parts = [
            QCoreApplication.translate(
                "Tab 2", "Statistics for «{param}»"
            ).format(param=attr_name)
        ]
        if null_count > 0:
            info_parts.append(
                QCoreApplication.translate(
                    "Tab 2", "{count} null value(s) omitted"
                ).format(count=null_count)
            )
        if transform == 'log' and excluded_non_positive > 0:
            info_parts.append(
                QCoreApplication.translate(
                    "Tab 2", "{count} value(s) ≤ 0 omitted for log"
                ).format(count=excluded_non_positive)
            )
        if transform == 'log':
            info_parts.append(
                QCoreApplication.translate("Tab 2", "(Log transformation)")
            )

        self.stats_info_label.setText(" · ".join(info_parts))
        self.stats_info_label.setStyleSheet("color: gray; font-style: italic;")

        log_transform = transform == 'log'
        existing_state = self._get_variogram_state(attr_name)
        if existing_state is not None:
            synced_state = dict(existing_state)
            synced_state['log_transform'] = log_transform
            self._set_variogram_state(attr_name, synced_state)

        self._update_variogram_from_statistics(
            coordinates, transformed, attr_name, log_transform, force_reextract
        )

        self._refresh_tab2_cross_validation(attr_name)

    def _update_variogram_from_statistics(
        self, coordinates, values, attr_name, log_transform, force_reextract
    ):
        """Loads data into the variogram widget and fits or restores parameters."""
        if not hasattr(self, 'variogram_widget'):
            return

        fingerprint = self._variogram_data_fingerprint(
            attr_name, log_transform, values
        )
        state = self._get_variogram_state(attr_name)
        needs_fit = (
            force_reextract
            or state is None
            or state.get('data_fingerprint') != fingerprint
        )

        self.variogram_widget.set_data(
            coordinates,
            values,
            attr_name,
            log_transform=log_transform,
            auto_fit=needs_fit,
        )

        if state and not needs_fit:
            self._apply_store_to_widget(attr_name)
            self.variogram_widget.update_plot(emit_signal=False)

    def _find_params_table_row(self, attr_name):
        for row in range(self.var_params_table.rowCount()):
            item = self.var_params_table.item(row, 0)
            if item and item.text() == attr_name:
                return row
        return -1

    def _set_params_table_cell(self, row, col, text):
        item = self.var_params_table.item(row, col)
        if item is None:
            self.var_params_table.setItem(row, col, QTableWidgetItem(str(text)))
        else:
            item.setText(str(text))

    def on_spacing_changed(self, value):
        """Handles spacing change and updates point estimate"""
        self.update_estimated_points()

    def update_estimated_points(self):
        """Updates the estimated total node count from the current spacing."""
        layer = self.input_data_layer.currentLayer()
        if not layer or not hasattr(self, 'spacing_spin') or not hasattr(self, 'estimated_points_label'):
            return
        try:
            extent = layer.extent()
            width = extent.width()
            height = extent.height()
            spacing = self.spacing_spin.value()
            buffer = self.buffer_spin.value() if hasattr(self, 'buffer_spin') else 0
            
            if spacing > 0 and width > 0 and height > 0:
                # Adjust dimensions with buffer
                width_with_buffer = width + (2 * buffer)
                height_with_buffer = height + (2 * buffer)
                
                # Calculate number of nodes based on spacing
                n_x = max(1, int(width_with_buffer / spacing) + 1)
                n_y = max(1, int(height_with_buffer / spacing) + 1)
                n_nodes = n_x * n_y
                
                self.estimated_points_label.setText(
                    QCoreApplication.translate(
                        "Tab 3",
                        "{count} points ({n_x} × {n_y})",
                    ).format(count=n_nodes, n_x=n_x, n_y=n_y)
                )
            else:
                self.estimated_points_label.setText("N/A")
        except Exception:
            self.estimated_points_label.setText(
                QCoreApplication.translate("Tab 3", "Could not calculate estimate")
            )
    
    def preview_grid(self):
        """
        Previews the estimation grid by generating nodes over a concave hull
        that covers the input points, using the defined spacing and buffer.
        Uses the QGIS concave hull algorithm with ALPHA = 0.7 and no holes.
        """
        layer = self.input_data_layer.currentLayer()
        if not layer:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate("Tab 3", "Select a layer first."),
            )
            return

        try:
            import numpy as np
            from shapely.geometry import Point, Polygon
            from shapely.ops import unary_union

            buffer = self.buffer_spin.value()
            spacing = self.spacing_spin.value()

            # Input well/point coordinates
            data_points = extract_layer_coordinates(layer)
            if data_points is None or len(data_points) == 0:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3", "The selected layer has no points."
                    ),
                )
                return

            # Build hull using the QGIS concave hull algorithm
            if len(data_points) <= 2:
                # Cannot build a hull; use a buffered bounding box
                extent = layer.extent()
                minx, miny, maxx, maxy = extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()
                minx -= buffer
                miny -= buffer
                maxx += buffer
                maxy += buffer
                hull_polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            else:
                # Temporary point layer for the QGIS concave hull tool
                from qgis.core import (
                    QgsVectorLayer,
                    QgsField,
                    QgsFeature,
                    QgsFields,
                    QgsPointXY,
                    QgsGeometry,
                    QgsProject,
                    QgsProcessingFeedback,
                    QgsWkbTypes
                )
                import processing

                # In-memory temporary point layer
                point_layer = QgsVectorLayer("Point?crs={}".format(layer.crs().authid()), "temp_points", "memory")
                prov = point_layer.dataProvider()
                features = []
                for xy in data_points:
                    feat = QgsFeature()
                    feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(xy[0]), float(xy[1]))))
                    features.append(feat)
                prov.addFeatures(features)
                point_layer.updateExtents()

                # Run concave hull with alpha = 0.7, without holes
                alg_params = {
                    'INPUT': point_layer,
                    'ALPHA': 0.7,
                    'HOLES': False,
                    'OUTPUT': 'memory:concave_hull'
                }
                feedback = QgsProcessingFeedback()
                hull_result = processing.run("qgis:concavehull", alg_params, feedback=feedback)
                hull_layer = hull_result['OUTPUT']

                # Hull polygon from the processing output
                hull_features = list(hull_layer.getFeatures())
                if hull_features:
                    hull_geom = hull_features[0].geometry()
                    # Convert to a Shapely polygon for further processing
                    if hull_geom.isMultipart():
                        qgs_polys = hull_geom.asMultiPolygon()
                        polys = [Polygon([(pt.x(), pt.y()) for pt in ring]) for ring in qgs_polys[0:]]
                        hull_polygon = unary_union(polys)
                    else:
                        qgs_poly = hull_geom.asPolygon()
                        if qgs_poly:
                            hull_polygon = Polygon([(pt.x(), pt.y()) for pt in qgs_poly[0]])
                        else:
                            # Fallback when QGIS-to-Shapely conversion fails
                            raise Exception(
                                QCoreApplication.translate(
                                    "Tab 3",
                                    "Could not convert the QGIS hull to a Shapely polygon.",
                                )
                            )
                else:
                    raise Exception(
                        QCoreApplication.translate(
                            "Tab 3",
                            "Could not create the concave hull with QGIS.",
                        )
                    )

                # Buffer the computed hull
                hull_polygon = hull_polygon.buffer(buffer)

            # Rectangular grid over the hull bounding box
            minx, miny, maxx, maxy = hull_polygon.bounds
            x_coords = np.arange(minx, maxx + spacing, spacing)
            y_coords = np.arange(miny, maxy + spacing, spacing)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

            # Keep only nodes inside the hull polygon (concave hull + buffer)
            shapely_points = [Point(xy) for xy in grid_points]
            mask_in_hull = np.array([hull_polygon.contains(pt) or hull_polygon.touches(pt) for pt in shapely_points])
            estimation_points = grid_points[mask_in_hull]
            n_nodes = len(estimation_points)

            # Store grid metadata for later workflow steps
            grid_info = {
                'grid_points': estimation_points,
                'n_nodes': n_nodes,
                'hull_polygon': hull_polygon,
                'spacing': spacing,
                'buffer': buffer,
                'n_x': len(x_coords),
                'n_y': len(y_coords),
                'weights': np.ones(n_nodes, dtype=float),
                'grid_type': 'generated',
            }
            self.current_grid_info = grid_info
            self.current_grid_points = estimation_points
            
            # Invalidate prioritization/results when the grid changes
            if hasattr(self, 'prioritization_grid_points'):
                # Detect grid changes by comparing node counts
                if (self.prioritization_grid_points is None or 
                    len(self.prioritization_grid_points) != len(estimation_points)):
                    self.prioritization_grid_points = None
                    if hasattr(self, 'variance_results'):
                        self.variance_results = {}
                        # Clear dependent visualizations
                        if hasattr(self, 'variance_ax'):
                            self._clear_variance_reduction_plot()
                            self.variance_info_label.setText(
                                QCoreApplication.translate(
                                    "Tab 4",
                                    "The estimation grid changed. Click Optimize again.",
                                )
                            )
                        self._clear_ok_interpolation_plots(
                            QCoreApplication.translate(
                                "Tab 5",
                                "The estimation grid changed. Re-run optimization on tab 4.",
                            )
                        )

            ceg_values = np.ones(len(estimation_points), dtype=int)
            self.current_ceg_values = ceg_values

            # Preview plot
            self.grid_figure.clear()
            ax = self.grid_figure.add_subplot(111)

            self._plot_wells_on_map_axes(
                ax,
                data_points,
                label=QCoreApplication.translate("Tab 3", "Wells"),
            )

            # Hull outline (QGIS concave hull with buffer)
            hull_x, hull_y = hull_polygon.exterior.xy
            ax.plot(
                hull_x,
                hull_y,
                color='blue',
                linestyle='--',
                linewidth=1,
                label=QCoreApplication.translate(
                    "Tab 3", "Hull (QGIS, buffered)"
                ),
            )

            # Estimation grid nodes
            if n_nodes > 0:
                ax.scatter(
                    estimation_points[:, 0], estimation_points[:, 1],
                    c='black', marker='o',
                    label=QCoreApplication.translate("Tab 3", "Estimation grid"),
                    s=5, alpha=0.7, edgecolors='black', linewidths=0.5
                )

            ax.set_xlabel(QCoreApplication.translate("Tab 3", "X"))
            ax.set_ylabel(QCoreApplication.translate("Tab 3", "Y"))
            ax.set_title(
                QCoreApplication.translate(
                    "Tab 3", "Estimation Grid on QGIS Concave Hull"
                )
            )
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            self.grid_canvas.draw()

            # Keep grid parameters for export
            self.grid_type_saved = 'Concave Hull (QGIS)'
            self.n_nodes_saved = n_nodes
            self.buffer_saved = buffer

            self._update_mn_grid_weight_info()

        except Exception as e:
            import traceback
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate(
                    "Tab 3",
                    "Could not create the estimation grid:\n{error}\n{details}",
                ).format(error=str(e), details=traceback.format_exc()),
            )

    def save_grid_as_layer(self):
        """Saves the generated grid as a temporary QGIS point layer."""
        if not hasattr(self, 'current_grid_points') or self.current_grid_points is None:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate(
                    "Tab 3",
                    "No generated grid is available. Preview the grid first.",
                ),
            )
            return
        
        layer = self.input_data_layer.currentLayer()
        if not layer:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate("Tab 3", "No layer is selected."),
            )
            return
        
        try:
            from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsField, QgsFields
            from qgis.PyQt.QtCore import QVariant
            from qgis.core import QgsProject
            
            grid_points = self.current_grid_points
            grid_info = getattr(self, 'current_grid_info', None)
            ceg_values = getattr(self, 'current_ceg_values', None)
            crs = layer.crs()
            
            # Temporary point layer
            layer_name = QCoreApplication.translate(
                "Tab 3", "Estimation Grid"
            )
            temp_layer = QgsVectorLayer(
                f"Point?crs={crs.authid()}", layer_name, "memory"
            )
            
            # Field definitions
            fields = QgsFields()
            fields.append(QgsField('ID', QVariant.Int))
            fields.append(QgsField('Coord_X', QVariant.Double))
            fields.append(QgsField('Coord_Y', QVariant.Double))
            fields.append(QgsField('Row', QVariant.Int))
            fields.append(QgsField('Column', QVariant.Int))
            fields.append(QgsField('CEG_Value', QVariant.Int))
            fields.append(QgsField('Grid_Type', QVariant.String))
            fields.append(QgsField('Node_Count', QVariant.Int))
            fields.append(QgsField('Buffer', QVariant.Double))
            
            # Saved or current grid parameters
            grid_type = getattr(self, 'grid_type_saved', 'Rectangular')
            n_nodes = getattr(self, 'n_nodes_saved', 0)
            buffer = getattr(self, 'buffer_saved', self.buffer_spin.value() if hasattr(self, 'buffer_spin') else 100)
            
            temp_layer.dataProvider().addAttributes(fields)
            temp_layer.updateFields()
            
            # Row/column indices when available
            row_col_indices = None
            if grid_info is not None and 'row_col_indices' in grid_info:
                row_col_indices = grid_info['row_col_indices']
            
            # Build features
            features = []
            for i, point in enumerate(grid_points):
                feature = QgsFeature()
                qgs_point = QgsPointXY(point[0], point[1])
                feature.setGeometry(QgsGeometry.fromPointXY(qgs_point))
                
                row = -1
                col = -1
                if row_col_indices is not None and i < len(row_col_indices):
                    row = int(row_col_indices[i][0])
                    col = int(row_col_indices[i][1])
                
                ceg_value = -1
                if ceg_values is not None and i < len(ceg_values):
                    ceg_value = int(ceg_values[i])
                
                attrs = [
                    i + 1,
                    float(point[0]),
                    float(point[1]),
                    row,
                    col,
                    ceg_value,
                    grid_type,
                    n_nodes,
                    float(buffer),
                ]
                
                feature.setAttributes(attrs)
                features.append(feature)
            
            temp_layer.dataProvider().addFeatures(features)
            temp_layer.updateExtents()
            
            # Add layer to the project
            QgsProject.instance().addMapLayer(temp_layer)
            
            QMessageBox.information(
                self,
                QCoreApplication.translate("Tab 3", "Success"),
                QCoreApplication.translate(
                    "Tab 3",
                    "Saved the estimation grid as a temporary layer with "
                    "{count} points.",
                ).format(count=len(features)),
            )
            
        except Exception as e:
            import traceback
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate(
                    "Tab 3",
                    "Could not save the estimation grid:\n{error}\n{details}",
                ).format(error=str(e), details=traceback.format_exc()),
            )
    
    def _draw_imported_estimation_grid(self, xs, ys, weights):
        """
        Draws an imported estimation grid with a discrete weight legend.

        Each unique grid weight is shown as a separate legend entry instead of
        a continuous color ramp.
        """
        self.grid_figure.clear()
        ax = self.grid_figure.add_subplot(111)

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        weights = np.asarray(weights, dtype=float)
        unique_weights = np.unique(weights)
        cmap = plt.get_cmap('viridis')
        n_unique = unique_weights.size

        for index, weight_val in enumerate(unique_weights):
            mask = weights == weight_val
            if not np.any(mask):
                continue
            color = cmap(index / max(n_unique - 1, 1))
            ax.scatter(
                xs[mask],
                ys[mask],
                c=[color],
                s=30,
                alpha=0.9,
                edgecolors='k',
                linewidths=0.3,
                label=f"{weight_val:g}",
                zorder=3,
            )

        layer = (
            self.input_data_layer.currentLayer()
            if hasattr(self, 'input_data_layer')
            else None
        )
        if layer is not None:
            well_coords = extract_layer_coordinates(layer)
            self._plot_wells_on_map_axes(ax, well_coords)

        ax.set_title(
            QCoreApplication.translate("Tab 3", "Imported Estimation Grid")
        )
        ax.set_xlabel(QCoreApplication.translate("Tab 3", "X"))
        ax.set_ylabel(QCoreApplication.translate("Tab 3", "Y"))
        ax.legend(
            title=QCoreApplication.translate("Tab 3", "Weight"),
            loc='best',
            fontsize=8,
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
        self.grid_canvas.draw()

    def load_grid_as_layer(self):
        """
        Load estimation grid from an .xlsx file and plot on Tab 3.
        Simply assumes the first 4 columns are id, x, y, weight;
        skips records that are incomplete or non-numeric in any of those fields.
        """
        try:
            file_path = self._prompt_open_xlsx(
                QCoreApplication.translate(
                    "Tab 3", "Select Estimation Grid Excel File"
                )
            )
            if not file_path:
                return

            try:
                table_rows = read_excel_table_rows(file_path)
            except ImportError as exc:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    str(exc),
                )
                return
            except Exception as e:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3", "Could not read Excel file:\n{error}"
                    ).format(error=str(e)),
                )
                return

            if not table_rows:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3", "The selected Excel file is empty."
                    ),
                )
                return

            if len(table_rows[0]) < 4:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Invalid File"),
                    QCoreApplication.translate(
                        "Tab 3",
                        "The selected file must have at least 4 columns for "
                        "ID, X, Y, and Weight (in that order).",
                    ),
                )
                return

            ids = []
            xs = []
            ys = []
            weights = []
            for row in table_rows[1:]:
                if row is None or len(row) < 4:
                    continue
                id_val, x_val, y_val, w_val = row[0], row[1], row[2], row[3]
                if (
                    id_val is None
                    or x_val is None
                    or y_val is None
                    or w_val is None
                ):
                    continue
                try:
                    x_f = float(x_val)
                    y_f = float(y_val)
                    w_f = float(w_val)
                except (ValueError, TypeError):
                    continue
                if np.isnan(x_f) or np.isnan(y_f) or np.isnan(w_f):
                    continue
                ids.append(id_val)
                xs.append(x_f)
                ys.append(y_f)
                weights.append(w_f)

            if not ids:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3",
                        "No valid records with ID, X, Y, and Weight found in "
                        "the file.",
                    ),
                )
                return

            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            weights = np.asarray(weights, dtype=float)
            grid_points = np.column_stack([xs, ys])
            n_nodes = grid_points.shape[0]

            # Save grid information for workflow integration
            self.current_grid_info = {
                'grid_points': grid_points,
                'n_nodes': n_nodes,
                'ids': np.asarray(ids),
                'weights': weights,
                'grid_type': "imported_from_xlsx",
                'buffer': 0.0,
                'n_x': None,  # unknown for imported grid
                'n_y': None
            }
            self.current_grid_points = grid_points

            self._update_mn_grid_weight_info()

            # For compatibility with CEG values if needed
            self.current_ceg_values = weights

            # Plot on Tab 3 (Estimation Grid Visualization)
            self._draw_imported_estimation_grid(xs, ys, weights)

            QMessageBox.information(
                self,
                QCoreApplication.translate("Tab 3", "Success"),
                QCoreApplication.translate(
                    "Tab 3",
                    "Grid loaded successfully with {count} points.\n"
                    "(used columns: first 4 columns of file)",
                ).format(count=n_nodes),
            )
        except Exception as e:
            import traceback
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate(
                    "Tab 3",
                    "Failed to load grid:\n{error}\n{details}",
                ).format(error=str(e), details=traceback.format_exc()),
            )

class VariogramWidget(QWidget):
    """Widget for variogram visualization and parameter adjustment."""

    params_changed = pyqtSignal(dict)

    def __init__(self, parent=None, dialog=None):
        super().__init__(parent)
        self.dialog = dialog  # Reference to the main dialog
        self.init_ui()
        self.current_data = None
        self.current_model = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model controls (hidden but functional)
        # Model selector
        self.model_combo = WheelIgnoringComboBox()
        self.model_combo.addItems(['spherical', 'exponential', 'gaussian', 'stable', 'matern'])
        self.model_combo.currentTextChanged.connect(self.on_params_changed)
        
        # Nugget
        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0, 200000)
        self.nugget_spin.setDecimals(3)
        self.nugget_spin.setSingleStep(0.1)
        self.nugget_spin.valueChanged.connect(self.on_params_changed)
        
        # Sill
        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setRange(0, 200000)
        self.sill_spin.setDecimals(3)
        self.sill_spin.setSingleStep(0.1)
        self.sill_spin.valueChanged.connect(self.on_params_changed)
        
        # Range
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(0, 2000000)
        self.range_spin.setDecimals(2)
        self.range_spin.setSingleStep(1)
        self.range_spin.valueChanged.connect(self.on_params_changed)
        
        # R²
        self.r2_label = QLabel("0.000")
        
        # Canvas for the plot
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250) #Fixed minimum Height
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def on_params_changed(self):
        """Emit signal when variogram parameters change."""
        if self.current_data is not None:
            params = {
                'model': self.model_combo.currentText(),
                'nugget': self.nugget_spin.value(),
                'sill': self.sill_spin.value(),
                'range': self.range_spin.value()
            }
            self.params_changed.emit(params)
            # Do not emit signal again in update_plot, it was already emitted above
            self.update_plot(emit_signal=False)
    
    def auto_fit(self):
        """Automatically fits the variogram model."""
        if self.current_data is None:
            return

        # Access progress elements from main dialog
        main_dialog = self.dialog
        progress_bar = getattr(main_dialog, 'var_progress_bar', None)
        status_label = getattr(main_dialog, 'var_progress_status_label', None)

        # Show progress bar and set initial status
        if progress_bar and status_label:
            progress_bar.setValue(0)
            status_label.setText(
                QCoreApplication.translate(
                    "Tab 2", "State: Starting variogram calculation..."
                )
            )
            QApplication.processEvents()

        try:
            coordinates = self.current_data['coordinates']
            values = self.current_data['values']
            model_type = self.model_combo.currentText()

            if progress_bar and status_label:
                progress_bar.setValue(10)
                status_label.setText(
                    QCoreApplication.translate(
                        "Tab 2", "State: Computing experimental variogram..."
                    )
                )
                QApplication.processEvents()

            bin_edges = np.linspace(0, self.current_data['max_dist']/2,15)
            bin_center, gamma = gs.vario_estimate(coordinates.T, values, bin_edges=bin_edges)
            
            valid_mask = np.isfinite(bin_center) & np.isfinite(gamma) & (gamma >= 0)
            if np.sum(valid_mask) < 3:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 2", "Error"),
                    QCoreApplication.translate(
                        "Tab 2",
                        "Not enough valid points in the experimental variogram "
                        "to fit the model.",
                    ),
                )
                return
            
            bin_center = bin_center[valid_mask]
            gamma = gamma[valid_mask]

            if progress_bar and status_label:
                progress_bar.setValue(30)
                status_label.setText(
                    QCoreApplication.translate(
                        "Tab 2", "State: Computing initial values..."
                    )
                )
                QApplication.processEvents()

            var_values = np.var(values)
            mean_gamma = np.mean(gamma) if len(gamma) > 0 else var_values
            max_gamma = np.max(gamma) if len(gamma) > 0 else var_values
            max_dist_used = np.max(bin_center) if len(bin_center) > 0 else self.current_data['max_dist']/2

            # Estimate initial values
            estimated_nugget = gamma[0] if len(gamma) > 0 else 0.0
            estimated_sill = max_gamma
            estimated_range = max_dist_used / 3.0  # Common approximation
            
            if progress_bar and status_label:
                progress_bar.setValue(50)
                status_label.setText(
                    QCoreApplication.translate(
                        "Tab 2", "State: Creating variogram model..."
                    )
                )
                QApplication.processEvents()

            if model_type == 'spherical':
                model = gs.Spherical(dim=2)
            elif model_type == 'exponential':
                model = gs.Exponential(dim=2)
            elif model_type == 'gaussian':
                model = gs.Gaussian(dim=2)
            elif model_type == 'stable':
                model = gs.Stable(dim=2)
            else:
                model = gs.Matern(dim=2)
            
            if progress_bar and status_label:
                progress_bar.setValue(70)
                status_label.setText(
                    QCoreApplication.translate(
                        "Tab 2", "State: Setting initial parameters..."
                    )
                )
                QApplication.processEvents()

            try:
                # Set initial values before fitting
                model.set_arg(nugget=estimated_nugget)
                model.len_scale = estimated_range
                model.var = max(0.01, estimated_sill - estimated_nugget)
            except:
                # If set_arg is not available, try setting directly
                try:
                    model.nugget = max(0.0, estimated_nugget)
                    model.len_scale = max(0.01, estimated_range)
                    model.var = max(0.01, estimated_sill - estimated_nugget)
                except:
                    pass

            # Initialize r2 with a default value
            r2 = 0.0

            if progress_bar and status_label:
                progress_bar.setValue(80)
                status_label.setText(
                    QCoreApplication.translate(
                        "Tab 2", "State: Fitting variogram model..."
                    )
                )
                QApplication.processEvents()

            try:
                try:
                    if progress_bar and status_label:
                        progress_bar.setValue(80)
                        status_label.setText(
                            QCoreApplication.translate(
                                "Tab 2", "State: Fitting variogram model..."
                            )
                        )
                        QApplication.processEvents()

                    para, pcov, r2 = model.fit_variogram(
                        bin_center,
                        gamma,
                        return_r2=True,
                        max_eval=5000  # Reduced from 10000 for better performance
                    )
                except TypeError:
                    # If max_eval is not available, try without it
                    try:
                        para, pcov, r2 = model.fit_variogram(
                            bin_center, 
                            gamma, 
                            return_r2=True
                        )
                    except Exception as e:
                        # If it still fails, use estimated values directly
                        raise e
            except Exception as fit_error:
                # If fitting fails, try with fewer constraints
                try:
                    # Reset initial values
                    try:
                        model.set_arg(nugget=estimated_nugget)
                        model.len_scale = estimated_range
                        model.var = max(0.01, estimated_sill - estimated_nugget)
                    except:
                        model.nugget = max(0.0, estimated_nugget)
                        model.len_scale = max(0.01, estimated_range)
                        model.var = max(0.01, estimated_sill - estimated_nugget)
                    
                    # Try simple fitting without additional parameters
                    para, pcov, r2 = model.fit_variogram(bin_center, gamma, return_r2=True)
                except Exception as fit_error2:
                    # If it still fails, use estimated values directly without optimization
                    try:
                        model.nugget = max(0.0, estimated_nugget)
                        model.len_scale = max(0.01, estimated_range)
                        model.var = max(0.01, estimated_sill - estimated_nugget)
                    except:
                        pass
                    r2 = 0.0  # R² unknown when no fitting
                    QMessageBox.information(
                        self,
                        QCoreApplication.translate("Tab 2", "Notice"),
                        QCoreApplication.translate(
                            "Tab 2",
                            "Automatic fitting could not be completed. Estimated "
                            "values were used instead.\n"
                            "Error: {error}\n"
                            "You can adjust the parameters manually.",
                        ).format(error=str(fit_error2)),
                    )
                    if progress_bar and status_label:
                        progress_bar.setValue(100)
                        status_label.setText(
                            QCoreApplication.translate(
                                "Tab 2", "State: Completed with estimated values"
                            )
                        )
                        QApplication.processEvents()

                        # Reset progress elements after a short delay
                        from qgis.PyQt.QtCore import QTimer
                        QTimer.singleShot(3000, lambda: self._hide_progress_elements())
            
            # Update spinboxes
            if progress_bar and status_label:
                progress_bar.setValue(95)
                status_label.setText(
                    QCoreApplication.translate(
                        "Tab 2", "State: Updating controls..."
                    )
                )
                QApplication.processEvents()

            self.nugget_spin.setValue(max(0.0, model.nugget))
            self.sill_spin.setValue(max(0.0, model.var + model.nugget))
            self.range_spin.setValue(max(0.01, model.len_scale))
            self.r2_label.setText(f"{r2:.3f}")
            self.current_model = model

            # Emit signal to update table
            params = {
                'model': model_type,
                'nugget': model.nugget,
                'sill': model.var + model.nugget,
                'range': model.len_scale
            }
            self.params_changed.emit(params)

            # Do not emit signal again, it was already emitted above
            self.update_plot(emit_signal=False)

            # Complete progress
            if progress_bar and status_label:
                progress_bar.setValue(100)
                status_label.setText(
                    QCoreApplication.translate(
                        "Tab 2", "State: Completed successfully"
                    )
                )
                QApplication.processEvents()

                # Reset progress elements after a short delay
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(2000, lambda: self._hide_progress_elements())
        except Exception as e:
            if progress_bar and status_label:
                progress_bar.setValue(0)
                status_label.setText(
                    QCoreApplication.translate("Tab 2", "State: Error")
                )
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 2", "Error"),
                QCoreApplication.translate(
                    "Tab 2", "Automatic fitting failed: {error}"
                ).format(error=str(e)),
            )

    def _hide_progress_elements(self):
        """Reset progress elements after completion."""
        main_dialog = self.dialog
        progress_bar = getattr(main_dialog, 'var_progress_bar', None)
        status_label = getattr(main_dialog, 'var_progress_status_label', None)

        if progress_bar:
            progress_bar.setValue(0)
        if status_label:
            status_label.setText(
                QCoreApplication.translate("Tab 2", "State: Ready")
            )

    def clear(self):
        """Clears data, model controls and plot when input layer/attributes change."""
        self.model_combo.blockSignals(True)
        self.nugget_spin.blockSignals(True)
        self.sill_spin.blockSignals(True)
        self.range_spin.blockSignals(True)
        try:
            self.current_data = None
            self.current_model = None
            self.model_combo.setCurrentText('spherical')
            self.nugget_spin.setValue(0.0)
            self.sill_spin.setValue(0.0)
            self.range_spin.setValue(0.0)
            self.r2_label.setText("0.000")
        finally:
            self.model_combo.blockSignals(False)
            self.nugget_spin.blockSignals(False)
            self.sill_spin.blockSignals(False)
            self.range_spin.blockSignals(False)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            QCoreApplication.translate("Tab 2", "No variogram data"),
            ha='center',
            va='center',
            transform=ax.transAxes,
            color='gray',
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def set_data(self, coordinates, values, attribute_name, log_transform=False, auto_fit=False):
        """Sets variogram input data."""
        from scipy.spatial import distance
        
        # Optimization: calculate max_dist more efficiently
        # For many points, use approximation based on bounding box
        if len(coordinates) > 100:
            x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
            y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
            max_dist = np.sqrt(x_range**2 + y_range**2)
        else:
            # For few points, calculate exact distances
            max_dist = np.max(distance.pdist(coordinates))
        
        self.current_data = {
            'coordinates': coordinates,
            'values': values,
            'attribute': attribute_name,
            'max_dist': max_dist,
            'log_transform': log_transform
        }
        
        # Only do automatic fitting if explicitly requested
        if auto_fit:
            self.auto_fit()
        # Do not update the plot here - it will be updated after setting the parameters
        # from the mathematical models table (if they exist)
    
    def set_parameters(self, model_type=None, nugget=None, sill=None, range_val=None, r2=None):
        """Updates variogram controls from the given parameters."""
        # Block signals to avoid unnecessary emissions
        self.model_combo.blockSignals(True)
        self.nugget_spin.blockSignals(True)
        self.sill_spin.blockSignals(True)
        self.range_spin.blockSignals(True)
        
        try:
            if model_type is not None:
                if model_type in ['spherical', 'exponential', 'gaussian', 'stable', 'matern']:
                    self.model_combo.setCurrentText(model_type)
            
            if nugget is not None:
                self.nugget_spin.setValue(float(nugget))
            
            if sill is not None:
                self.sill_spin.setValue(float(sill))
            
            if range_val is not None:
                self.range_spin.setValue(float(range_val))
            
            if r2 is not None:
                self.r2_label.setText(f"{float(r2):.3f}")
        finally:
            # Unblock signals
            self.model_combo.blockSignals(False)
            self.nugget_spin.blockSignals(False)
            self.sill_spin.blockSignals(False)
            self.range_spin.blockSignals(False)
    
    def update_plot(self, emit_signal=True):
        """Updates the variogram plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if self.current_data is not None:
            try:
                coordinates = self.current_data['coordinates']
                values = self.current_data['values']
                max_dist = self.current_data['max_dist']
                
                bin_edges = np.linspace(0, max_dist/2, 20)
                bin_center, gamma = gs.vario_estimate(coordinates.T, values, bin_edges=bin_edges)
                
                ax.scatter(
                    bin_center,
                    gamma,
                    label=QCoreApplication.translate("Tab 2", "Experimental"),
                    alpha=0.7,
                )
                
                # Plot theoretical model
                if self.current_model is not None or True:
                    # Create model with current parameters
                    model_type = self.model_combo.currentText()
                    nugget = self.nugget_spin.value()
                    var = self.sill_spin.value() - nugget
                    len_scale = self.range_spin.value()
                    
                    if model_type == 'spherical':
                        model = gs.Spherical(dim=2, var=var, len_scale=len_scale, nugget=nugget)
                    elif model_type == 'exponential':
                        model = gs.Exponential(dim=2, var=var, len_scale=len_scale, nugget=nugget)
                    elif model_type == 'gaussian':
                        model = gs.Gaussian(dim=2, var=var, len_scale=len_scale, nugget=nugget)
                    elif model_type == 'stable':
                        model = gs.Stable(dim=2, var=var, len_scale=len_scale, nugget=nugget, alpha=1.5)
                    else:
                        model = gs.Matern(dim=2, var=var, len_scale=len_scale, nugget=nugget, nu=1.5)
                    
                    # Save the updated model for use in other tabs
                    self.current_model = model
                    
                    # Emit signal to update table only if explicitly requested
                    # (for example, when parameters are changed manually or automatic fitting is done)
                    if emit_signal and self.current_data:
                        params = {
                            'model': model_type,
                            'nugget': nugget,
                            'sill': self.sill_spin.value(),
                            'range': len_scale
                        }
                        self.params_changed.emit(params)
                    
                    x_model = np.linspace(0, max_dist/2, 100)
                    y_model = model.variogram(x_model)
                    ax.plot(
                        x_model,
                        y_model,
                        'r-',
                        label=QCoreApplication.translate(
                            "Tab 2", "Model: {model}"
                        ).format(model=model_type),
                    )
                
                ax.set_xlabel(QCoreApplication.translate("Tab 2", "Distance"))
                ax.set_ylabel(QCoreApplication.translate("Tab 2", "Semivariance"))
                title = QCoreApplication.translate(
                    "Tab 2", "Variogram – {param}"
                ).format(param=self.current_data["attribute"])
                if self.current_data.get('log_transform', False):
                    title += QCoreApplication.translate(
                        "Tab 2", " (Log transform)"
                    )
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error updating plot: {str(e)}")
        
        self.canvas.draw()


class WheelIgnoringComboBox(QComboBox):
    """QComboBox that ignores mouse wheel events until it is open"""
    def wheelEvent(self, event):
        # Only process wheel event if combo box is open (dropdown visible)
        view = self.view()
        if view and view.isVisible():
            # If dropdown is open, allow normal scroll
            super().wheelEvent(event)
        else:
            # If dropdown is not open, ignore wheel event
            event.ignore()

class NoWheelFilter(QObject):
    """Filter the wheel mouse event (To prevent tab change)"""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            return True  # block wheel event
        return False
