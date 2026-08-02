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
# * Wells or
# * Parameters or
# * Areas of interest.
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
                                QApplication, QStyle,
                                QSizePolicy)
from qgis.core import (QgsProject, QgsVectorLayer, QgsMapLayerProxyModel,
                      QgsFieldProxyModel, QgsFeature, QgsGeometry, QgsPointXY,
                      QgsField, QgsFields, QgsWkbTypes, QgsVectorDataProvider)
from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox
from PyQt5.QtGui import QColor, QBrush
import numpy as np
import gstools as gs
from scipy import stats
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .monitoring_networks_analysis import (
    extract_layer_coordinates,
    extract_point_records_from_layer,
    build_point_id_layer_attribute_map,
    align_coordinates_with_transform,
    align_point_ids_with_transform,
    align_well_weights_with_transform,
    resolve_well_weight_field,
    compute_descriptive_stats,
    run_ordinary_kriging_cross_validation,
    compute_variance_reduction_curve,
    compute_total_variance_percent,
    wells_at_fraction_of_max_reduction,
    build_covariance_model,
    compute_variogram_lag_r2,
    validate_variogram_autofit,
    variogram_autofit_fallback_params,
    compute_nearest_neighbor_stats,
    build_default_experimental_variogram_settings,
    experimental_variogram_bin_edges,
    experimental_variogram_cutoff,
    clamp_variogram_factor_max_dist,
    compute_n_bins,
    lag_size_choices,
    ExperimentalVariogramSettings,
    VARIOGRAM_FACTOR_MAX_DIST_MIN,
    VARIOGRAM_FACTOR_MAX_DIST_MAX,
    OptimizationInput,
    ParameterInput,
    ordinary_kriging_interpolation,
    write_excel_sheets,
    read_excel_table_rows,
    STAT_KEYS,
)

# Tab 2 stats_table: per-parameter weight for tab 4 combined optimization
# (not the per-well weight checkbox on tab 4).
STATS_COL_WEIGHT = 0
STATS_TRANSFORM_COL = 1
STATS_VALUE_COL_OFFSET = 2

STATS_VALUE_HEADERS = (
    'Count', 'Min', 'Max', 'Mean', 'Median',
    'Std Dev', 'Variance', 'Asymmetry', 'Kurtosis',
)

STATS_COL_ASYMMETRY = STATS_VALUE_COL_OFFSET + STAT_KEYS.index('asymmetry')
STATS_COL_KURTOSIS = STATS_VALUE_COL_OFFSET + STAT_KEYS.index('kurtosis')
STATS_TABLE_MAX_VISIBLE_ROWS = 5

# Tab 2 distribution-shape highlight colors (skewness / excess kurtosis cells).
STATS_SKEWNESS_BG_SYMMETRIC = QColor('#d5f5e3')      # light green
STATS_SKEWNESS_BG_NEAR_SYMMETRIC = QColor('#fcf3cf')  # light yellow
STATS_SKEWNESS_BG_HIGH = QColor('#fadbd8')            # light red
STATS_KURTOSIS_BG_NORMAL = QColor('#d5f5e3')           # light green
STATS_KURTOSIS_BG_NON_NORMAL = QColor('#fdebd0')        # light orange

VAR_COL_PARAM = 0
VAR_COL_TRANSFORM = 1
VAR_COL_MODEL = 2
VAR_COL_NUGGET = 3
VAR_COL_SILL = 4
VAR_COL_RANGE = 5

VARIOGRAM_MODEL_TYPES = [
    'spherical', 'exponential', 'gaussian', 'stable', 'matern',
]

CV_SUMMARY_KEYS = ('min', 'max', 'mean', 'mae', 'rmse', 'ase', 'mse', 'rmsse')

CV_SUMMARY_FORMATS = {
    'min': '{:.4f}',
    'max': '{:.4f}',
    'mean': '{:.4f}',
    'mae': '{:.4f}',
    'rmse': '{:.4f}',
    'ase': '{:.4f}',
    'mse': '{:.4f}',
    'rmsse': '{:.4f}',
}

CV_COL_ID = 0
CV_COL_INCLUDED = 1
CV_COL_MEASURED = 2
CV_COL_PREDICTED = 3
CV_COL_ERROR = 4
CV_COL_SE = 5
CV_COL_STD_ERROR = 6

# Internal combo item data for tab 4 multi-parameter optimization (not translated).
MN_COMBINED_PARAMETERS_KEY = '_combined_param_'

def stats_value_header_labels():
    """Column headers for the tab 2 basic statistics table."""
    return [
        QCoreApplication.translate("Tab 2", header)
        for header in STATS_VALUE_HEADERS
    ]


def skewness_distribution_tooltip():
    """Tooltip text for the tab 2 skewness (asymmetry) statistic."""
    return QCoreApplication.translate(
        "Tab 2",
        "Skewness measures the lack of symmetry in the data distribution.\n\n"
        "Between -0.5 and 0.5: Symmetric distribution\n"
        "From -1 to -0.5 or from 0.5 to 1: Distribution close to symmetric\n"
        "Less than -1 or greater than 1: High skewness, significant deviation",
    )


def kurtosis_distribution_tooltip():
    """Tooltip text for the tab 2 excess kurtosis statistic."""
    return QCoreApplication.translate(
        "Tab 2",
        "Excess kurtosis measures how peaked or flattened a distribution is.\n\n"
        "Between -2 and 2: Normal distribution\n"
        "Greater than 2: Sharper peak and heavy tails (more outliers)\n"
        "Less than -2: Flatter peak and light tails (fewer outliers)",
    )


def skewness_cell_background(value):
    """Background color for a skewness cell, or None when not applicable."""
    if value is None or not np.isfinite(value):
        return None
    skew = float(value)
    if -0.5 <= skew <= 0.5:
        return STATS_SKEWNESS_BG_SYMMETRIC
    if (-1.0 <= skew < -0.5) or (0.5 < skew <= 1.0):
        return STATS_SKEWNESS_BG_NEAR_SYMMETRIC
    if skew < -1.0 or skew > 1.0:
        return STATS_SKEWNESS_BG_HIGH
    return None


def kurtosis_cell_background(value):
    """Background color for an excess kurtosis cell, or None when not applicable."""
    if value is None or not np.isfinite(value):
        return None
    kurt = float(value)
    if -2.0 <= kurt <= 2.0:
        return STATS_KURTOSIS_BG_NORMAL
    if kurt > 2.0 or kurt < -2.0:
        return STATS_KURTOSIS_BG_NON_NORMAL
    return None


def apply_distribution_shape_cell_style(item, stat_key, value):
    """Apply tooltip and background color to skewness / kurtosis table cells."""
    if stat_key == 'asymmetry':
        item.setToolTip(skewness_distribution_tooltip())
        background = skewness_cell_background(value)
    elif stat_key == 'kurtosis':
        item.setToolTip(kurtosis_distribution_tooltip())
        background = kurtosis_cell_background(value)
    else:
        item.setToolTip("")
        background = None

    if background is not None:
        item.setBackground(QBrush(background))
    else:
        item.setBackground(QBrush())

def cv_summary_header_labels(context_name):
    """Column headers for CV summary tables (tabs 2 and 5)."""
    return [
        QCoreApplication.translate(context_name, "Min error"),
        QCoreApplication.translate(context_name, "Max error"),
        QCoreApplication.translate(context_name, "Mean error"),
        QCoreApplication.translate(context_name, "MAE"),
        QCoreApplication.translate(context_name, "RMSE"),
        QCoreApplication.translate(context_name, "ASE"),
        QCoreApplication.translate(context_name, "MSE"),
        QCoreApplication.translate(context_name, "RMSSE"),
    ]


def cv_summary_header_tooltips(context_name):
    """Tooltips for CV summary headers (ArcGIS-style metric names)."""
    return {
        'ase': QCoreApplication.translate(
            context_name,
            "Average Standard Error (ASE): root mean square of the kriging "
            "standard errors. Ideally close to RMSE.",
        ),
        'mse': QCoreApplication.translate(
            context_name,
            "Mean Standardized Error (MSE): mean of error/SE. Ideally close "
            "to 0 (unbiased standardized residuals).",
        ),
        'rmsse': QCoreApplication.translate(
            context_name,
            "Root-Mean-Square Standardized Error (RMSSE): root mean square of "
            "error/SE. Ideally close to 1.",
        ),
    }


def apply_cv_summary_header_tooltips(summary_table, context_name):
    """Attach ASE/MSE/RMSSE tooltips to an existing CV summary header row."""
    tooltips = cv_summary_header_tooltips(context_name)
    for col, key in enumerate(CV_SUMMARY_KEYS):
        tip = tooltips.get(key)
        if not tip:
            continue
        header_item = summary_table.horizontalHeaderItem(col)
        if header_item is not None:
            header_item.setToolTip(tip)


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
        # Independent top-level window usable alongside QGIS and other instances.
        self.setWindowModality(Qt.NonModal)
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
        self.results_all_se_colorbar = None
        self.results_sel_se_colorbar = None
        self.results_se_color_limits = None
        # Well marker PathCollections on Tab 5 maps, keyed by canvas id.
        self._ok_map_well_scatters = {}
        self._syncing_variogram = False
        # Nested guard for programmatic stats_table updates (combos/weights).
        self._stats_table_sync_depth = 0
        # When True, batch auto-fit only writes the store (no CV / table refresh).
        self._batch_fitting_variograms = False
        # Suppress nested auto_fit progress messages during multi-parameter runs.
        self._suppress_variogram_progress = False
        # Tab 2 active parameter for plots/variogram (row selection in stats_table).
        self._active_analysis_attribute = None
        # Experimental variogram session settings + ANN cache (layer select).
        self._nn_stats = None
        self._experimental_variogram_settings_store = (
            ExperimentalVariogramSettings()
        )
        self._syncing_experimental_variogram_controls = False
        self.init_ui()
        self.current_grid_points = None  # Store grid generated in tab 3 (array of points)
        self.variogram_models_by_attribute = {}  # Dictionary to store models by attribute
        self.stats_by_attribute = {}  # Per-parameter descriptive stats and plot inputs
        self._layer_data_by_attribute = {}  # Per-parameter raw layer extraction cache
        self.variance_results = {}  # Variance reduction curves keyed by parameter
        # Last Tab 1 parameter selection used to invalidate optimization cache.
        self._tab1_params_for_optimization = frozenset()
        # Feature IDs included in analysis (Tab 1 checkboxes); None when no layer.
        self._included_well_fids = None
        self._syncing_well_include = False

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
        self.setup_map_tab()
        self.tabs.addTab(
            self.tab_results, 
            "5. " + QCoreApplication.translate("Main Window", "Map")
        )
        
        layout.addWidget(self.tabs)

        # Single shared progress bar for all tabs (value + in-bar status text).
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat(
            QCoreApplication.translate("Main Window", "State: Ready")
        )
        layout.addWidget(self.progress_bar)
        
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
        # Disable Next only on the last tab (Map); refresh Tab 1 progress hint.
        self.tabs.currentChanged.connect(self._on_main_tab_changed)
        # Tabs 2–5 stay visible but not clickable until Next advances the workflow.
        self._set_workflow_tabs_locked()
        self._update_tab1_progress_hint()
    
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

        layer_group.setLayout(layer_layout)
        data_tab_layout.addWidget(layer_group)
        
        # Attribute selector section
        attr_group = QGroupBox(QCoreApplication.translate("Tab 1", "Selection of Attributes for Analysis"))
        attr_layout = QVBoxLayout()
        attr_layout.addWidget(QLabel(QCoreApplication.translate("Tab 1", "Attributes available to use as parameters of the monitoring network (numeric):")))
        
        # List of selectable attributes
        self.selected_data_parameters = QListWidget()
        self.selected_data_parameters.setSelectionMode(QAbstractItemView.MultiSelection) #Allow multiple attribute selection
        self.selected_data_parameters.itemSelectionChanged.connect(self.on_attribute_changed)
        attr_layout.addWidget(self.selected_data_parameters)

        # Layer attribute values table
        attr_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 1", "Layer attribute values:")
        ))
        self.layer_fields_table = QTableWidget()
        self.layer_fields_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layer_fields_table.setAlternatingRowColors(True)
        self.layer_fields_table.setSortingEnabled(True)
        self.layer_fields_table.horizontalHeader().setStretchLastSection(True)
        self.layer_fields_table.setMinimumHeight(140)
        self.layer_fields_table.setMaximumHeight(280)
        self.layer_fields_table.itemChanged.connect(
            self._on_layer_fields_include_item_changed
        )
        attr_layout.addWidget(self.layer_fields_table)

        wells_select_row = QHBoxLayout()
        self.select_all_wells_btn = QPushButton(
            QCoreApplication.translate("Tab 1", "Select all wells")
        )
        self.select_all_wells_btn.clicked.connect(self._select_all_wells)
        self.deselect_all_wells_btn = QPushButton(
            QCoreApplication.translate("Tab 1", "Deselect all wells")
        )
        self.deselect_all_wells_btn.clicked.connect(self._deselect_all_wells)
        wells_select_row.addWidget(self.select_all_wells_btn)
        wells_select_row.addWidget(self.deselect_all_wells_btn)
        wells_select_row.addStretch()
        attr_layout.addLayout(wells_select_row)

        attr_group.setLayout(attr_layout)
        data_tab_layout.addWidget(attr_group)

        # Assign the layout to the data tab
        self.tab_data.setLayout(data_tab_layout)
        
    
    def setup_variogram_tab(self):
        """Setup the variogram tab"""

        # Layout for the variogram tab
        variogram_tab_layout = QVBoxLayout()

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
        table_headers = [
            QCoreApplication.translate("Tab 2", "Weight"),
            QCoreApplication.translate("Tab 2", "Transformation"),
        ] + stats_value_header_labels()
        self.stats_table.setColumnCount(len(table_headers))
        self.stats_table.setRowCount(0)
        self.stats_table.setHorizontalHeaderLabels(table_headers)
        weight_header = self.stats_table.horizontalHeaderItem(STATS_COL_WEIGHT)
        if weight_header is not None:
            weight_header.setToolTip(
                QCoreApplication.translate(
                    "Tab 2",
                    "Relative importance when combining parameters on tab 4. "
                    "Default: equal share (1/N) so weights sum to 1.",
                )
            )
        transform_header = self.stats_table.horizontalHeaderItem(
            STATS_TRANSFORM_COL
        )
        if transform_header is not None:
            transform_header.setToolTip(
                QCoreApplication.translate(
                    "Tab 2",
                    "Data transform for this parameter. Changing it recalculates "
                    "statistics and variogram for this parameter only.",
                )
            )
        asymmetry_header = self.stats_table.horizontalHeaderItem(
            STATS_COL_ASYMMETRY
        )
        if asymmetry_header is not None:
            asymmetry_header.setToolTip(skewness_distribution_tooltip())
        kurtosis_header = self.stats_table.horizontalHeaderItem(
            STATS_COL_KURTOSIS
        )
        if kurtosis_header is not None:
            kurtosis_header.setToolTip(kurtosis_distribution_tooltip())
        self.stats_table.verticalHeader().setVisible(True)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        # Column widths come from content; horizontal scroll is allowed when needed.
        self.stats_table.resizeColumnsToContents()
        self.stats_table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.SelectedClicked
        )
        # Active parameter is shown with bold text (not Qt selection fill),
        # so asymmetry/kurtosis cell colors stay visible.
        self.stats_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.stats_table.itemChanged.connect(self._on_stats_table_weight_changed)
        # Click the leftmost parameter-name header cell to activate that row.
        self.stats_table.verticalHeader().sectionClicked.connect(
            self._on_stats_table_parameter_header_clicked
        )
        self.stats_table.verticalHeader().setToolTip(
            QCoreApplication.translate(
                "Tab 2",
                "Click a parameter name to show its geostatistics, "
                "variogram and cross-validation.",
            )
        )
        self._stats_table_height_filter = StatsTableHeightFilter(self)
        self.stats_table.installEventFilter(self._stats_table_height_filter)
        self._adjust_stats_table_height()

        stats_layout.addWidget(self.stats_table)

        # Informative label for statistics / active parameter views
        self.stats_info_label = QLabel(
            QCoreApplication.translate(
                "Tab 2",
                "Select attributes on tab 1, then click Next to calculate "
                "geostatistics. Click a parameter name in the table to "
                "inspect plots and the variogram.",
            )
        )
        self.stats_info_label.setStyleSheet("color: gray; font-style: italic;")
        stats_layout.addWidget(self.stats_info_label) 

        # Histogram plot
        PLOT_HEIGHT = 100
        plots_layout = QHBoxLayout()
        # constrained_layout recomputes margins on every draw/resize so the
        # title and axis labels never get clipped when the canvas shrinks.
        self.stats_hist_figure = Figure(
            figsize=(4, 3), constrained_layout=True
        )
        self.stats_hist_canvas = FigureCanvas(self.stats_hist_figure)
        self.stats_hist_canvas.setMinimumHeight(PLOT_HEIGHT)
        self.stats_hist_ax = self.stats_hist_figure.add_subplot(111)
        plots_layout.addWidget(self.stats_hist_canvas) 

        # Spatial distribution plot
        self.stats_map_figure = Figure(
            figsize=(4, 3), constrained_layout=True
        )
        self.stats_map_canvas = FigureCanvas(self.stats_map_figure)
        self.stats_map_canvas.setMinimumHeight(PLOT_HEIGHT)
        self.stats_map_ax = self.stats_map_figure.add_subplot(111)
        plots_layout.addWidget(self.stats_map_canvas)
        stats_layout.addLayout(plots_layout)
        self._clear_stats_plots()

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # 2. Variogram
        variogram_group = QGroupBox(QCoreApplication.translate("Tab 2", "Variogram Parameters"))
        variogram_layout = QVBoxLayout()

        # Auto-fit success/failure status shown above the parameters table.
        self.var_autofit_status_label = QLabel("")
        self.var_autofit_status_label.setWordWrap(True)
        self.var_autofit_status_label.setVisible(False)
        variogram_layout.addWidget(self.var_autofit_status_label)

        self.var_params_table = QTableWidget()
        self.var_params_table.setColumnCount(6)
        self.var_params_table.setHorizontalHeaderLabels([
            QCoreApplication.translate("Tab 2", "Parameter"),
            QCoreApplication.translate("Tab 2", "Transform"),
            QCoreApplication.translate("Tab 2", "Model"),
            QCoreApplication.translate("Tab 2", "Nugget"),
            QCoreApplication.translate("Tab 2", "Sill"),
            QCoreApplication.translate("Tab 2", "Range"),
        ])
        range_header = self.var_params_table.horizontalHeaderItem(VAR_COL_RANGE)
        if range_header is not None:
            range_header.setToolTip(
                QCoreApplication.translate(
                    "Tab 2",
                    "Range shown is GSTools length scale (len_scale). It is not necessarily the"
                    "practical range. \nFor Spherical len_scale = practical range\n"
                    "For Exponential ≈ 3×len_scale\n"
                    "Gaussian ≈ √3×len_scale\n"
                    "For Matérn/Stable it depends on the shape parameter.",
                    # TODO: Show separately the practical range and the length scale
                )
            )
        #self.var_params_table.setFixedHeight(126)
        # Connect table changes to update the variogram
        self.var_params_table.itemChanged.connect(self.on_params_table_changed)
        self.var_params_table.resizeColumnsToContents()
        self.var_params_table.horizontalHeader().setStretchLastSection(True)
        self.var_params_table.setFixedHeight(70) 
        variogram_layout.addWidget(self.var_params_table)

        # 3: Variogram
        self.variogram_widget = VariogramWidget(self, self)
        self.variogram_widget.params_changed.connect(
            self._on_variogram_widget_params_changed
        )
        variogram_layout.addWidget(self.variogram_widget)

        # Experimental lag controls: hint (Avg D, max_dist) + lag size combo.
        exp_controls_row = QHBoxLayout()
        self.var_exp_hint_label = QLabel("")
        self.var_exp_hint_label.setStyleSheet("color: gray; font-style: italic;")
        self.var_exp_hint_label.setWordWrap(True)
        exp_controls_row.addWidget(self.var_exp_hint_label, stretch=1)

        lag_label = QLabel(
            QCoreApplication.translate("Tab 2", "Lag size:")
        )
        exp_controls_row.addWidget(lag_label)
        self.var_lag_size_combo = WheelIgnoringComboBox()
        self.var_lag_size_combo.setToolTip(
            QCoreApplication.translate(
                "Tab 2",
                "Lag spacing for the experimental variogram, based on the "
                "observed mean nearest-neighbor distance (Avg D).",
            )
        )
        self.var_lag_size_combo.currentIndexChanged.connect(
            self._on_variogram_lag_size_changed
        )
        exp_controls_row.addWidget(self.var_lag_size_combo)
        variogram_layout.addLayout(exp_controls_row)

        variogram_group.setLayout(variogram_layout)
        layout.addWidget(variogram_group)

        variogram_layout.addStretch()  # To keep layout spacing #Do we need this?

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
        apply_cv_summary_header_tooltips(self.cv_summary_table, "Tab 2")
        self.cv_summary_table.verticalHeader().setVisible(False)
        self.cv_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cv_summary_table.setFixedHeight(70)
        self.cv_summary_table.horizontalHeader().setStretchLastSection(True)
        cv_layout.addWidget(self.cv_summary_table)

        # Recompute margins on every draw and resize so the complete title and
        # both axis labels remain visible at narrower dialog sizes.
        self.cv_figure = Figure(figsize=(6, 6), constrained_layout=True)
        self.cv_canvas = FigureCanvas(self.cv_figure)
        self.cv_canvas.setMinimumHeight(PLOT_HEIGHT)
        self.cv_ax = self.cv_figure.add_subplot(111)
        cv_layout.addWidget(self.cv_canvas)

        cv_results_label = QLabel(
            QCoreApplication.translate("Tab 2", "Cross-Validation Details")
        )
        cv_layout.addWidget(cv_results_label)

        self.cv_results_table = QTableWidget()
        self.cv_results_table.setColumnCount(7)
        self.cv_results_table.setHorizontalHeaderLabels(
            cv_results_header_labels("Tab 2")
        )
        self.cv_results_table.verticalHeader().setVisible(True)
        self.cv_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cv_results_table.setSortingEnabled(True)
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

        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group)
        self._clear_cross_validation()

        # Add stretch at the end so the content doesn't expand more than necessary
        layout.addStretch()
        
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
        self.spacing_spin.setDecimals(0)
        self.spacing_spin.setSingleStep(1)
        # Default Tab 3 spacing = ANN D_o (set on layer change).
        self._apply_default_node_spacing()
        self.spacing_spin.valueChanged.connect(self.on_spacing_changed)
        gen_grid_layout.addWidget(self.spacing_spin, 0, 1)
        
        # Label to display estimated total points
        gen_grid_layout.addWidget(QLabel(QCoreApplication.translate("Tab 3", "Estimated total points:")), 1, 0)
        self.estimated_points_label = QLabel(QCoreApplication.translate("Tab 3", "Calculating..."))
        self.estimated_points_label.setStyleSheet("color: gray; font-style: italic;")
        gen_grid_layout.addWidget(self.estimated_points_label, 1, 1)
        
        # Buffer outside the concave hull, expressed as a multiple of node spacing.
        gen_grid_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 3", "Boundary buffer (x times node spacing):")
        ), 2, 0)
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0.0, 3.0)
        self.buffer_spin.setValue(1.0)
        self.buffer_spin.setDecimals(1)
        self.buffer_spin.setSingleStep(0.5)
        self.buffer_spin.setToolTip(
            QCoreApplication.translate(
                "Tab 3",
                "Expands the concave hull outward by this many times the node "
                "spacing. Default 1.0 adds one spacing interval beyond the hull.",
            )
        )
        self.buffer_spin.valueChanged.connect(self.update_estimated_points)
        gen_grid_layout.addWidget(self.buffer_spin, 2, 1)

        # Concave hull ALPHA (QGIS qgis:concavehull); 0 = concave , 1 = convex.
        gen_grid_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 3", "Boundary tightness (tight ↔ smooth)")
        ), 3, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 1.0)
        self.alpha_spin.setValue(0.7)
        self.alpha_spin.setDecimals(1)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setToolTip(
            QCoreApplication.translate(
                "Tab 3",
                "Alpha parameter for the QGIS hull around the wells. 0 = concave , 1 = convex .",
            )
        )
        gen_grid_layout.addWidget(self.alpha_spin, 3, 1)
        
        # Calculate / save grid buttons side by side
        self.preview_grid_btn = QPushButton(QCoreApplication.translate("Tab 3", "Calculate grid"))
        self.preview_grid_btn.clicked.connect(self.preview_grid)
        self.save_grid_btn = QPushButton(QCoreApplication.translate("Tab 3", "Save Grid as Temporary Layer"))
        self.save_grid_btn.clicked.connect(self.save_grid_as_layer,1,0)
        grid_btns = QHBoxLayout()
        grid_btns.addWidget(self.preview_grid_btn)
        grid_btns.addWidget(self.save_grid_btn)
        gen_grid_layout.addLayout(grid_btns, 4, 1, 1,1)

        gen_grid_group.setLayout(gen_grid_layout)
        layout.addWidget(gen_grid_group)

        # Option to upload the grid from a file
        load_grid_group = QGroupBox(QCoreApplication.translate("Tab 3", "Or Upload the estimation grid from *.XLSX file [Optional]"))
        load_grid_layout = QGridLayout()

        # Button to upload grid from file
        load_grid_layout.addWidget(QLabel(
            QCoreApplication.translate(
                "Tab 3",
                "Required columns: ID, X, Y. Optional column: weight.",
            )
        ))
        self.load_grid_btn = QPushButton(QCoreApplication.translate("Tab 3", "Select *.XLSX file"))
        self.load_grid_btn.clicked.connect(self.load_grid_as_layer)
        load_grid_layout.addWidget(self.load_grid_btn, 2, 1)

        load_grid_group.setLayout(load_grid_layout)
        layout.addWidget(load_grid_group)
        
        # Canvas for visualization
        self.grid_figure = Figure(figsize=(8, 6))
        self.grid_canvas = FigureCanvas(self.grid_figure)
        layout.addWidget(self.grid_canvas)
        
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
                "Tab 4", "Use personalized well weight"
            )
        )
        self.mn_use_well_weights.setToolTip(
            QCoreApplication.translate(
                "Tab 4",
                "Detected on the input point layer when a field name matches "
                "(case-insensitive): peso_pozo, well_weight, w_pozo, peso_w, "
                "well_w, weight_well, pozo_peso, or w. "
                "The analysis attribute and ID-like fields are ignored.",
            )
        )
        self.mn_use_well_weights.setEnabled(False)
        self.mn_use_well_weights.stateChanged.connect(
            self._on_mn_well_weight_toggled
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
                "Tab 4", "Use estimation grid node weights"
            )
        )
        self.mn_use_grid_weights.setToolTip(
            QCoreApplication.translate(
                "Tab 4",
                "Available after importing an estimation grid Excel on tab 3. "
                "Required columns in order: ID, X, Y. Optional 4th column: "
                "weight (used as per-node grid weight). "
                "Generated grids without an imported weight column cannot use "
                "this option.",
            )
        )
        self.mn_use_grid_weights.setEnabled(False)
        self.mn_use_grid_weights.stateChanged.connect(
            self._on_mn_grid_weight_toggled
        )
        mn_param_select_layout.addWidget(self.mn_use_grid_weights)

        self.mn_grid_weight_info_label = QLabel()
        self.mn_grid_weight_info_label.setStyleSheet(
            "color: gray; font-style: italic;"
        )
        self.mn_grid_weight_info_label.setWordWrap(True)
        mn_param_select_layout.addWidget(self.mn_grid_weight_info_label)

        # Add Buttons to optimize and download prioritization
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
        mn_param_select_layout.addLayout(optimize_btn_row)

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

        MN_PLOT_HEIGHT = 250
        self.variance_figure = Figure(figsize=(6, 3))
        self.variance_canvas = FigureCanvas(self.variance_figure)
        self.variance_canvas.setMinimumHeight(MN_PLOT_HEIGHT)
        self.variance_ax = self.variance_figure.add_subplot(111)
        variance_layout.addWidget(self.variance_canvas)

        order_label = QLabel(
            QCoreApplication.translate("Tab 4", "Optimization order")
        )
        variance_layout.addWidget(order_label)

        self.mn_prioritization_order_table = QTableWidget()
        self.mn_prioritization_order_table.setColumnCount(6)
        self.mn_prioritization_order_table.setHorizontalHeaderLabels([
            QCoreApplication.translate("Tab 4", "Priority"),
            QCoreApplication.translate("Tab 4", "ID"),
            QCoreApplication.translate("Tab 4", "Well weight"),
            QCoreApplication.translate("Tab 4", "Total variance (%)"),
            QCoreApplication.translate("Tab 4", "Adverse order"),
            QCoreApplication.translate("Tab 4", "Adverse variance (%)"),
        ])
        self.mn_prioritization_order_table.verticalHeader().setVisible(True)
        self.mn_prioritization_order_table.setSortingEnabled(True)
        self.mn_prioritization_order_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.mn_prioritization_order_table.setMinimumHeight(320)
        #self.mn_prioritization_order_table.setMaximumHeight(380)
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


    def setup_map_tab(self):
        """Configures the results tab (OK maps and CV for all vs selected wells)."""
        # Scroll area wraps the whole tab so buttons, maps, and tables stay reachable.
        results_tab_layout = QVBoxLayout()
        results_tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_widget = QWidget()
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(10, 10, 10, 10)

        interp_group = QGroupBox(
            QCoreApplication.translate("Tab 5", "Ordinary Kriging (OK) Interpolation")
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
        # Refresh on every change of the spinbox
        self.results_well_spin.valueChanged.connect(
            self._refresh_ok_interpolation_selected_wells
            #TODO: add a delay after the user stops typing to allow 2 digits to be typed
        )
        wells_row.addWidget(self.results_well_spin)
        self.save_selected_wells_btn = QPushButton(
            QCoreApplication.translate("Tab 5", "Download selected wells as layer")
        )
        self.save_selected_wells_btn.clicked.connect(
            self.save_selected_wells_as_layer
        )
        wells_row.addWidget(self.save_selected_wells_btn)
        self.save_interpolation_btn = QPushButton(
            QCoreApplication.translate("Tab 5", "Download interpolation as layer")
        )
        self.save_interpolation_btn.setToolTip(
            QCoreApplication.translate(
                "Tab 5",
                "Download the krigging interpolation map as a temporary layer. The raster resolution is based on the estimation grid spacing.",
            )
        )
        self.save_interpolation_btn.clicked.connect(
            self.save_ok_interpolation_as_layer
        )
        wells_row.addWidget(self.save_interpolation_btn)
        wells_row.addStretch()
        interp_layout.addLayout(wells_row)

        # Visibility-only controls: toggle existing widgets/artists, never re-krige.
        self.results_show_se_maps_cb = QCheckBox(
            QCoreApplication.translate(
                "Tab 5", "Show kriging standard error maps"
            )
        )
        self.results_show_se_maps_cb.setChecked(False)
        self.results_show_se_maps_cb.toggled.connect(
            self._on_results_show_se_maps_toggled
        )
        interp_layout.addWidget(self.results_show_se_maps_cb)

        self.results_show_wells_cb = QCheckBox(
            QCoreApplication.translate(
                "Tab 5", "Show monitoring wells on maps"
            )
        )
        self.results_show_wells_cb.setChecked(True)
        self.results_show_wells_cb.toggled.connect(
            self._on_results_show_wells_toggled
        )
        interp_layout.addWidget(self.results_show_wells_cb)

        plots_row = QHBoxLayout()
        # Extra height leaves room for titles, colorbars, and legends.
        RESULTS_PLOT_HEIGHT = 320

        all_column = QVBoxLayout()
        all_column.addWidget(QLabel(
            QCoreApplication.translate("Tab 5", "All wells")
        ))
        self.results_all_interp_figure = Figure(figsize=(5, 4.2))
        self.results_all_interp_canvas = FigureCanvas(self.results_all_interp_figure)
        self.results_all_interp_canvas.setMinimumHeight(RESULTS_PLOT_HEIGHT)
        self.results_all_interp_ax = self.results_all_interp_figure.add_subplot(111)
        all_column.addWidget(self.results_all_interp_canvas)

        # SE label + canvas in one container so the checkbox can hide both together.
        self.results_all_se_container = QWidget()
        all_se_layout = QVBoxLayout()
        all_se_layout.setContentsMargins(0, 0, 0, 0)
        all_se_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 5", "Kriging standard error")
        ))
        self.results_all_se_figure = Figure(figsize=(5, 4.2))
        self.results_all_se_canvas = FigureCanvas(self.results_all_se_figure)
        self.results_all_se_canvas.setMinimumHeight(RESULTS_PLOT_HEIGHT)
        self.results_all_se_ax = self.results_all_se_figure.add_subplot(111)
        all_se_layout.addWidget(self.results_all_se_canvas)
        self.results_all_se_container.setLayout(all_se_layout)
        all_column.addWidget(self.results_all_se_container)

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
        self.results_sel_interp_figure = Figure(figsize=(5, 4.2))
        self.results_sel_interp_canvas = FigureCanvas(self.results_sel_interp_figure)
        self.results_sel_interp_canvas.setMinimumHeight(RESULTS_PLOT_HEIGHT)
        self.results_sel_interp_ax = self.results_sel_interp_figure.add_subplot(111)
        sel_column.addWidget(self.results_sel_interp_canvas)

        self.results_sel_se_container = QWidget()
        sel_se_layout = QVBoxLayout()
        sel_se_layout.setContentsMargins(0, 0, 0, 0)
        sel_se_layout.addWidget(QLabel(
            QCoreApplication.translate("Tab 5", "Kriging standard error")
        ))
        self.results_sel_se_figure = Figure(figsize=(5, 4.2))
        self.results_sel_se_canvas = FigureCanvas(self.results_sel_se_figure)
        self.results_sel_se_canvas.setMinimumHeight(RESULTS_PLOT_HEIGHT)
        self.results_sel_se_ax = self.results_sel_se_figure.add_subplot(111)
        sel_se_layout.addWidget(self.results_sel_se_canvas)
        self.results_sel_se_container.setLayout(sel_se_layout)
        sel_column.addWidget(self.results_sel_se_container)

        (
            sel_cv_widget,
            self.results_sel_cv_summary_table,
            self.results_sel_cv_results_table,
            self.results_sel_cv_info_label,
        ) = self._build_cv_tables_widget("Tab 5")
        sel_column.addWidget(sel_cv_widget)
        plots_row.addLayout(sel_column, stretch=1)

        interp_layout.addLayout(plots_row)
        # Checkbox starts unchecked and does not emit toggled; hide SE maps now.
        self._set_ok_se_maps_visible(False)
        interp_group.setLayout(interp_layout)
        results_layout.addWidget(interp_group)

        scroll_widget.setLayout(results_layout)
        scroll_area.setWidget(scroll_widget)
        results_tab_layout.addWidget(scroll_area)
        self.tab_results.setLayout(results_tab_layout)
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
        apply_cv_summary_header_tooltips(summary_table, context_name)
        summary_table.verticalHeader().setVisible(False)
        summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        summary_table.setFixedHeight(70)
        summary_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(summary_table)

        results_label = QLabel(
            QCoreApplication.translate(context_name, "Cross-Validation Details")
        )
        layout.addWidget(results_label)

        results_table = QTableWidget()
        results_table.setColumnCount(7)
        results_table.setHorizontalHeaderLabels(
            cv_results_header_labels(context_name)
        )
        results_table.verticalHeader().setVisible(True)
        results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        results_table.setSortingEnabled(True)
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
            layer, attr_name, include_fids=self._included_well_fids_for_extract()
        )
        if coordinates is None or values is None:
            return 0
        return int(values.size)

    def _update_results_well_spinbox(self):
        """Sets Tab 5 well-count spin box range and default value.

        When Tab 4 optimize results exist for the current parameter, the default
        is the smallest well count that reaches ≥90% of the maximum variance
        reduction (same threshold as the Tab 4 reference line). Otherwise the
        value is only clamped to the valid [3, n_wells] range.
        """
        if not hasattr(self, 'results_well_spin'):
            return

        attr_name = self._current_mn_parameter()
        count_attr = attr_name
        if self._is_combined_mn_parameter(attr_name):
            selected = self._selected_analysis_parameters()
            count_attr = selected[0] if selected else None
        n_wells = self._count_layer_wells(count_attr) if count_attr else 0

        # Prefer the 95%-of-max-reduction well count from prioritized curve.
        default_wells = None
        if attr_name and attr_name in getattr(self, 'variance_results', {}):
            default_wells = wells_at_fraction_of_max_reduction(
                self.variance_results[attr_name],
                fraction=0.95,
            )

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
            if default_wells is not None:
                value = max(3, min(int(default_wells), n_wells))
                self.results_well_spin.setValue(value)
            else:
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

    @staticmethod
    def _cv_results_numeric_item(value, read_only_flags):
        """
        Read-only cell with numeric Qt.DisplayRole for native header sorting.

        Non-finite values use an empty string so nulls do not sort as zero.
        """
        item = QTableWidgetItem()
        item.setFlags(read_only_flags)
        try:
            number = float(value)
        except (TypeError, ValueError):
            item.setData(Qt.DisplayRole, "")
            return item
        if not np.isfinite(number):
            item.setData(Qt.DisplayRole, "")
            return item
        item.setData(Qt.DisplayRole, float(number))
        return item

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

        # Avoid re-sorting on every setItem while rows are filled.
        results_table.setSortingEnabled(False)
        results_table.setRowCount(len(cv_rows))

        numeric_keys = (
            'measured',
            'predicted',
            'error',
            'se',
            'standardized_error',
        )

        for row, (point_id, row_data) in enumerate(zip(point_ids, cv_rows)):
            included = row_data.get('included', True)

            # ID: numeric DisplayRole when the identifier is a finite number.
            id_item = QTableWidgetItem()
            id_item.setFlags(read_only)
            try:
                id_number = float(point_id)
            except (TypeError, ValueError):
                id_number = None
            if id_number is not None and np.isfinite(id_number):
                if float(id_number).is_integer():
                    id_item.setData(Qt.DisplayRole, int(id_number))
                else:
                    id_item.setData(Qt.DisplayRole, float(id_number))
            else:
                id_item.setData(Qt.DisplayRole, str(point_id))
            if highlight_selected and included:
                id_item.setBackground(selected_row_color)
            results_table.setItem(row, CV_COL_ID, id_item)

            included_item = QTableWidgetItem(
                yes_text if included else no_text
            )
            included_item.setFlags(read_only)
            if highlight_selected and included:
                included_item.setBackground(selected_row_color)
            results_table.setItem(row, CV_COL_INCLUDED, included_item)

            # Measured / Predicted / Error / SE / Standardized Error.
            for col_offset, key in enumerate(numeric_keys):
                col = CV_COL_MEASURED + col_offset
                item = self._cv_results_numeric_item(
                    row_data.get(key), read_only
                )
                if highlight_selected and included:
                    item.setBackground(selected_row_color)
                results_table.setItem(row, col, item)

        results_table.resizeColumnsToContents()
        results_table.setSortingEnabled(True)

    def _on_results_show_se_maps_toggled(self, checked):
        """Show or hide SE map containers without recomputing kriging."""
        self._set_ok_se_maps_visible(bool(checked))

    def _set_ok_se_maps_visible(self, visible):
        """Toggle visibility of both Tab 5 standard-error map containers."""
        for attr_name in ('results_all_se_container', 'results_sel_se_container'):
            container = getattr(self, attr_name, None)
            if container is not None:
                container.setVisible(visible)

    def _on_results_show_wells_toggled(self, checked):
        """Show or hide well markers on Tab 5 maps without recomputing kriging."""
        self._set_ok_map_wells_visible(bool(checked))

    def _register_ok_map_well_scatter(self, canvas, scatter):
        """Remember a well-marker PathCollection for visibility toggles.

        Keyed by canvas so a redraw replaces the previous artist for that map
        without leaving stale references.
        """
        if scatter is None or canvas is None:
            return
        scatters = getattr(self, '_ok_map_well_scatters', None)
        if not isinstance(scatters, dict):
            self._ok_map_well_scatters = {}
            scatters = self._ok_map_well_scatters
        scatters[id(canvas)] = scatter

    def _set_ok_map_wells_visible(self, visible):
        """Toggle well-marker artists on all Tab 5 maps; redraw canvases only."""
        visible = bool(visible)
        scatters = getattr(self, '_ok_map_well_scatters', None) or {}
        if isinstance(scatters, dict):
            artists = scatters.values()
        else:
            artists = scatters
        for scatter in list(artists):
            try:
                scatter.set_visible(visible)
            except Exception:
                pass

        for canvas_attr in (
            'results_all_interp_canvas',
            'results_sel_interp_canvas',
            'results_all_se_canvas',
            'results_sel_se_canvas',
        ):
            canvas = getattr(self, canvas_attr, None)
            if canvas is not None:
                canvas.draw_idle()

    def _apply_ok_map_wells_visibility(self):
        """Re-apply the wells checkbox after a map redraw (no recalculation)."""
        checkbox = getattr(self, 'results_show_wells_cb', None)
        visible = True if checkbox is None else checkbox.isChecked()
        # Apply visibility without a second full canvas refresh loop when possible.
        visible = bool(visible)
        scatters = getattr(self, '_ok_map_well_scatters', None) or {}
        artists = scatters.values() if isinstance(scatters, dict) else scatters
        for scatter in list(artists):
            try:
                scatter.set_visible(visible)
            except Exception:
                pass

    def _finalize_ok_map_figure_layout(self, figure, canvas):
        """Keep titles and axes labels inside the canvas with compact margins.

        When a horizontal colorbar is already attached (extra axes), leave
        margins alone so the colorbar does not overlap the map.
        """
        try:
            if len(figure.axes) <= 1:
                figure.subplots_adjust(
                    left=0.14, right=0.96, top=0.90, bottom=0.14
                )
        except Exception:
            pass
        canvas.draw()

    @staticmethod
    def _ok_map_locator_levels(vmin, vmax, nbins=6):
        """Pretty breaks via MaxNLocator for Tab 5 colorbars and isolines."""
        from matplotlib.ticker import MaxNLocator

        if vmin is None or vmax is None:
            return None
        try:
            vmin = float(vmin)
            vmax = float(vmax)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if vmin == vmax:
            return np.asarray([vmin], dtype=float)
        levels = np.asarray(
            MaxNLocator(nbins=nbins).tick_values(vmin, vmax),
            dtype=float,
        )
        levels = levels[np.isfinite(levels)]
        return levels if levels.size else None

    @staticmethod
    def _format_ok_map_axis_tick(value):
        """Compact tick label for large projected coordinates."""
        value = float(value)
        abs_v = abs(value)
        if abs_v >= 1e5 or (abs_v > 0 and abs_v < 1e-2):
            return f'{value:.2e}'
        if abs_v >= 1000:
            return f'{value:.0f}'
        return f'{value:g}'

    def _style_ok_map_axes(self, ax):
        """3 divisions per axis, 2 labels, no grid, vertical Y tick labels."""
        from matplotlib.ticker import MaxNLocator

        ax.grid(False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xticks = np.asarray(
            MaxNLocator(nbins=3).tick_values(xmin, xmax), dtype=float
        )
        yticks = np.asarray(
            MaxNLocator(nbins=3).tick_values(ymin, ymax), dtype=float
        )
        xticks = xticks[(xticks >= xmin) & (xticks <= xmax)]
        yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
        if xticks.size > 3:
            xticks = xticks[:3]
        if yticks.size > 3:
            yticks = yticks[:3]

        def _two_end_labels(ticks):
            labels = [''] * int(ticks.size)
            if ticks.size == 0:
                return labels
            if ticks.size == 1:
                labels[0] = self._format_ok_map_axis_tick(ticks[0])
                return labels
            labels[0] = self._format_ok_map_axis_tick(ticks[0])
            labels[-1] = self._format_ok_map_axis_tick(ticks[-1])
            return labels

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(_two_end_labels(xticks))
        ax.set_yticklabels(
            _two_end_labels(yticks), rotation=90, va='center', ha='right'
        )
        # Keep axis titles inside the axes band (ylabel above xlabel strip).
        ax.xaxis.set_label_coords(0.5, -0.10)
        ax.yaxis.set_label_coords(-0.10, 0.5)

    def _add_ok_map_colorbar(self, figure, mappable, ax, label, levels=None):
        """Horizontal colorbar below the axes; ticks match isoline levels."""
        # Reserve bottom space before attaching the colorbar axes.
        figure.subplots_adjust(
            left=0.14, right=0.96, top=0.90, bottom=0.20
        )
        colorbar = figure.colorbar(
            mappable,
            ax=ax,
            orientation='horizontal',
            fraction=0.055,
            pad=0.14,
            aspect=35,
            shrink=0.90,
        )
        colorbar.set_label(label)
        if levels is not None and len(levels) > 0:
            colorbar.set_ticks(np.asarray(levels, dtype=float))
        return colorbar

    def _clear_ok_interpolation_plot_side(
        self, figure, canvas, colorbar_attr, message=..., default_placeholder=None
    ):
        """Resets one tab 5 interpolation or standard-error figure."""
        if figure is None or canvas is None:
            return

        self._disconnect_point_hover(canvas)
        figure.clear()
        # Placeholder/empty figures have no well markers for this canvas.
        scatters = getattr(self, '_ok_map_well_scatters', None)
        if isinstance(scatters, dict):
            scatters.pop(id(canvas), None)
        ax = figure.add_subplot(111)
        colorbar = getattr(self, colorbar_attr, None)
        if colorbar is not None:
            setattr(self, colorbar_attr, None)

        if message is None:
            self._finalize_ok_map_figure_layout(figure, canvas)
            return ax

        if message is ...:
            placeholder = default_placeholder or QCoreApplication.translate(
                "Tab 5", "No interpolation map available"
            )
        else:
            placeholder = message
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
        self._finalize_ok_map_figure_layout(figure, canvas)
        return ax

    def _clear_ok_interpolation_plots(self, message=...):
        """Resets both tab 5 interpolation maps, SE maps, and CV tables."""
        self.results_interp_color_limits = None
        self.results_se_color_limits = None
        # Drop stale scatter references; placeholders have no well markers.
        self._ok_map_well_scatters = {}
        default_cv_message = (
            QCoreApplication.translate(
                "Tab 5",
                "Run Kalman optimization on tab 4 to compute cross-validation.",
            )
            if message is ...
            else message
        )
        se_default = QCoreApplication.translate(
            "Tab 5", "No standard error map available"
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
        self._clear_ok_interpolation_plot_side(
            getattr(self, 'results_all_se_figure', None),
            getattr(self, 'results_all_se_canvas', None),
            'results_all_se_colorbar',
            message,
            default_placeholder=se_default,
        )
        self._clear_ok_interpolation_plot_side(
            getattr(self, 'results_sel_se_figure', None),
            getattr(self, 'results_sel_se_canvas', None),
            'results_sel_se_colorbar',
            message,
            default_placeholder=se_default,
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
        self.results_all_se_ax = (
            self.results_all_se_figure.axes[0]
            if hasattr(self, 'results_all_se_figure')
            and self.results_all_se_figure.axes
            else None
        )
        self.results_sel_se_ax = (
            self.results_sel_se_figure.axes[0]
            if hasattr(self, 'results_sel_se_figure')
            and self.results_sel_se_figure.axes
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
        # Keep well IDs aligned with the filtered scatter points.
        if point_ids is not None and len(point_ids) == measured.size:
            plot_ids = [point_ids[i] for i in range(measured.size) if valid[i]]
        else:
            plot_ids = None
        self._update_cross_validation_plot(
            measured[valid], predicted[valid], attr_name, point_ids=plot_ids
        )

        self.cv_info_label.setText(
            QCoreApplication.translate(
                "Tab 2",
                "Leave-one-out cross-validation for «{param}» ({n} points).",
            ).format(param=attr_name, n=int(np.sum(valid)))
        )
        self.cv_info_label.setStyleSheet("color: gray; font-style: italic;")

    def _clear_all_optimization_results(self, tab5_message=None, tab4_message=None):
        """
        Drop every cached Optimize result and reset Tab 4/5 views.

        Used when inputs that affect all rankings change (layer, Tab 1
        selection, weight toggles, grid rebuild).
        """
        if hasattr(self, 'variance_results'):
            self.variance_results.clear()
        if hasattr(self, 'prioritization_grid_points'):
            self.prioritization_grid_points = None
        self._clear_variance_reduction_plot()
        if tab4_message and hasattr(self, 'variance_info_label'):
            self.variance_info_label.setText(tab4_message)
            self.variance_info_label.setStyleSheet(
                "color: gray; font-style: italic;"
            )
        self._update_results_well_spinbox()
        self._clear_ok_interpolation_plots(
            tab5_message
            if tab5_message is not None
            else QCoreApplication.translate(
                "Tab 5",
                "Optimization cleared. Run Optimize on tab 4 again.",
            )
        )
        self._update_workflow_navigation_state()

    def _load_optimization_views_for_current_parameter(self):
        """
        Show cached Optimize results for the Tab 4 combo selection, or clear
        the views so another parameter's ranking is never left on screen.
        """
        param_key = self._current_mn_parameter()
        if param_key and param_key in getattr(self, 'variance_results', {}):
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
        else:
            self._clear_variance_reduction_plot()
            if hasattr(self, 'variance_info_label'):
                self.variance_info_label.setText(
                    QCoreApplication.translate(
                        "Tab 4",
                        "No optimization for this parameter. Click Optimize.",
                    )
                )
                self.variance_info_label.setStyleSheet(
                    "color: gray; font-style: italic;"
                )
            self._update_results_well_spinbox()
            self._clear_ok_interpolation_plots(
                QCoreApplication.translate(
                    "Tab 5",
                    "Run Optimize on tab 4 for the selected parameter.",
                )
            )
        self._update_workflow_navigation_state()

    def _invalidate_optimization_after_variogram_change(self, attr_name=None):
        """
        Clear Tab 4 optimization results when a variogram changes.

        Well ranking and variance reduction depend on the covariance model
        (nugget, sill, range, type) and on the transformed values. After a
        manual parameter edit or a transform change those results are stale
        and must not remain visible on tabs 4–5.

        Args:
            attr_name: Parameter whose variogram changed. Also drops the
                combined-parameter optimization. ``None`` clears all results
                (e.g. after re-fitting every variogram from Tab 1).
        """
        if not hasattr(self, 'variance_results'):
            return

        if attr_name is None:
            self.variance_results.clear()
        else:
            self.variance_results.pop(attr_name, None)
            # Combined mode mixes every selected parameter's variogram.
            self.variance_results.pop(MN_COMBINED_PARAMETERS_KEY, None)

        current = self._current_mn_parameter()
        if current is None or current not in self.variance_results:
            self._clear_variance_reduction_plot()
            self._update_results_well_spinbox()
            self._clear_ok_interpolation_plots(
                QCoreApplication.translate(
                    "Tab 5",
                    "Variogram changed. Run Optimize on tab 4 again.",
                )
            )
        self._update_workflow_navigation_state()

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

        # Univariate OK uses the first listed parameter (not a multivariate model).
        ok_parameter = (
            param_data.get('reference_parameter')
            or self._resolve_cv_parameter_name(attr_name)
            or attr_name
        )

        return {
            'attr_name': attr_name,
            'ok_parameter': ok_parameter,
            'param_data': param_data,
            'state': param_data['state'],
            'coordinates': param_data['coordinates'],
            'values': param_data['values'],
            'point_ids': param_data.get('point_ids'),
            'grid': np.asarray(self.current_grid_points, dtype=float),
            'selection_order': self.variance_results[attr_name]['selection_order'],
        }, None

    def _compute_ok_estimates(self, context, well_indices, return_std_error=False):
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
            return_std_error=return_std_error,
        )

    def _compute_ok_estimates_and_se(self, context, well_indices):
        """Returns (estimates, std_error) for the given well subset on the grid."""
        return self._compute_ok_estimates(
            context, well_indices, return_std_error=True
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

    def _ensure_ok_se_color_limits(self, context):
        """
        Shared color ramp for Tab 5 kriging standard-error maps.

        Uses the all-wells SE field so both panels stay comparable.
        """
        if self.results_se_color_limits is not None:
            return self.results_se_color_limits

        n_wells = context['coordinates'].shape[0]
        try:
            _, all_se = self._compute_ok_estimates_and_se(
                context, np.arange(n_wells)
            )
        except Exception:
            return None
        if all_se is None:
            return None

        self.results_se_color_limits = self._ok_interpolation_color_limits(all_se)
        return self.results_se_color_limits

    def _grid_field_to_raster(self, grid_xy, values):
        """
        Build a regular 2-D raster from estimation-grid nodes.

        Shared by Tab 5 O.K. / S.E. plots and the GeoTIFF export path.
        Tab 3 grids are rectangular lattices clipped to a hull, so unique X/Y
        reconstruct the mesh; missing cells stay NaN.

        Returns:
            (xs, ys, Z) with Z shape (len(ys), len(xs)), row 0 = min Y
            (imshow origin='lower'), or None if reshape fails.
        """
        grid_xy = np.asarray(grid_xy, dtype=float)
        values = np.asarray(values, dtype=float).ravel()
        if (
            grid_xy.ndim != 2
            or grid_xy.shape[1] != 2
            or values.size != grid_xy.shape[0]
            or values.size == 0
        ):
            return None

        # Round for stable unique axes when floating-point coords jitter slightly.
        xs = np.unique(np.round(grid_xy[:, 0], decimals=8))
        ys = np.unique(np.round(grid_xy[:, 1], decimals=8))
        if xs.size < 2 or ys.size < 2:
            return None

        x_idx = {float(x): i for i, x in enumerate(xs)}
        y_idx = {float(y): i for i, y in enumerate(ys)}
        z = np.full((ys.size, xs.size), np.nan, dtype=float)
        for (x, y), value in zip(grid_xy, values):
            if not np.isfinite(value):
                continue
            xi = x_idx.get(float(np.round(x, decimals=8)))
            yi = y_idx.get(float(np.round(y, decimals=8)))
            if xi is None or yi is None:
                continue
            z[yi, xi] = float(value)
        return xs, ys, z

    def _raster_cell_size(self, xs, ys):
        """Pixel width/height: prefer Tab 3 spacing, else median axis diffs."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        dx = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
        dy = float(np.median(np.diff(ys))) if ys.size > 1 else 1.0
        grid_info = getattr(self, 'current_grid_info', None) or {}
        spacing = grid_info.get('spacing')
        if spacing is not None:
            try:
                spacing = float(spacing)
                if spacing > 0:
                    dx = spacing
                    dy = spacing
            except (TypeError, ValueError):
                pass
        if dx <= 0:
            dx = 1.0
        if dy <= 0:
            dy = 1.0
        return dx, dy

    def _raster_imshow_extent(self, xs, ys):
        """Cell-edge extent for imshow given sorted cell-center coordinates."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        dx, dy = self._raster_cell_size(xs, ys)
        return [
            float(xs[0] - 0.5 * dx),
            float(xs[-1] + 0.5 * dx),
            float(ys[0] - 0.5 * dy),
            float(ys[-1] + 0.5 * dy),
        ]

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
        """Draw OK interpolation as nearest neighbor imshow with contour isolines.

        Color ramp and isolines share MaxNLocator(nbins=6) levels from the
        all-wells min–max when ``color_limits`` is provided. Well IDs on well
        markers are shown on hover only (raw IDs, not translated).
        """
        coordinates = context['coordinates']
        # Colorbar shows the interpolated parameter (first of combined list).
        ok_parameter = context.get('ok_parameter') or context['attr_name']
        selected_indices = np.asarray(selected_indices, dtype=int)
        n_selected = selected_indices.size

        gx = context['grid'][:, 0]
        gy = context['grid'][:, 1]
        self._disconnect_point_hover(canvas)
        ax.clear()

        if color_limits is None:
            color_limits = self._ok_interpolation_color_limits(estimates)
        vmin = vmax = None
        if color_limits is not None:
            vmin, vmax = color_limits
        levels = self._ok_map_locator_levels(vmin, vmax, nbins=6)

        raster = self._grid_field_to_raster(context['grid'], estimates)
        mappable = None
        try:
            if raster is None:
                raise ValueError("cannot reshape OK field to a regular raster")
            xs, ys, z = raster
            z_plot = np.ma.masked_invalid(z)
            mappable = ax.imshow(
                z_plot,
                origin='lower',
                extent=self._raster_imshow_extent(xs, ys),
                aspect='equal',
                interpolation='nearest',
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                zorder=1,
            )
            if levels is not None and levels.size > 0:
                contours = ax.contour(
                    xs,
                    ys,
                    z_plot,
                    levels=levels,
                    colors='black',
                    linewidths=0.4,
                    zorder=2,
                )
                ax.clabel(
                    contours,
                    levels=levels,
                    inline=True,
                    fontsize=7,
                    colors='black',
                    fmt='%g',
                )
        except Exception:
            # Scatter fallback when the node set is not a regular lattice.
            mappable = ax.scatter(
                gx,
                gy,
                c=estimates,
                cmap='viridis',
                s=12,
                alpha=0.9,
                vmin=vmin,
                vmax=vmax,
            )

        all_selected = n_selected == coordinates.shape[0]

        sel_coords = coordinates[selected_indices]
        well_label = (
            QCoreApplication.translate("Tab 5", "All wells")
            if all_selected
            else QCoreApplication.translate("Tab 5", "Selected wells")
        )
        well_scatter = ax.scatter(
            sel_coords[:, 0],
            sel_coords[:, 1],
            c='red',
            s=30,
            edgecolors='k',
            linewidths=0.5,
            label=well_label,
            zorder=5,
        )
        self._register_ok_map_well_scatter(canvas, well_scatter)

        # Hover targets: all wells in the network (raw IDs, never translated).
        point_ids = context.get('point_ids')
        if point_ids is not None and coordinates is not None:
            coords = np.asarray(coordinates, dtype=float)
            id_labels = [str(pid) for pid in point_ids]
            if len(id_labels) == coords.shape[0]:
                self._attach_point_hover(canvas, ax, coords, id_labels)

        ax.set_xlabel(QCoreApplication.translate("Tab 5", "X"))
        ax.set_ylabel(QCoreApplication.translate("Tab 5", "Y"))
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.set_aspect('equal')
        self._style_ok_map_axes(ax)
        colorbar = self._add_ok_map_colorbar(
            figure, mappable, ax, ok_parameter, levels=levels
        )
        setattr(self, colorbar_attr, colorbar)
        self._apply_ok_map_wells_visibility()
        self._finalize_ok_map_figure_layout(figure, canvas)

    def _draw_ok_std_error_map(
        self,
        figure,
        canvas,
        colorbar_attr,
        ax,
        context,
        std_error,
        selected_indices,
        title,
        color_limits=None,
    ):
        """Draw OK kriging standard-error surface as nearest-neighbor imshow.

        No isolines. Colormap is ``Reds`` so SE is visually distinct from
        estimate ``viridis``. Colorbar ticks use MaxNLocator(nbins=6).
        """
        coordinates = context['coordinates']
        selected_indices = np.asarray(selected_indices, dtype=int)
        n_selected = selected_indices.size

        gx = context['grid'][:, 0]
        gy = context['grid'][:, 1]
        self._disconnect_point_hover(canvas)
        ax.clear()

        if color_limits is None:
            color_limits = self._ok_interpolation_color_limits(std_error)
        vmin = vmax = None
        if color_limits is not None:
            vmin, vmax = color_limits
        levels = self._ok_map_locator_levels(vmin, vmax, nbins=6)

        se_label = QCoreApplication.translate("Tab 5", "Standard error")
        raster = self._grid_field_to_raster(context['grid'], std_error)
        mappable = None
        try:
            if raster is None:
                raise ValueError("cannot reshape SE field to a regular raster")
            xs, ys, z = raster
            z_plot = np.ma.masked_invalid(z)
            mappable = ax.imshow(
                z_plot,
                origin='lower',
                extent=self._raster_imshow_extent(xs, ys),
                aspect='equal',
                interpolation='nearest',
                cmap='Reds',
                vmin=vmin,
                vmax=vmax,
                zorder=1,
            )
        except Exception:
            mappable = ax.scatter(
                gx,
                gy,
                c=std_error,
                cmap='Reds',
                s=12,
                alpha=0.9,
                vmin=vmin,
                vmax=vmax,
            )

        all_selected = n_selected == coordinates.shape[0]
        sel_coords = coordinates[selected_indices]
        well_label = (
            QCoreApplication.translate("Tab 5", "All wells")
            if all_selected
            else QCoreApplication.translate("Tab 5", "Selected wells")
        )
        well_scatter = ax.scatter(
            sel_coords[:, 0],
            sel_coords[:, 1],
            c='red',
            s=30,
            edgecolors='k',
            linewidths=0.5,
            label=well_label,
            zorder=5,
        )
        self._register_ok_map_well_scatter(canvas, well_scatter)

        # Hover targets: all wells in the network (raw IDs, never translated).
        point_ids = context.get('point_ids')
        if point_ids is not None and coordinates is not None:
            coords = np.asarray(coordinates, dtype=float)
            id_labels = [str(pid) for pid in point_ids]
            if len(id_labels) == coords.shape[0]:
                self._attach_point_hover(canvas, ax, coords, id_labels)

        ax.set_xlabel(QCoreApplication.translate("Tab 5", "X"))
        ax.set_ylabel(QCoreApplication.translate("Tab 5", "Y"))
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.set_aspect('equal')
        self._style_ok_map_axes(ax)
        colorbar = self._add_ok_map_colorbar(
            figure, mappable, ax, se_label, levels=levels
        )
        setattr(self, colorbar_attr, colorbar)
        self._apply_ok_map_wells_visibility()
        self._finalize_ok_map_figure_layout(figure, canvas)

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
        ok_parameter = context.get('ok_parameter') or context['attr_name']
        if network_indices is None:
            info_label.setText(
                QCoreApplication.translate(
                    "Tab 5",
                    "Leave-one-out cross-validation for «{param}» ({n} points).",
                ).format(param=ok_parameter, n=n_total)
            )
        else:
            info_label.setText(
                QCoreApplication.translate(
                    "Tab 5",
                    "Cross-validation for «{param}» using {n_network} network "
                    "wells ({n_total} points).",
                ).format(
                    param=ok_parameter,
                    n_network=len(network_indices),
                    n_total=n_total,
                )
            )
        info_label.setStyleSheet("color: gray; font-style: italic;")

    def _refresh_ok_interpolation_all_wells(self):
        """Builds the all-wells OK map, SE map, and CV (not tied to the spinbox)."""
        if not hasattr(self, 'results_all_interp_ax'):
            return

        se_default = QCoreApplication.translate(
            "Tab 5", "No standard error map available"
        )
        context, error = self._get_ok_interpolation_context()
        if context is None:
            self._clear_ok_interpolation_plot_side(
                self.results_all_interp_figure,
                self.results_all_interp_canvas,
                'results_all_interp_colorbar',
                error,
            )
            self._clear_ok_interpolation_plot_side(
                getattr(self, 'results_all_se_figure', None),
                getattr(self, 'results_all_se_canvas', None),
                'results_all_se_colorbar',
                error,
                default_placeholder=se_default,
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
            estimates, std_error = self._compute_ok_estimates_and_se(
                context, all_indices
            )
        except Exception:
            estimates, std_error = None, None

        if (
            estimates is None
            or estimates.size != context['grid'].shape[0]
        ):
            message = QCoreApplication.translate(
                "Tab 5", "Interpolation could not be computed."
            )
            self._clear_ok_interpolation_plot_side(
                self.results_all_interp_figure,
                self.results_all_interp_canvas,
                'results_all_interp_colorbar',
                message,
            )
            self._clear_ok_interpolation_plot_side(
                getattr(self, 'results_all_se_figure', None),
                getattr(self, 'results_all_se_canvas', None),
                'results_all_se_colorbar',
                message,
                default_placeholder=se_default,
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
        ok_parameter = context.get('ok_parameter') or context['attr_name']
        title = QCoreApplication.translate(
            "Tab 5",
            "O.K. – {param}\n(all {n} wells)",
        ).format(param=ok_parameter, n=n_wells)
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

        if (
            hasattr(self, 'results_all_se_figure')
            and std_error is not None
            and std_error.size == context['grid'].shape[0]
        ):
            self._clear_ok_interpolation_plot_side(
                self.results_all_se_figure,
                self.results_all_se_canvas,
                'results_all_se_colorbar',
                message=None,
                default_placeholder=se_default,
            )
            self.results_all_se_ax = self.results_all_se_figure.axes[0]
            se_limits = self._ok_interpolation_color_limits(std_error)
            self.results_se_color_limits = se_limits
            se_title = QCoreApplication.translate(
                "Tab 5",
                "S.E. – {param} \n(all {n} wells)",
            ).format(param=ok_parameter, n=n_wells)
            self._draw_ok_std_error_map(
                self.results_all_se_figure,
                self.results_all_se_canvas,
                'results_all_se_colorbar',
                self.results_all_se_ax,
                context,
                std_error,
                all_indices,
                se_title,
                color_limits=se_limits,
            )
        elif hasattr(self, 'results_all_se_figure'):
            self._clear_ok_interpolation_plot_side(
                self.results_all_se_figure,
                self.results_all_se_canvas,
                'results_all_se_colorbar',
                QCoreApplication.translate(
                    "Tab 5", "Standard error could not be computed."
                ),
                default_placeholder=se_default,
            )

        self._update_results_cross_validation(
            context,
            network_indices=None,
            summary_table=self.results_all_cv_summary_table,
            results_table=self.results_all_cv_results_table,
            info_label=self.results_all_cv_info_label,
        )

    def _refresh_ok_interpolation_selected_wells(self, report_progress=True):
        """Builds the selected-wells OK map, SE map, and CV (spinbox-driven)."""
        if not hasattr(self, 'results_sel_interp_ax'):
            return

        if report_progress:
            self._set_progress(
                10,
                QCoreApplication.translate(
                    "Tab 5",
                    "State: Updating selected-wells maps...",
                ),
            )

        se_default = QCoreApplication.translate(
            "Tab 5", "No standard error map available"
        )
        completed_ok = False
        try:
            context, error = self._get_ok_interpolation_context()
            if context is None:
                self._clear_ok_interpolation_plot_side(
                    self.results_sel_interp_figure,
                    self.results_sel_interp_canvas,
                    'results_sel_interp_colorbar',
                    error,
                )
                self._clear_ok_interpolation_plot_side(
                    getattr(self, 'results_sel_se_figure', None),
                    getattr(self, 'results_sel_se_canvas', None),
                    'results_sel_se_colorbar',
                    error,
                    default_placeholder=se_default,
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
                self._clear_ok_interpolation_plot_side(
                    getattr(self, 'results_sel_se_figure', None),
                    getattr(self, 'results_sel_se_canvas', None),
                    'results_sel_se_colorbar',
                    default_placeholder=se_default,
                )
                return

            if report_progress:
                self._set_progress(
                    40,
                    QCoreApplication.translate(
                        "Tab 5", "State: Computing kriging..."
                    ),
                )

            indices = selection_order[:n_use]
            try:
                estimates, std_error = self._compute_ok_estimates_and_se(
                    context, indices
                )
            except Exception:
                estimates, std_error = None, None

            if (
                estimates is None
                or estimates.size != context['grid'].shape[0]
            ):
                message = QCoreApplication.translate(
                    "Tab 5", "Interpolation could not be computed."
                )
                self._clear_ok_interpolation_plot_side(
                    self.results_sel_interp_figure,
                    self.results_sel_interp_canvas,
                    'results_sel_interp_colorbar',
                    message,
                )
                self._clear_ok_interpolation_plot_side(
                    getattr(self, 'results_sel_se_figure', None),
                    getattr(self, 'results_sel_se_canvas', None),
                    'results_sel_se_colorbar',
                    message,
                    default_placeholder=se_default,
                )
                self._clear_results_cv_tables(
                    self.results_sel_cv_summary_table,
                    self.results_sel_cv_results_table,
                    self.results_sel_cv_info_label,
                    message,
                )
                return

            if report_progress:
                self._set_progress(
                    70,
                    QCoreApplication.translate(
                        "Tab 5", "State: Drawing maps..."
                    ),
                )

            self._clear_ok_interpolation_plot_side(
                self.results_sel_interp_figure,
                self.results_sel_interp_canvas,
                'results_sel_interp_colorbar',
                message=None,
            )
            self.results_sel_interp_ax = self.results_sel_interp_figure.axes[0]
            color_limits = self._ensure_ok_interpolation_color_limits(context)
            ok_parameter = context.get('ok_parameter') or context['attr_name']
            title = QCoreApplication.translate(
                "Tab 5",
                "O.K. – {param}\n({n} wells)",
            ).format(param=ok_parameter, n=n_use)
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

            if (
                hasattr(self, 'results_sel_se_figure')
                and std_error is not None
                and std_error.size == context['grid'].shape[0]
            ):
                self._clear_ok_interpolation_plot_side(
                    self.results_sel_se_figure,
                    self.results_sel_se_canvas,
                    'results_sel_se_colorbar',
                    message=None,
                    default_placeholder=se_default,
                )
                self.results_sel_se_ax = self.results_sel_se_figure.axes[0]
                se_limits = self._ensure_ok_se_color_limits(context)
                se_title = QCoreApplication.translate(
                    "Tab 5",
                    "S.E. – {param} \n({n} wells)",
                ).format(param=ok_parameter, n=n_use)
                self._draw_ok_std_error_map(
                    self.results_sel_se_figure,
                    self.results_sel_se_canvas,
                    'results_sel_se_colorbar',
                    self.results_sel_se_ax,
                    context,
                    std_error,
                    indices,
                    se_title,
                    color_limits=se_limits,
                )
            elif hasattr(self, 'results_sel_se_figure'):
                self._clear_ok_interpolation_plot_side(
                    self.results_sel_se_figure,
                    self.results_sel_se_canvas,
                    'results_sel_se_colorbar',
                    QCoreApplication.translate(
                        "Tab 5", "Standard error could not be computed."
                    ),
                    default_placeholder=se_default,
                )

            self._update_results_cross_validation(
                context,
                network_indices=indices,
                summary_table=self.results_sel_cv_summary_table,
                results_table=self.results_sel_cv_results_table,
                info_label=self.results_sel_cv_info_label,
            )
            completed_ok = True
        finally:
            if report_progress:
                if completed_ok:
                    combined_hint = self._tab5_combined_ok_progress_hint()
                    if combined_hint is not None:
                        self._set_progress(0, combined_hint)
                    else:
                        self._set_progress(
                            100,
                            QCoreApplication.translate(
                                "Tab 5", "State: Completed"
                            ),
                        )
                        QTimer.singleShot(1000, self._reset_progress)
                else:
                    self._reset_progress()

    def _refresh_ok_interpolation_plots(
        self, refresh_all=True, refresh_selected=True
    ):
        """Refreshes tab 5 maps/CV; the all-wells side skips spinbox-only updates."""
        if refresh_all:
            self._refresh_ok_interpolation_all_wells()
        if refresh_selected:
            # Nested refresh: parent owns the progress bar when both sides update.
            self._refresh_ok_interpolation_selected_wells(report_progress=False)

    def _refresh_ok_interpolation_plot(self):
        """Refreshes both tab 5 interpolation panels."""
        self._set_progress(
            10,
            QCoreApplication.translate(
                "Tab 5", "State: Updating interpolation maps..."
            ),
        )
        try:
            self._refresh_ok_interpolation_plots(
                refresh_all=True, refresh_selected=True
            )
            combined_hint = self._tab5_combined_ok_progress_hint()
            if combined_hint is not None:
                self._set_progress(0, combined_hint)
            else:
                self._set_progress(
                    100,
                    QCoreApplication.translate("Tab 5", "State: Completed"),
                )
                QTimer.singleShot(1000, self._reset_progress)
        except Exception:
            self._reset_progress(
                QCoreApplication.translate("Tab 5", "State: Error")
            )
            raise

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

        self._set_progress(
            20,
            QCoreApplication.translate(
                "Tab 5", "State: Downloading selected wells layer..."
            ),
        )
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

            self._set_progress(
                100,
                QCoreApplication.translate("Tab 5", "State: Completed"),
            )
            QTimer.singleShot(1000, self._reset_progress)
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
            self._reset_progress(
                QCoreApplication.translate("Tab 5", "State: Error")
            )
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 5", "Download selected wells as layer"),
                QCoreApplication.translate(
                    "Tab 5",
                    "Could not create the wells layer:\n{error}\n{details}",
                ).format(error=str(exc), details=traceback.format_exc()),
            )

    def _ok_estimates_to_raster_grid(self, grid_xy, estimates, nodata=-9999.0):
        """
        Maps scattered estimation-grid values onto a regular 2-D raster array.

        Wraps ``_grid_field_to_raster`` and adapts it for GDAL GeoTIFF export:
        north-up row order and ``nodata`` for cells outside the hull.

        Returns:
            (raster_array, geotransform) where raster_array is float32 with
            north-up row order (row 0 = max Y), and geotransform is a GDAL
            6-tuple (origin_x, pixel_width, 0, origin_y, 0, -pixel_height).
            Returns (None, None) when spacing cannot be inferred.
        """
        built = self._grid_field_to_raster(grid_xy, estimates)
        if built is None:
            return None, None
        xs, ys, z = built
        dx, dy = self._raster_cell_size(xs, ys)

        # North-up for GDAL: flip so row 0 is max Y; fill NaN with nodata.
        raster = np.flipud(z).astype(np.float32, copy=False)
        raster = np.where(np.isfinite(raster), raster, np.float32(nodata))

        # GDAL geotransform: top-left corner of the top-left pixel.
        origin_x = float(xs[0]) - dx / 2.0
        origin_y = float(ys[-1]) + dy / 2.0
        geotransform = (origin_x, dx, 0.0, origin_y, 0.0, -dy)
        return raster, geotransform

    def _write_temp_geotiff(
        self, raster, geotransform, crs, nodata=-9999.0, band_description=None
    ):
        """Writes a single-band Float32 GeoTIFF to a temp file; returns its path."""
        import tempfile
        from osgeo import gdal, osr

        path = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
        n_rows, n_cols = raster.shape
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            path,
            n_cols,
            n_rows,
            1,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW'],
        )
        if dataset is None:
            raise RuntimeError(
                QCoreApplication.translate(
                    "Tab 5", "Could not create the temporary GeoTIFF file."
                )
            )

        dataset.SetGeoTransform(geotransform)
        srs = osr.SpatialReference()
        auth_id = crs.authid() if crs is not None else ''
        if auth_id:
            srs.SetFromUserInput(auth_id)
            dataset.SetProjection(srs.ExportToWkt())

        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(float(nodata))
        if band_description:
            band.SetDescription(str(band_description))
        band.WriteArray(raster)
        band.FlushCache()
        dataset.FlushCache()
        dataset = None
        return path

    def _style_ok_interpolation_raster(self, raster_layer, estimates):
        """Applies a continuous viridis-like color ramp to the OK raster."""
        from qgis.core import (
            QgsColorRampShader,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )

        valid = np.asarray(estimates, dtype=float)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return

        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmin == vmax:
            pad = max(abs(vmin) * 0.05, 0.5)
            vmin -= pad
            vmax += pad

        # Approximate matplotlib viridis at low / mid / high values.
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
        color_ramp.setClassificationMode(QgsColorRampShader.Continuous)
        color_ramp.setMinimumValue(vmin)
        color_ramp.setMaximumValue(vmax)
        color_ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(vmin, QColor(68, 1, 84)),
            QgsColorRampShader.ColorRampItem(
                vmin + 0.5 * (vmax - vmin), QColor(33, 145, 140)
            ),
            QgsColorRampShader.ColorRampItem(vmax, QColor(253, 231, 37)),
        ])

        shader = QgsRasterShader()
        shader.setRasterShaderFunction(color_ramp)
        renderer = QgsSingleBandPseudoColorRenderer(
            raster_layer.dataProvider(), 1, shader
        )
        renderer.setClassificationMin(vmin)
        renderer.setClassificationMax(vmax)
        raster_layer.setRenderer(renderer)
        raster_layer.triggerRepaint()

    def save_ok_interpolation_as_layer(self):
        """Adds the selected-wells OK interpolation as a temporary raster layer."""
        title = QCoreApplication.translate(
            "Tab 5", "Download interpolation as layer"
        )
        context, error = self._get_ok_interpolation_context()
        if context is None:
            QMessageBox.warning(self, title, error)
            return

        layer = self.input_data_layer.currentLayer()
        if not layer:
            QMessageBox.warning(
                self,
                title,
                QCoreApplication.translate(
                    "Tab 5", "No input point layer is selected."
                ),
            )
            return

        n_requested = self.results_well_spin.value()
        selection_order = np.asarray(context['selection_order'], dtype=int)
        n_use = min(n_requested, selection_order.size)
        if n_use < 1:
            QMessageBox.warning(
                self,
                title,
                QCoreApplication.translate(
                    "Tab 5",
                    "No selected wells are available for interpolation.",
                ),
            )
            return

        indices = selection_order[:n_use]
        self._set_progress(
            15,
            QCoreApplication.translate(
                "Tab 5", "State: Downloading interpolation layer..."
            ),
        )
        try:
            estimates = self._compute_ok_estimates(context, indices)
        except Exception:
            estimates = None

        if estimates is None or estimates.size != context['grid'].shape[0]:
            self._reset_progress(
                QCoreApplication.translate("Tab 5", "State: Error")
            )
            QMessageBox.warning(
                self,
                title,
                QCoreApplication.translate(
                    "Tab 5", "Interpolation could not be computed."
                ),
            )
            return

        self._set_progress(
            50,
            QCoreApplication.translate(
                "Tab 5", "State: Building interpolation raster..."
            ),
        )
        nodata = -9999.0
        raster, geotransform = self._ok_estimates_to_raster_grid(
            context['grid'], estimates, nodata=nodata
        )
        if raster is None:
            self._reset_progress(
                QCoreApplication.translate("Tab 5", "State: Error")
            )
            QMessageBox.warning(
                self,
                title,
                QCoreApplication.translate(
                    "Tab 5",
                    "Could not build a regular raster from the estimation grid. "
                    "Use a rectangular or regularly spaced grid.",
                ),
            )
            return

        # Layer name uses the interpolated parameter, not "Parameters combined".
        display_name = context.get('ok_parameter') or context['attr_name']
        try:
            from qgis.core import QgsRasterLayer

            tif_path = self._write_temp_geotiff(
                raster,
                geotransform,
                layer.crs(),
                nodata=nodata,
                band_description=display_name,
            )
            layer_name = QCoreApplication.translate(
                "Tab 5",
                "OK Interpolation - {param} ({n} wells)",
            ).format(param=display_name, n=n_use)

            raster_layer = QgsRasterLayer(tif_path, layer_name)
            if not raster_layer.isValid():
                self._reset_progress(
                    QCoreApplication.translate("Tab 5", "State: Error")
                )
                QMessageBox.warning(
                    self,
                    title,
                    QCoreApplication.translate(
                        "Tab 5",
                        "The temporary raster layer could not be loaded.",
                    ),
                )
                return

            self._style_ok_interpolation_raster(raster_layer, estimates)
            QgsProject.instance().addMapLayer(raster_layer)

            self._set_progress(
                100,
                QCoreApplication.translate("Tab 5", "State: Completed"),
            )
            QTimer.singleShot(1000, self._reset_progress)
            QMessageBox.information(
                self,
                title,
                QCoreApplication.translate(
                    "Tab 5",
                    "Saved the OK interpolation for «{param}» "
                    "({n} wells) as a temporary raster layer.",
                ).format(param=display_name, n=n_use),
            )
        except Exception as exc:
            import traceback
            self._reset_progress(
                QCoreApplication.translate("Tab 5", "State: Error")
            )
            QMessageBox.warning(
                self,
                title,
                QCoreApplication.translate(
                    "Tab 5",
                    "Could not create the interpolation layer:\n{error}\n{details}",
                ).format(error=str(exc), details=traceback.format_exc()),
            )

    def _set_workflow_tabs_locked(self):
        """
        Disable navigation for tabs 2–5 (indices 1–4).

        Tab pages stay enabled so their widgets remain interactive; only the
        tab-bar labels are non-clickable via setTabEnabled. Always return the
        user to Tab 1 so they are never left on a locked page.
        """
        if not hasattr(self, 'tabs'):
            return
        self.tabs.setTabEnabled(0, True)
        for index in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(index, False)
        self.tabs.setCurrentIndex(0)

    def _enable_workflow_tab(self, index):
        """Enable one workflow tab by index (progressive unlock)."""
        if not hasattr(self, 'tabs'):
            return
        if 0 <= index < self.tabs.count():
            self.tabs.setTabEnabled(index, True)

    def previous_tab(self):
        """Go to the previous enabled tab."""
        current = self.tabs.currentIndex()
        for index in range(current - 1, -1, -1):
            if self.tabs.isTabEnabled(index):
                self.tabs.setCurrentIndex(index)
                return

    def next_tab(self):
        """
        Advance the guided workflow.

        From Tab 1 (index 0): run calculate_geostatistics; on success unlock
        Tab 2 and switch to it. From later tabs: unlock the next tab and move
        there. Geostatistics runs only from Tab 1.
        """
        current = self.tabs.currentIndex()
        last_index = self.tabs.count() - 1
        if current >= last_index:
            return

        if current == 0:
            if not self.calculate_geostatistics():
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Main Window", "Next"),
                    QCoreApplication.translate(
                        "Main Window",
                        "Select a point layer and at least one parameter on "
                        "tab 1, then try again.",
                    ),
                )
                return
            self._enable_workflow_tab(1)
            self.tabs.setCurrentIndex(1)
            return

        next_index = current + 1
        if current == 2 and not self._has_estimation_grid():
            self._update_workflow_navigation_state()
            return
        if current == 3 and not self._has_current_optimization():
            self._update_workflow_navigation_state()
            return
        self._enable_workflow_tab(next_index)
        self.tabs.setCurrentIndex(next_index)

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

    @staticmethod
    def _is_numeric_qgs_field(field):
        """True when the QgsField type is Int / Double / LongLong."""
        if field is None:
            return False
        return field.type() in (QVariant.Int, QVariant.Double, QVariant.LongLong)

    @classmethod
    def make_layer_attribute_table_item(cls, value, field, read_only_flags):
        """
        Build a read-only cell for Tab 1 layer attribute values.

        Numeric QgsFields store an int/float in Qt.DisplayRole so native
        QTableWidget header sorting is numeric (not lexicographic). Null or
        non-finite values stay as an empty string (not 0).
        """
        item = QTableWidgetItem()
        item.setFlags(read_only_flags)

        if value is None or (isinstance(value, QVariant) and value.isNull()):
            item.setData(Qt.DisplayRole, "")
            return item
        if isinstance(value, QVariant):
            value = value.value()
        if value is None:
            item.setData(Qt.DisplayRole, "")
            return item

        if cls._is_numeric_qgs_field(field):
            try:
                number = float(value)
            except (TypeError, ValueError):
                item.setData(
                    Qt.DisplayRole, cls._format_layer_field_value(value)
                )
                return item
            if not np.isfinite(number):
                item.setData(Qt.DisplayRole, "")
                return item
            # Prefer int DisplayRole for integer field types.
            if field.type() in (QVariant.Int, QVariant.LongLong):
                item.setData(Qt.DisplayRole, int(round(number)))
            else:
                item.setData(Qt.DisplayRole, float(number))
            return item

        item.setData(Qt.DisplayRole, cls._format_layer_field_value(value))
        return item

    def _all_layer_feature_ids(self, layer):
        """Returns the set of all feature IDs in the layer."""
        if not layer:
            return set()
        return {feature.id() for feature in layer.getFeatures()}

    def _included_well_fids_for_extract(self):
        """
        Feature IDs to pass to layer extractors.

        Returns None when every layer feature is included (no filtering),
        so extractors keep their default all-features behavior.
        """
        layer = (
            self.input_data_layer.currentLayer()
            if hasattr(self, 'input_data_layer')
            else None
        )
        if not layer or self._included_well_fids is None:
            return None

        all_fids = self._all_layer_feature_ids(layer)
        if not all_fids:
            return None
        if self._included_well_fids >= all_fids:
            return None
        return set(self._included_well_fids)

    def _set_layer_fields_include_checks(self, checked):
        """Updates visible Include checkboxes without emitting reset signals."""
        if not hasattr(self, 'layer_fields_table'):
            return
        self._syncing_well_include = True
        try:
            state = Qt.Checked if checked else Qt.Unchecked
            for row in range(self.layer_fields_table.rowCount()):
                item = self.layer_fields_table.item(row, 0)
                if item is not None:
                    item.setCheckState(state)
        finally:
            self._syncing_well_include = False

    def _select_all_wells(self):
        """Marks every well as included and resets downstream calculations."""
        layer = self.input_data_layer.currentLayer()
        if not layer:
            return
        self._included_well_fids = self._all_layer_feature_ids(layer)
        self._set_layer_fields_include_checks(True)
        self._on_included_wells_changed()

    def _deselect_all_wells(self):
        """Unmarks every well and resets downstream calculations."""
        layer = self.input_data_layer.currentLayer()
        if not layer:
            return
        self._included_well_fids = set()
        self._set_layer_fields_include_checks(False)
        self._on_included_wells_changed()

    def _on_layer_fields_include_item_changed(self, item):
        """Keeps the include set in sync when a row checkbox is toggled."""
        if self._syncing_well_include or item is None or item.column() != 0:
            return

        fid = item.data(Qt.UserRole)
        if fid is None:
            return
        if self._included_well_fids is None:
            self._included_well_fids = set()

        if item.checkState() == Qt.Checked:
            self._included_well_fids.add(fid)
        else:
            self._included_well_fids.discard(fid)

        self._on_included_wells_changed()

    def _on_included_wells_changed(self):
        """
        Clears Tab 2 / 4 / 5 results after the included-well set changes.

        Does not clear Tab 1 attribute selection; the user recalculates
        geostatistics on Tab 2 as usual.
        """
        self._cached_layer = None
        self._cached_attribute = None
        self._cached_raw_values = None
        self._cached_coordinates = None
        self._cached_point_ids = None
        self._cached_null_count = 0
        self.stats_by_attribute.clear()
        self._layer_data_by_attribute.clear()
        self.variogram_models_by_attribute.clear()
        self.variance_results = {}
        self._sync_var_params_table_from_store()
        self.clear_stats_table()
        self._clear_variogram_widget()
        self._clear_variance_reduction_plot()
        self._clear_prioritization_order_table()
        self._update_results_well_spinbox()
        self._clear_ok_interpolation_plots()
        self._update_mn_well_weight_info()
        self._update_mn_grid_weight_info()
        self._clear_stats_plots()
        self._clear_cross_validation()

    def _refresh_layer_fields_table(self, layer, max_rows=500):
        """Fills the tab 1 table with Include checkboxes and attribute values."""
        if not hasattr(self, 'layer_fields_table'):
            return

        self._syncing_well_include = True
        try:
            self.layer_fields_table.blockSignals(True)
            # Disable sorting while filling so rows are not reordered mid-loop.
            self.layer_fields_table.setSortingEnabled(False)
            self.layer_fields_table.clear()
            self.layer_fields_table.setRowCount(0)
            self.layer_fields_table.setColumnCount(0)

            if not layer:
                self._included_well_fids = None
                return

            fields = layer.fields()
            field_names = [field.name() for field in fields]
            if not field_names:
                self._included_well_fids = set()
                return

            # Track every feature ID so Select/Deselect all covers rows beyond
            # the table preview (max_rows).
            all_fids = set()
            features = []
            for feature in layer.getFeatures():
                all_fids.add(feature.id())
                if len(features) < max_rows:
                    features.append(feature)
            self._included_well_fids = all_fids

            headers = [
                QCoreApplication.translate("Tab 1", "Include")
            ] + field_names
            self.layer_fields_table.setColumnCount(len(headers))
            self.layer_fields_table.setRowCount(len(features))
            self.layer_fields_table.setHorizontalHeaderLabels(headers)

            check_flags = (
                Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )
            read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled

            for row, feature in enumerate(features):
                include_item = QTableWidgetItem()
                include_item.setFlags(check_flags)
                include_item.setCheckState(Qt.Checked)
                include_item.setData(Qt.UserRole, feature.id())
                self.layer_fields_table.setItem(row, 0, include_item)

                for col, field_name in enumerate(field_names):
                    # Numeric fields use numeric DisplayRole for native sort.
                    value_item = self.make_layer_attribute_table_item(
                        feature[field_name],
                        fields[col],
                        read_only,
                    )
                    self.layer_fields_table.setItem(row, col + 1, value_item)

            self.layer_fields_table.resizeColumnsToContents()
        finally:
            self.layer_fields_table.setSortingEnabled(True)
            self.layer_fields_table.blockSignals(False)
            self._syncing_well_include = False

    def _set_progress(self, percent=None, status_message=None):
        """
        Update the shared dialog progress bar.

        Args:
            percent: Progress value 0–100, or None to leave the value unchanged.
            status_message: Already-translated text shown inside the bar.
        """
        if not hasattr(self, 'progress_bar'):
            return
        if percent is not None:
            self.progress_bar.setValue(max(0, min(100, int(percent))))
        if status_message is not None:
            self.progress_bar.setFormat(status_message)
        QApplication.processEvents()

    def _tab1_progress_hint(self):
        """Idle guidance text for Tab 1 based on attribute selection."""
        if self._selected_analysis_parameters():
            return QCoreApplication.translate(
                "Tab 1", "Click Next to run the geostatistics"
            )
        return QCoreApplication.translate(
            "Tab 1",
            "Select an attribute of the layer to use as optimization param",
        )

    def _update_tab1_progress_hint(self):
        """Show Tab 1 guidance on the progress bar when idle on that tab."""
        if not hasattr(self, 'progress_bar') or not hasattr(self, 'tabs'):
            return
        if self.tabs.currentIndex() != 0:
            return
        if self.progress_bar.value() > 0:
            return
        self._set_progress(0, self._tab1_progress_hint())

    def _has_estimation_grid(self):
        """True when Tab 3 has a generated or imported estimation grid."""
        points = getattr(self, 'current_grid_points', None)
        return points is not None and len(points) > 0

    def _has_current_optimization(self):
        """True when Tab 4 has optimization results for the active parameter."""
        attr_name = self._current_mn_parameter()
        if not attr_name:
            return False
        return attr_name in getattr(self, 'variance_results', {})

    def _can_advance_from_tab(self, index):
        """Whether the guided workflow may move to the next tab from ``index``."""
        last_index = self.tabs.count() - 1
        if index >= last_index:
            return False
        if index == 2:
            return self._has_estimation_grid()
        if index == 3:
            return self._has_current_optimization()
        return True

    def _tab3_progress_hint(self):
        """Idle guidance on Tab 3 when no estimation grid is available."""
        if self._has_estimation_grid():
            return QCoreApplication.translate(
                "Tab 3", "Click Next to continue to well prioritization."
            )
        return QCoreApplication.translate(
            "Tab 3",
            "Calculate or import an estimation grid to continue.",
        )

    def _tab4_progress_hint(self):
        """Idle guidance on Tab 4 when optimization has not been run."""
        if self._has_current_optimization():
            return QCoreApplication.translate(
                "Tab 4", "Click Next to view the optimization map."
            )
        return QCoreApplication.translate(
            "Tab 4",
            "Run Optimize to compute well prioritization before continuing.",
        )

    def _update_tab3_progress_hint(self):
        """Show Tab 3 guidance on the progress bar when idle."""
        if not hasattr(self, 'progress_bar') or not hasattr(self, 'tabs'):
            return
        if self.tabs.currentIndex() != 2:
            return
        if self.progress_bar.value() > 0:
            return
        self._set_progress(0, self._tab3_progress_hint())

    def _update_tab4_progress_hint(self):
        """Show Tab 4 guidance on the progress bar when idle."""
        if not hasattr(self, 'progress_bar') or not hasattr(self, 'tabs'):
            return
        if self.tabs.currentIndex() != 3:
            return
        if self.progress_bar.value() > 0:
            return
        self._set_progress(0, self._tab4_progress_hint())

    def _update_workflow_navigation_state(self):
        """Refresh Next-button state and tab-specific idle progress hints."""
        if not hasattr(self, 'tabs') or not hasattr(self, 'next_btn'):
            return
        index = self.tabs.currentIndex()
        self.next_btn.setEnabled(self._can_advance_from_tab(index))
        if index == 0:
            self._update_tab1_progress_hint()
        elif index == 2:
            self._update_tab3_progress_hint()
        elif index == 3:
            self._update_mn_well_weight_info()
            self._update_tab4_progress_hint()
        elif index == self.tabs.count() - 1:
            self._update_tab5_progress_hint()

    def _tab5_combined_ok_progress_hint(self):
        """Idle note for Tab 5 when OK maps use the first combined parameter."""
        if not self._is_combined_mn_parameter(self._current_mn_parameter()):
            return None
        ok_parameter = self._tab5_reference_parameter()
        if not ok_parameter:
            return None
        return QCoreApplication.translate(
            "Tab 5",
            "O.K. Interpolation was generated with just the first listed "
            "parameter of the combined ones («{param}»).",
        ).format(param=ok_parameter)

    def _update_tab5_progress_hint(self):
        """Show the combined-OK note when idle on Tab 5."""
        if not hasattr(self, 'progress_bar') or not hasattr(self, 'tabs'):
            return
        # Tab 5 is the last main tab (Map).
        if self.tabs.currentIndex() != self.tabs.count() - 1:
            return
        if self.progress_bar.value() > 0:
            return
        hint = self._tab5_combined_ok_progress_hint()
        if hint is not None:
            self._set_progress(0, hint)

    def _on_main_tab_changed(self, index):
        """Update Next-button enable state and tab-specific progress hints."""
        self._update_workflow_navigation_state()

    def _reset_progress(self, status_message=None):
        """Reset the shared progress bar to idle (Ready) or a custom message."""
        if status_message is None:
            if hasattr(self, 'tabs'):
                tab_index = self.tabs.currentIndex()
                if tab_index == 0:
                    status_message = self._tab1_progress_hint()
                elif tab_index == 2:
                    status_message = self._tab3_progress_hint()
                elif tab_index == 3:
                    status_message = self._tab4_progress_hint()
                elif tab_index == self.tabs.count() - 1:
                    status_message = self._tab5_combined_ok_progress_hint()
                    if status_message is None:
                        status_message = QCoreApplication.translate(
                            "Main Window", "State: Ready"
                        )
                else:
                    status_message = QCoreApplication.translate(
                        "Main Window", "State: Ready"
                    )
            else:
                status_message = QCoreApplication.translate(
                    "Main Window", "State: Ready"
                )
        self._set_progress(0, status_message)
        self._update_workflow_navigation_state()

    def _set_variogram_progress(self, percent=None, status_message=None):
        """Tab 2 compatibility: write variogram status to the shared bar."""
        if getattr(self, '_suppress_variogram_progress', False):
            return
        message = None
        if status_message is not None:
            message = QCoreApplication.translate("Tab 2", status_message)
        self._set_progress(percent, message)

    def _reset_variogram_progress(self):
        """Tab 2 compatibility: reset shared bar after variogram work."""
        self._reset_progress(
            QCoreApplication.translate("Tab 2", "State: Ready")
        )

    def _clear_variogram_widget(self):
        """Resets the tab 2 variogram plot and parameters."""
        if not hasattr(self, 'variogram_widget'):
            return

        self._syncing_variogram = True
        try:
            self.variogram_widget.clear()
        finally:
            self._syncing_variogram = False

        self._clear_variogram_autofit_status()
        self._reset_variogram_progress()

    def _set_variogram_autofit_status(self, ok):
        """Show success/failure message above the variogram parameters table."""
        if not hasattr(self, 'var_autofit_status_label'):
            return
        label = self.var_autofit_status_label
        if ok is None:
            label.clear()
            label.setVisible(False)
            return
        if ok:
            label.setText(
                QCoreApplication.translate(
                    "Tab 2",
                    "Variogram parameters auto-fit converged. "
                    "You can compare different models type.",
                )
            )
            label.setStyleSheet("color: #1e8449; font-style: italic;")
        else:
            label.setText(
                QCoreApplication.translate(
                    "Tab 2",
                    "Auto-fit did not converge; check outliers, variogram limit, lag size and adjust the model "
                    "parameters manually to fit to the experimental variogram.",
                )
            )
            label.setStyleSheet("color: #a04000; font-style: italic;")
        label.setVisible(True)

    def _refresh_variogram_autofit_status(self, attr_name=None):
        """Refresh autofit status label from stored state for one parameter."""
        if attr_name is None:
            attr_name = self._current_analysis_attribute()
        if not attr_name:
            self._clear_variogram_autofit_status()
            return
        state = self._get_variogram_state(attr_name)
        if not state or 'autofit_ok' not in state:
            self._clear_variogram_autofit_status()
            return
        self._set_variogram_autofit_status(bool(state.get('autofit_ok')))

    def _clear_variogram_autofit_status(self):
        """Hide the autofit status label (layer/data reset)."""
        self._set_variogram_autofit_status(None)

    def _clear_estimation_grid(self):
        """
        Clears Tab 3 estimation-grid state and plot.

        Called when the Tab 1 input layer changes so generated or imported
        nodes from the previous layer cannot be reused for optimization or
        Tab 5 interpolation.
        """
        self.current_grid_points = None
        self.current_grid_info = None
        if hasattr(self, 'current_ceg_values'):
            self.current_ceg_values = None
        if hasattr(self, 'prioritization_grid_points'):
            self.prioritization_grid_points = None
        for attr in (
            'grid_type_saved',
            'n_nodes_saved',
            'buffer_saved',
            'buffer_spacing_factor_saved',
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)

        if hasattr(self, 'grid_figure') and hasattr(self, 'grid_canvas'):
            self.grid_figure.clear()
            ax = self.grid_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                QCoreApplication.translate(
                    "Tab 3", "No estimation grid. Calculate or import a grid."
                ),
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='gray',
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.grid_canvas.draw()

        if hasattr(self, 'estimated_points_label'):
            self.estimated_points_label.setText(
                QCoreApplication.translate("Tab 3", "N/A")
            )

        self._update_mn_grid_weight_info()
        self._update_workflow_navigation_state()

    def on_layer_changed(self, layer):
        """Update the layer information and the numeric attributes."""
        self._cached_layer = None
        self._cached_attribute = None
        self._cached_raw_values = None
        self._cached_coordinates = None
        self._cached_point_ids = None
        self._cached_null_count = 0
        self.stats_by_attribute.clear()
        self._layer_data_by_attribute.clear()
        self.variogram_models_by_attribute.clear()
        self.variance_results = {}
        self._tab1_params_for_optimization = frozenset()
        self._sync_var_params_table_from_store()
        self.selected_data_parameters.clear()
        self.clear_stats_table()
        self._clear_variogram_widget()
        self._sync_attr_name_combo()
        self._clear_variance_reduction_plot()
        self._clear_prioritization_order_table()
        self._update_results_well_spinbox()
        self._clear_ok_interpolation_plots()
        self._clear_estimation_grid()
        # New layer invalidates the guided workflow; tabs 2–5 lock again.
        self._set_workflow_tabs_locked()
        self._update_mn_well_weight_info()
        self._update_mn_grid_weight_info()

        if not layer:
            self._nn_stats = None
            self._set_experimental_variogram_settings_store(
                ExperimentalVariogramSettings()
            )
            self._sync_experimental_variogram_controls_from_settings()
            self.layer_info.clear()
            self._refresh_layer_fields_table(None)
            self._apply_default_node_spacing(None)
            if hasattr(self, 'estimated_points_label'):
                self.estimated_points_label.setText(
                    QCoreApplication.translate("Tab 3", "N/A")
                )
            self._update_tab1_progress_hint()
            return

        self._refresh_layer_fields_table(layer)
        # ANN once per layer selection; drives lag defaults and Tab 3 spacing.
        self._refresh_ann_for_layer(layer)
        # Recalculate Tab 3 default spacing and node estimate for the new layer.
        self._apply_default_node_spacing(layer)
        if hasattr(self, 'update_estimated_points'):
            self.update_estimated_points()
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
        self._update_tab1_progress_hint()

    def _current_analysis_attribute(self):
        """Returns the Tab 2 active parameter (stats-table row selection), or None."""
        attr = getattr(self, '_active_analysis_attribute', None)
        if not attr:
            return None
        # Drop stale names if Tab 1 selection changed.
        selected = self._selected_analysis_parameters()
        if attr not in selected:
            return None
        return attr

    def _highlight_active_stats_row(self):
        """Mark the active parameter row with bold text (no selection fill)."""
        if not hasattr(self, 'stats_table'):
            return

        self.stats_table.clearSelection()

        attr = self._current_analysis_attribute()
        active_row = (
            self._stats_table_row_for_attribute(attr) if attr else -1
        )

        for row in range(self.stats_table.rowCount()):
            bold = row == active_row
            header_item = self.stats_table.verticalHeaderItem(row)
            if header_item is not None:
                font = header_item.font()
                font.setBold(bold)
                header_item.setFont(font)
            for col in range(self.stats_table.columnCount()):
                item = self.stats_table.item(row, col)
                if item is None:
                    continue
                font = item.font()
                font.setBold(bold)
                item.setFont(font)

    def _on_stats_table_parameter_header_clicked(self, row):
        """
        Activate geostatistics views when the user clicks a parameter name
        in the leftmost (vertical header) cell of the stats table.
        """
        parameters = self._selected_analysis_parameters()
        if row < 0 or row >= len(parameters):
            return
        attr_name = parameters[row]
        if attr_name == self._active_analysis_attribute:
            # Re-clicking the same row still re-syncs views (e.g. after edits).
            self._highlight_active_stats_row()
            self._on_tab2_attribute_changed()
            return
        self._active_analysis_attribute = attr_name
        self._highlight_active_stats_row()
        self._on_tab2_attribute_changed()

    def _sync_attr_name_combo(self):
        """
        Keep Tab 2 active parameter and Tab 4 parameter combo in sync with
        Tab 1 attribute selection (legacy name retained for call sites).
        """
        selected = self._selected_analysis_parameters()
        previous = getattr(self, '_active_analysis_attribute', None)
        if not selected:
            self._active_analysis_attribute = None
        elif previous in selected:
            self._active_analysis_attribute = previous
        else:
            self._active_analysis_attribute = selected[0]
        self._highlight_active_stats_row()
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
                    "Tab 4", "Parameters combined (Weighted)"
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
        selected = self._selected_analysis_parameters()
        attr_name = self._current_mn_parameter()
        # Prefer Tab 1 selection so a single selected parameter still resolves
        # a lookup attribute when the Tab 4 combo is empty/disabled.
        if self._is_combined_mn_parameter(attr_name):
            lookup_attr = selected[0] if selected else None
        elif selected:
            lookup_attr = selected[0]
        elif attr_name:
            lookup_attr = attr_name
        else:
            lookup_attr = None

        weight_field = None
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

    def _on_mn_well_weight_toggled(self, _checked):
        """User toggled well weights: rankings that used the old setting are stale."""
        if getattr(self, 'variance_results', None):
            self._clear_all_optimization_results(
                tab5_message=QCoreApplication.translate(
                    "Tab 5",
                    "Weight option changed. Run Optimize on tab 4 again.",
                ),
                tab4_message=QCoreApplication.translate(
                    "Tab 4",
                    "Well weight option changed. Click Optimize again.",
                ),
            )
        self._update_mn_well_weight_info()

    def _on_mn_grid_weight_toggled(self, _checked):
        """User toggled grid weights: rankings that used the old setting are stale."""
        if getattr(self, 'variance_results', None):
            self._clear_all_optimization_results(
                tab5_message=QCoreApplication.translate(
                    "Tab 5",
                    "Weight option changed. Run Optimize on tab 4 again.",
                ),
                tab4_message=QCoreApplication.translate(
                    "Tab 4",
                    "Grid weight option changed. Click Optimize again.",
                ),
            )
        self._update_mn_grid_weight_info()

    def _on_mn_parameter_changed(self, _index):
        """Show cached Optimize results for the selected parameter, or clear UI."""
        self._update_results_well_spinbox()
        self._update_mn_well_weight_info()
        self._update_mn_grid_weight_info()
        self._load_optimization_views_for_current_parameter()

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
        selected = self._selected_analysis_parameters()
        if (
            not hasattr(self, 'mn_param_select_combo')
            or not self.mn_param_select_combo.isEnabled()
        ):
            # Combo may be disabled before sync; Tab 1 selection is still valid.
            return selected[0] if selected else None
        data = self.mn_param_select_combo.currentData()
        if data == MN_COMBINED_PARAMETERS_KEY:
            return MN_COMBINED_PARAMETERS_KEY
        if data:
            return str(data)
        text = self.mn_param_select_combo.currentText()
        if text == QCoreApplication.translate("Tab 4", "No parameter selected"):
            return selected[0] if selected else None
        if text:
            return text
        return selected[0] if selected else None

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
            ) = extract_point_records_from_layer(
                layer,
                name,
                include_fids=self._included_well_fids_for_extract(),
            )
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
                    weight=float(
                        state.get(
                            'weight',
                            self._equal_parameter_weight(len(param_names)),
                        )
                    ),
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

    def _reset_mn_progress(self, status_text=None):
        """Tab 4 compatibility: reset shared bar after optimization."""
        if status_text is None:
            status_text = QCoreApplication.translate("Tab 4", "State: Ready")
        self._reset_progress(status_text)

    def _update_mn_progress(self, percent, status_text=None):
        """Tab 4 compatibility: write optimization status to the shared bar."""
        self._set_progress(percent, status_text)

    def _optimization_progress_callback(self, fraction, message="", **kwargs):
        """Translate optimization phase messages and update the progress bar."""
        if message == "preparing":
            status = QCoreApplication.translate(
                "Tab 4", "State: Preparing optimization..."
            )
        elif message == "selecting":
            status = QCoreApplication.translate(
                "Tab 4", "State: Selecting wells ({current}/{total})..."
            ).format(
                current=kwargs.get("current", 0),
                total=kwargs.get("total", 0),
            )
        elif message == "selecting_adverse":
            status = QCoreApplication.translate(
                "Tab 4",
                "State: Selecting adverse order ({current}/{total})...",
            ).format(
                current=kwargs.get("current", 0),
                total=kwargs.get("total", 0),
            )
        elif message == "evaluating":
            status = QCoreApplication.translate(
                "Tab 4", "State: Evaluating variance ({current}/{total})..."
            ).format(
                current=kwargs.get("current", 0),
                total=kwargs.get("total", 0),
            )
        elif message == "evaluating_adverse":
            status = QCoreApplication.translate(
                "Tab 4",
                "State: Evaluating adverse variance ({current}/{total})...",
            ).format(
                current=kwargs.get("current", 0),
                total=kwargs.get("total", 0),
            )
        elif message == "complete":
            status = QCoreApplication.translate("Tab 4", "State: Completed")
        elif message == "error":
            status = QCoreApplication.translate("Tab 4", "State: Error")
        else:
            status = message or None

        self._update_mn_progress(int(round(fraction * 100)), status)

    def _clear_variance_reduction_plot(self):
        """Resets the variance reduction plot and info label."""
        if not hasattr(self, 'variance_ax'):
            return

        self._disconnect_point_hover(getattr(self, 'variance_canvas', None))
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

        self._reset_mn_progress()
        self._clear_prioritization_order_table()

    def _clear_prioritization_order_table(self):
        """Clears the tab 4 optimization-order table."""
        if not hasattr(self, 'mn_prioritization_order_table'):
            return
        self.mn_prioritization_order_table.setRowCount(0)

    def _update_prioritization_order_table(self, results, param_key=None):
        """Fills tab 4 optimization-order table (priority + adverse ranks)."""
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
        # Adverse remaining variance (%) along adverse_selection_order (index 0 = baseline).
        adverse_total_variance = np.asarray(
            results.get('adverse_total_variance_percent', []), dtype=float
        )

        # Map well index -> 1-based adverse rank (least reduction first).
        adverse_rank_by_well_idx = {}
        adverse_order = results.get('adverse_selection_order')
        if adverse_order is not None:
            for adverse_rank, well_idx in enumerate(
                np.asarray(adverse_order, dtype=int), start=1
            ):
                adverse_rank_by_well_idx[int(well_idx)] = adverse_rank

        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        table = self.mn_prioritization_order_table
        # Avoid re-sorting on every setItem while rows are being filled.
        table.setSortingEnabled(False)
        table.setRowCount(selection_order.size)

        for row, well_idx in enumerate(selection_order):
            rank = row + 1
            well_idx = int(well_idx)
            well_id = (
                str(point_ids[well_idx])
                if point_ids is not None
                else str(well_idx)
            )
            if well_weights is not None and well_idx < well_weights.size:
                weight_val = float(well_weights[well_idx])
            else:
                weight_val = 1.0

            if rank < total_variance.size:
                total_var_val = float(total_variance[rank])
                if not np.isfinite(total_var_val):
                    total_var_val = None
            else:
                total_var_val = None

            adverse_rank = adverse_rank_by_well_idx.get(well_idx)
            adverse_var_val = None
            if (
                adverse_rank is not None
                and adverse_rank < adverse_total_variance.size
            ):
                adverse_var_val = float(adverse_total_variance[adverse_rank])
                if not np.isfinite(adverse_var_val):
                    adverse_var_val = None

            # Numeric DisplayRole so header sorting is numeric (not lexicographic).
            priority_item = QTableWidgetItem()
            priority_item.setData(Qt.DisplayRole, int(rank))
            priority_item.setFlags(read_only)
            table.setItem(row, 0, priority_item)

            id_item = QTableWidgetItem(well_id)
            id_item.setFlags(read_only)
            table.setItem(row, 1, id_item)

            weight_item = QTableWidgetItem()
            weight_item.setData(Qt.DisplayRole, float(weight_val))
            weight_item.setFlags(read_only)
            table.setItem(row, 2, weight_item)

            var_item = QTableWidgetItem()
            if total_var_val is None:
                var_item.setData(Qt.DisplayRole, format_total_variance_pct(None))
            else:
                var_item.setData(Qt.DisplayRole, float(total_var_val))
            var_item.setFlags(read_only)
            table.setItem(row, 3, var_item)

            adverse_item = QTableWidgetItem()
            if adverse_rank is None:
                adverse_item.setData(Qt.DisplayRole, "—")
            else:
                adverse_item.setData(Qt.DisplayRole, int(adverse_rank))
            adverse_item.setFlags(read_only)
            table.setItem(row, 4, adverse_item)

            adverse_var_item = QTableWidgetItem()
            if adverse_var_val is None:
                adverse_var_item.setData(
                    Qt.DisplayRole, format_total_variance_pct(None)
                )
            else:
                adverse_var_item.setData(Qt.DisplayRole, float(adverse_var_val))
            adverse_var_item.setFlags(read_only)
            table.setItem(row, 5, adverse_var_item)

        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    def _variance_curve_hover_points(self, n_points, y_values, selection_order, point_ids):
        """
        Build hover markers for a variance-reduction curve.

        n_points[0] is the zero-wells baseline (no ID). For n >= 1 the well
        added at that step is selection_order[n - 1].
        """
        n_points = np.asarray(n_points, dtype=float).ravel()
        y_values = np.asarray(y_values, dtype=float).ravel()
        selection_order = np.asarray(selection_order, dtype=int).ravel()
        if (
            point_ids is None
            or n_points.size == 0
            or y_values.size != n_points.size
            or selection_order.size == 0
        ):
            return np.empty((0, 2)), []

        xs = []
        ys = []
        labels = []
        for x_val, y_val in zip(n_points, y_values):
            n_wells = int(round(float(x_val)))
            if n_wells < 1 or n_wells > selection_order.size:
                continue
            if not np.isfinite(y_val):
                continue
            well_idx = int(selection_order[n_wells - 1])
            if well_idx < 0 or well_idx >= len(point_ids):
                continue
            xs.append(float(x_val))
            ys.append(float(y_val))
            labels.append(str(point_ids[well_idx]))

        if not labels:
            return np.empty((0, 2)), []
        return np.column_stack([xs, ys]), labels

    def _disconnect_point_hover(self, canvas):
        """Disconnect prior motion handler; clear canvas._point_hover_* refs."""
        if canvas is None:
            return
        cid = getattr(canvas, '_point_hover_cid', None)
        if cid is not None:
            canvas.mpl_disconnect(cid)
        canvas._point_hover_cid = None
        canvas._point_hover_ann = None

    def _attach_point_hover(
        self, canvas, ax, xy, labels, *, pixel_tol=12, format_label=None
    ):
        """
        Show an annotation for the nearest point under the cursor.

        Nearest neighbor uses screen pixels via ax.transData.transform(xy)
        so x/y units with different scales still feel natural.

        xy: (N, 2) data coordinates
        labels: sequence length N (well IDs as raw strings; never translated)
        format_label: optional callable(label) -> str; default str(label)

        The annotation is excluded from constrained_layout so edge tooltips
        do not stretch the axes. Offset flips toward the axes interior.
        """
        self._disconnect_point_hover(canvas)
        if canvas is None or ax is None:
            return

        hover_coords = np.asarray(xy, dtype=float)
        if hover_coords.ndim != 2 or hover_coords.shape[0] == 0:
            return
        if len(labels) != hover_coords.shape[0]:
            return

        if format_label is None:
            format_label = str
        # Precompute display strings so the motion handler stays cheap.
        display_labels = [format_label(label) for label in labels]

        offset_pts = 8
        hover_ann = ax.annotate(
            '',
            xy=(0, 0),
            xytext=(offset_pts, offset_pts),
            textcoords='offset points',
            fontsize=8,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'),
            visible=False,
            zorder=10,
        )
        # Keep tooltip out of layout so draw_idle does not reflow margins.
        hover_ann.set_in_layout(False)
        canvas._point_hover_ann = hover_ann

        def _place_label_inward(ann, display_xy):
            """Flip offset/alignment so the label stays inside ax.bbox."""
            bbox = ax.bbox
            dx = offset_pts if display_xy[0] <= bbox.x0 + bbox.width * 0.5 else -offset_pts
            dy = offset_pts if display_xy[1] <= bbox.y0 + bbox.height * 0.5 else -offset_pts
            ann.set_position((dx, dy))
            ann.set_ha('left' if dx > 0 else 'right')
            ann.set_va('bottom' if dy > 0 else 'top')

        def on_motion(event):
            ann = canvas._point_hover_ann
            if ann is None:
                return
            if event.inaxes != ax or event.x is None or event.y is None:
                if ann.get_visible():
                    ann.set_visible(False)
                    canvas.draw_idle()
                return

            # Pixel-space distance: independent of data-axis scale differences.
            pts = ax.transData.transform(hover_coords)
            mouse = np.array([event.x, event.y])
            dist = np.hypot(pts[:, 0] - mouse[0], pts[:, 1] - mouse[1])
            nearest = int(np.argmin(dist))
            if dist[nearest] < pixel_tol:
                label = display_labels[nearest]
                point_xy = (
                    float(hover_coords[nearest, 0]),
                    float(hover_coords[nearest, 1]),
                )
                prev = (
                    ann.get_text(),
                    ann.get_visible(),
                    ann.get_position(),
                    ann.get_ha(),
                    ann.get_va(),
                )
                ann.xy = point_xy
                ann.set_text(label)
                _place_label_inward(ann, pts[nearest])
                ann.set_visible(True)
                now = (
                    ann.get_text(),
                    ann.get_visible(),
                    ann.get_position(),
                    ann.get_ha(),
                    ann.get_va(),
                )
                if now != prev:
                    canvas.draw_idle()
            elif ann.get_visible():
                ann.set_visible(False)
                canvas.draw_idle()

        canvas._point_hover_cid = canvas.mpl_connect(
            'motion_notify_event', on_motion
        )

    def _update_variance_reduction_plot(self, results, display_name, param_key=None):
        """Plot remaining total variance (%) for priority and adverse orders.

        Well ID labels appear on marker hover (same pattern as tab 5 maps).
        90%/95% reference lines follow the prioritized (max-reduction) curve.
        """
        self._disconnect_point_hover(getattr(self, 'variance_canvas', None))
        ax = self.variance_ax
        ax.clear()

        n_points = np.asarray(results['n_points'], dtype=float)
        total_variance = resolve_total_variance_series(results)

        ax.plot(
            n_points,
            total_variance,
            marker='o',
            color='#3aa82d',
            linewidth=1,
            markersize=3,
            label=QCoreApplication.translate("Tab 4", "Prioritized wells"),
        )

        adverse_n_points = results.get('adverse_n_points')
        adverse_total = results.get('adverse_total_variance_percent')
        adverse_order = results.get('adverse_selection_order')
        if adverse_n_points is not None and adverse_total is not None:
            adverse_n_points = np.asarray(adverse_n_points, dtype=float)
            adverse_total = np.asarray(adverse_total, dtype=float)
            ax.plot(
                adverse_n_points,
                adverse_total,
                marker='o',
                color='#e13232',
                linewidth=1,
                markersize=3,
                label=QCoreApplication.translate("Tab 4", "Adverse order"),
            )
        else:
            adverse_total = None
            adverse_n_points = None

        final_pct = float(total_variance[-1]) if total_variance.size else 0.0
        max_reduction_pct = 100.0 - final_pct
        if max_reduction_pct > 0:
            level_90 = 100.0 - 0.9 * max_reduction_pct
            level_95 = 100.0 - 0.95 * max_reduction_pct
            ax.axhline(
                level_90,
                linestyle='dotted',
                color='deepskyblue',
                linewidth=1,
                label=QCoreApplication.translate(
                    "Tab 4", "90% of max reduction ({:.2f}%)".format(level_90)
               
                ),
            )
            ax.axhline(
                level_95,
                linestyle='dotted',
                color='mediumblue',
                linewidth=1,
                label=QCoreApplication.translate(
                    "Tab 4", "95% of max reduction ({:.2f}%)".format(level_95)
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

        # Y-limits span both priority and adverse series when available.
        y_series = [total_variance] if total_variance.size else []
        if adverse_total is not None and adverse_total.size:
            y_series.append(adverse_total)
        if y_series:
            y_min = float(min(np.min(series) for series in y_series))
            y_max = float(max(np.max(series) for series in y_series))
            span = y_max - y_min
            margin = max(0.02 * span, 1.0) if span > 0 else 2.0
            ax.set_ylim(
                max(0.0, y_min - margin),
                min(100.0, y_max + margin),
            )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        self.variance_figure.tight_layout()
        self.variance_canvas.draw()

        # Hover IDs for markers that correspond to an added well (n >= 1).
        if param_key is None:
            param_key = self._current_mn_parameter()
        point_ids = None
        if param_key is not None:
            param_data = self._get_mn_parameter_data(param_key)
            if param_data is not None:
                point_ids = param_data.get('point_ids')

        hover_coords_list = []
        hover_labels = []
        selection_order = results.get('selection_order')
        if selection_order is not None and point_ids is not None:
            coords, labels = self._variance_curve_hover_points(
                n_points, total_variance, selection_order, point_ids
            )
            if labels:
                hover_coords_list.append(coords)
                hover_labels.extend(labels)
        if (
            adverse_n_points is not None
            and adverse_total is not None
            and adverse_order is not None
            and point_ids is not None
        ):
            coords, labels = self._variance_curve_hover_points(
                adverse_n_points, adverse_total, adverse_order, point_ids
            )
            if labels:
                hover_coords_list.append(coords)
                hover_labels.extend(labels)

        if hover_labels:
            self._attach_point_hover(
                self.variance_canvas,
                self.variance_ax,
                np.vstack(hover_coords_list),
                hover_labels,
            )

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
        """Export prioritization (layer attributes + results) and variogram settings."""
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
        adverse_total_variance = np.asarray(
            results.get('adverse_total_variance_percent', []), dtype=float
        )

        if self._is_combined_mn_parameter(attr_name):
            selected_parameters = self._selected_analysis_parameters()
            (
                _coordinates,
                point_ids,
                _well_weights,
                _parameters,
                error,
            ) = self._load_optimization_parameters(selected_parameters)
            if error:
                point_ids = None
            parameter_label = self._mn_combined_parameters_label()
            reference_attribute = (
                selected_parameters[0] if selected_parameters else attr_name
            )
        else:
            param_data = self._get_mn_parameter_data(attr_name)
            point_ids = param_data.get('point_ids') if param_data else None
            parameter_label = attr_name
            reference_attribute = attr_name

        layer = self.input_data_layer.currentLayer()
        field_names, records_by_point_id = build_point_id_layer_attribute_map(
            layer,
            reference_attribute,
            include_fids=self._included_well_fids_for_extract(),
        )

        # Map well index -> 1-based adverse rank (least reduction first).
        adverse_rank_by_well_idx = {}
        adverse_order = results.get('adverse_selection_order')
        if adverse_order is not None:
            for adverse_rank, well_idx in enumerate(
                np.asarray(adverse_order, dtype=int), start=1
            ):
                adverse_rank_by_well_idx[int(well_idx)] = adverse_rank

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

        # Layer attributes first; prioritization metrics appended .
        prioritization_suffix_columns = [
            'Priority',
            'Parameter',
            'Variance',
            'Adverse_Order',
            'Adverse_Variance',
            'Use_Well_Weight',
            'Use_Grid_Weight',
        ]
        prioritization_columns = field_names + prioritization_suffix_columns

        prioritization_rows = []
        for rank, well_idx in enumerate(selection_order, start=1):
            if rank >= total_variance.size:
                break
            well_idx = int(well_idx)
            well_id = (
                str(point_ids[well_idx])
                if point_ids is not None
                else str(well_idx)
            )
            layer_attrs = records_by_point_id.get(well_id, {})
            row = {
                field_name: self._format_layer_field_value(
                    layer_attrs.get(field_name)
                )
                for field_name in field_names
            }
            # Prioritization columns override any same-named layer fields.
            row['Priority'] = rank
            row['Parameter'] = parameter_label
            row['Variance'] = float(total_variance[rank])
            adverse_rank = adverse_rank_by_well_idx.get(well_idx)
            row['Adverse_Order'] = (
                adverse_rank if adverse_rank is not None else ''
            )
            if (
                adverse_rank is not None
                and adverse_rank < adverse_total_variance.size
            ):
                row['Adverse_Variance'] = float(
                    adverse_total_variance[adverse_rank]
                )
            else:
                row['Adverse_Variance'] = ''
            row['Use_Well_Weight'] = bool(use_well_weights)
            row['Use_Grid_Weight'] = bool(use_grid_weights)
            prioritization_rows.append(row)

        cv_summary_export_columns = {
            'min': 'CV_Min_Error',
            'max': 'CV_Max_Error',
            'mean': 'CV_Mean_Error',
            'mae': 'CV_MAE',
            'rmse': 'CV_RMSE',
            'ase': 'CV_ASE',
            'mse': 'CV_MSE',
            'rmsse': 'CV_RMSSE',
        }
        variogram_rows = []
        for param_name in sorted(self.variogram_models_by_attribute.keys()):
            state = self._get_variogram_state(param_name)
            if not state:
                continue
            row = {
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
            }
            # Tab 2 leave-one-out CV over all wells (same metrics as the summary table).
            cv_result, _point_ids, _cv_error = (
                self._compute_parameter_cross_validation(
                    param_name, network_indices=None
                )
            )
            summary = (
                cv_result.get('summary', {}) if cv_result is not None else {}
            )
            for key, column_name in cv_summary_export_columns.items():
                value = summary.get(key, np.nan)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = np.nan
                row[column_name] = (
                    number if np.isfinite(number) else ''
                )
            variogram_rows.append(row)
        variogram_columns = [
            'Parameter',
            'Weight',
            'Transform',
            'Model',
            'Nugget',
            'Sill',
            'Range',
            'R2',
        ] + [cv_summary_export_columns[key] for key in CV_SUMMARY_KEYS]

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
            marker='o',
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
        self._reset_mn_progress(
            QCoreApplication.translate("Tab 4", "State: Starting optimization...")
        )
        QApplication.processEvents()

        results = None
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
                    self._reset_mn_progress()
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
                self._reset_mn_progress()
                return

            results = compute_variance_reduction_curve(
                optimization_input,
                progress_callback=self._optimization_progress_callback,
            )
        except Exception as exc:
            self._optimization_progress_callback(0.0, "error")
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
            self._optimization_progress_callback(0.0, "error")
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
        self._update_workflow_navigation_state()

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
        #After 1.5 seconds, reset the optimization progress label.
        QTimer.singleShot(1500, self._reset_mn_progress)

        self._update_results_well_spinbox()
        self._refresh_ok_interpolation_plot()

    def _on_tab2_attribute_changed(self, _index=None):
        """Switch histogram, map, variogram and CV to the active parameter."""
        attr = self._current_analysis_attribute()
        if not attr:
            self._clear_stats_plots()
            self._clear_cross_validation()
            return
        self._sync_var_params_table_from_store()
        self._refresh_tab2_parameter_views(attr)

    def calculate_geostatistics(self):
        """
        Calculate descriptive statistics for all Tab-1 selected parameters.

        Returns:
            bool: True when at least one parameter was computed successfully.
        """
        return self._calculate_all_attribute_statistics(force_reextract=True)

    def _begin_stats_table_sync(self):
        """Enter a nested programmatic update of stats_table widgets."""
        self._stats_table_sync_depth += 1

    def _end_stats_table_sync(self):
        """Leave one nested programmatic update of stats_table widgets."""
        self._stats_table_sync_depth = max(0, self._stats_table_sync_depth - 1)

    def _is_syncing_stats_table(self):
        """True while any nested stats_table sync block is active."""
        return self._stats_table_sync_depth > 0

    def _get_attribute_log_transform(self, attr_name):
        """Per-parameter log transform stored in variogram state."""
        state = self._get_variogram_state(attr_name)
        if state is not None:
            return bool(state.get('log_transform', False))
        return False

    def _create_stats_transform_combo(self, log_transform=False):
        """Build a per-row transform combo for stats_table.

        Identity of the parameter is resolved from the table row of ``sender()``
        when the index changes, so reused cell widgets stay correct after
        stats_table rebuilds.
        """
        combo = WheelIgnoringComboBox()
        combo.addItems([
            QCoreApplication.translate("Tab 2", "None"),
            QCoreApplication.translate("Tab 2", "Logarithmic"),
        ])
        combo.setToolTip(
            QCoreApplication.translate(
                "Tab 2",
                "Data transform for this parameter. Changing it recalculates "
                "statistics and variogram for this parameter only.",
            )
        )
        combo.blockSignals(True)
        combo.setCurrentIndex(1 if log_transform else 0)
        combo.blockSignals(False)
        combo.currentIndexChanged.connect(self._on_stats_transform_combo_changed)
        return combo

    def _set_stats_table_transform_cell(self, row, attr_name):
        """Install or refresh the transform combo on one stats table row."""
        log_transform = self._get_attribute_log_transform(attr_name)
        existing = self.stats_table.cellWidget(row, STATS_TRANSFORM_COL)
        if isinstance(existing, QComboBox):
            self._begin_stats_table_sync()
            existing.blockSignals(True)
            try:
                existing.setCurrentIndex(1 if log_transform else 0)
            finally:
                existing.blockSignals(False)
                self._end_stats_table_sync()
            return
        if existing is not None:
            self.stats_table.removeCellWidget(row, STATS_TRANSFORM_COL)
        combo = self._create_stats_transform_combo(log_transform)
        self.stats_table.setCellWidget(row, STATS_TRANSFORM_COL, combo)

    def _sync_stats_transform_combo(self, attr_name, log_transform):
        """Set transform combo index without triggering recalculation."""
        row = self._stats_table_row_for_attribute(attr_name)
        if row < 0:
            return
        combo = self.stats_table.cellWidget(row, STATS_TRANSFORM_COL)
        if combo is None:
            return
        self._begin_stats_table_sync()
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(1 if log_transform else 0)
        finally:
            combo.blockSignals(False)
            self._end_stats_table_sync()

    def _recalculate_single_attribute_statistics(self, attr_name):
        """
        Recompute stats, variogram and CV for one parameter.

        Used when the per-row transform combo changes.
        """
        ok, error = self._compute_and_store_attribute_statistics(
            attr_name, force_reextract=False
        )
        if ok:
            row = self._stats_table_row_for_attribute(attr_name)
            if row >= 0:
                self._begin_stats_table_sync()
                self.stats_table.blockSignals(True)
                try:
                    self._populate_stats_table_row(row, attr_name)
                    self._adjust_stats_table_height()
                finally:
                    self.stats_table.blockSignals(False)
                    self._end_stats_table_sync()
            # Transform change invalidates the previous fit; re-estimate once.
            self._fit_variogram_for_attribute(attr_name, force_reextract=True)
            if attr_name == self._current_analysis_attribute():
                self._refresh_tab2_parameter_views(attr_name)
        return ok, error

    def _stats_transform_needs_recalculation(self, attr_name, log_transform):
        """
        True when combo/store transform is out of sync with stored stats.

        Recalculate when the requested transform differs from the variogram
        store, when stats are missing, when stored stats used a different
        transform, or when the variogram data fingerprint is stale.
        """
        prev_transform = self._get_attribute_log_transform(attr_name)
        if log_transform != prev_transform:
            return True

        stored = self.stats_by_attribute.get(attr_name)
        if not stored:
            return True
        if bool(stored.get('log_transform', False)) != log_transform:
            return True

        values = stored.get('values')
        if values is None:
            return True
        expected_fp = self._variogram_data_fingerprint(
            attr_name, log_transform, values
        )
        state = self._get_variogram_state(attr_name) or {}
        if state.get('data_fingerprint') != expected_fp:
            return True
        return False

    def _on_stats_transform_combo_changed(self, _index=None):
        """
        Recalculate one parameter when its row transform combo changes.

        Resolves the parameter from the table row that owns ``sender()`` so
        reused combo widgets never apply the change to the wrong attribute.
        Also activates that parameter for plots/variogram/CV.
        """
        if self._is_syncing_stats_table():
            return

        combo = self.sender()
        if not isinstance(combo, QComboBox):
            return

        row = -1
        for candidate_row in range(self.stats_table.rowCount()):
            if (
                self.stats_table.cellWidget(
                    candidate_row, STATS_TRANSFORM_COL
                )
                is combo
            ):
                row = candidate_row
                break
        if row < 0:
            return

        layer = self.input_data_layer.currentLayer()
        if not layer:
            return

        parameters = self._selected_analysis_parameters()
        if row >= len(parameters):
            return
        attr_name = parameters[row]

        log_transform = combo.currentIndex() == 1
        if not self._stats_transform_needs_recalculation(
            attr_name, log_transform
        ):
            return

        prev_transform = self._get_attribute_log_transform(attr_name)

        # Selecting the transformed parameter drives histogram, map, variogram, CV.
        self._active_analysis_attribute = attr_name
        self._highlight_active_stats_row()

        state = dict(self._get_variogram_state(attr_name) or {})
        state['log_transform'] = log_transform
        if 'weight' not in state:
            state['weight'] = self._get_attribute_weight(attr_name)
        self._set_variogram_state(attr_name, state)

        ok, error = self._recalculate_single_attribute_statistics(attr_name)
        if not ok:
            state['log_transform'] = prev_transform
            self._set_variogram_state(attr_name, state)
            self._sync_stats_transform_combo(attr_name, prev_transform)
            self.stats_info_label.setText(
                QCoreApplication.translate(
                    "Tab 2", "«{param}»: {error}"
                ).format(
                    param=attr_name,
                    error=error or QCoreApplication.translate(
                        "Tab 2", "Recalculation failed."
                    ),
                )
            )
            self.stats_info_label.setStyleSheet(
                "color: red; font-style: italic;"
            )
            return

        # Keep var-params table in sync; views already refreshed when active.
        self._sync_var_params_table_from_store()
        stored = self.stats_by_attribute.get(attr_name)
        if stored:
            self._set_stats_view_info_label(attr_name, stored)

    def _equal_parameter_weight(self, n_parameters=None):
        """Equal share so N parameter defaults sum to 1.0."""
        if n_parameters is None:
            n_parameters = len(self._selected_analysis_parameters())
        if n_parameters <= 0:
            return 1.0
        return 1.0 / n_parameters

    def _apply_equal_parameter_weights(self, parameters=None):
        """Set weight = 1/N for every selected parameter (overwrites existing)."""
        if parameters is None:
            parameters = self._selected_analysis_parameters()
        n = len(parameters)
        if n == 0:
            return
        share = self._equal_parameter_weight(n)
        for attr_name in parameters:
            state = dict(self._get_variogram_state(attr_name) or {})
            state['weight'] = share
            if 'log_transform' not in state:
                state['log_transform'] = False
            self._set_variogram_state(attr_name, state)

    def _format_parameter_weight(self, weight):
        """Format a parameter weight for display in stats_table."""
        return f"{float(weight):.2f}"

    def _get_attribute_weight(self, attr_name):
        """Per-parameter combination weight stored in variogram state (tab 4)."""
        state = self._get_variogram_state(attr_name)
        if state is not None and 'weight' in state:
            return float(state['weight'])
        return self._equal_parameter_weight()

    def _set_stats_table_weight_cell(self, row, attr_name):
        """Write the editable weight cell for one stats table row."""
        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        editable = read_only | Qt.ItemIsEditable
        weight_text = self._format_parameter_weight(
            self._get_attribute_weight(attr_name)
        )
        item = self.stats_table.item(row, STATS_COL_WEIGHT)
        if item is None:
            item = QTableWidgetItem(weight_text)
            self.stats_table.setItem(row, STATS_COL_WEIGHT, item)
        else:
            item.setText(weight_text)
        item.setFlags(editable)

    def _revert_stats_table_weight_cell(self, row, attr_name):
        """Restore the weight cell from stored state after invalid input."""
        self._begin_stats_table_sync()
        try:
            self._set_stats_table_weight_cell(row, attr_name)
        finally:
            self._end_stats_table_sync()

    def _on_stats_table_weight_changed(self, item):
        """
        Persist weight when the inline editor commits (Enter, Tab, or focus lost).

        itemChanged does not fire on every keystroke while typing.
        """
        if self._is_syncing_stats_table() or item.column() != STATS_COL_WEIGHT:
            return

        row = item.row()
        parameters = self._selected_analysis_parameters()
        if row < 0 or row >= len(parameters):
            return

        attr_name = parameters[row]
        try:
            weight = float(item.text())
        except (ValueError, TypeError):
            self._revert_stats_table_weight_cell(row, attr_name)
            return

        if weight <= 0:
            self._revert_stats_table_weight_cell(row, attr_name)
            return

        state = dict(self._get_variogram_state(attr_name) or {})
        state['weight'] = weight
        self._set_variogram_state(attr_name, state)

        self._begin_stats_table_sync()
        try:
            item.setText(self._format_parameter_weight(weight))
        finally:
            self._end_stats_table_sync()

    def _stats_table_row_for_attribute(self, attr_name):
        parameters = self._selected_analysis_parameters()
        try:
            return parameters.index(attr_name)
        except ValueError:
            return -1

    def _stats_table_estimated_viewport_width(self, row_count):
        """Estimate stats_table viewport width before the first layout pass."""
        table = self.stats_table
        width = table.viewport().width()
        if width > 0:
            return width

        width = table.width()
        parent = table.parentWidget()
        while width <= 0 and parent is not None:
            width = parent.width()
            parent = parent.parentWidget()
        if width <= 0:
            return 0

        width -= 2 * table.frameWidth()
        if table.verticalHeader().isVisible():
            width -= table.verticalHeader().width()
        if row_count > STATS_TABLE_MAX_VISIBLE_ROWS:
            width -= table.verticalScrollBar().sizeHint().width()
        return max(width, 0)

    def _stats_table_horizontal_scrollbar_height(self, row_count):
        """
        Extra height so the horizontal scrollbar does not cover the last row.

        QTableWidget draws the H-scrollbar inside the widget bounds. If the
        fixed table height only reserves row space, the bar overlaps the last
        row. Use a reliable scrollbar extent (style metric when sizeHint is 0)
        and detect overflow after columns have been sized.
        """
        del row_count  # reserved for call-site compatibility
        table = self.stats_table
        if table.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff:
            return 0

        hsb = table.horizontalScrollBar()
        sb_height = hsb.sizeHint().height()
        if sb_height <= 0:
            sb_height = table.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        if sb_height <= 0:
            sb_height = 16

        content_width = table.horizontalHeader().length()
        viewport_width = table.viewport().width()
        if viewport_width <= 0:
            viewport_width = self._stats_table_estimated_viewport_width(
                table.rowCount()
            )

        # Prefer isVisible after layout; also compare extents. When the
        # viewport is not ready yet, still reserve scrollbar height so the
        # first paint does not cover the last row.
        if viewport_width <= 0:
            if not getattr(self, '_stats_height_relayout_pending', False):
                self._stats_height_relayout_pending = True
                QTimer.singleShot(0, self._adjust_stats_table_height_after_layout)
            return sb_height

        if hsb.isVisible() or content_width > viewport_width:
            return sb_height
        return 0

    def _adjust_stats_table_height_after_layout(self):
        """Second-pass height fix once the stats_table has a real viewport size."""
        self._stats_height_relayout_pending = False
        if not hasattr(self, 'stats_table'):
            return
        if getattr(self, '_adjusting_stats_table_height', False):
            return

        table = self.stats_table
        hsb = table.horizontalScrollBar()
        # Recompute with a known viewport; avoid scheduling another pass.
        self._adjusting_stats_table_height = True
        try:
            table.resizeRowsToContents()
            table.resizeColumnsToContents()
            header_height = (
                0
                if table.horizontalHeader().isHidden()
                else table.horizontalHeader().height()
            )
            frame = table.frameWidth() * 2
            row_count = table.rowCount()
            if row_count == 0:
                height = header_height + frame + 4
            else:
                visible_rows = min(row_count, STATS_TABLE_MAX_VISIBLE_ROWS)
                rows_height = sum(
                    table.rowHeight(row) for row in range(visible_rows)
                )
                height = header_height + rows_height + frame
                content_width = table.horizontalHeader().length()
                viewport_width = table.viewport().width()
                sb_height = hsb.sizeHint().height()
                if sb_height <= 0:
                    sb_height = table.style().pixelMetric(
                        QStyle.PM_ScrollBarExtent
                    )
                if sb_height <= 0:
                    sb_height = 16
                if (
                    hsb.isVisible()
                    or (
                        viewport_width > 0
                        and content_width > viewport_width
                    )
                ):
                    height += sb_height
            table.setMinimumHeight(height)
            table.setMaximumHeight(height)
        finally:
            self._adjusting_stats_table_height = False

    def _adjust_stats_table_height(self):
        """Set stats_table height to fit visible rows and reserve scrollbar space."""
        if getattr(self, '_adjusting_stats_table_height', False):
            return
        if not hasattr(self, 'stats_table'):
            return

        self._adjusting_stats_table_height = True
        try:
            table = self.stats_table
            table.resizeRowsToContents()
            # Column widths affect whether an H-scrollbar is required.
            table.resizeColumnsToContents()

            header_height = (
                0
                if table.horizontalHeader().isHidden()
                else table.horizontalHeader().height()
            )
            frame = table.frameWidth() * 2
            row_count = table.rowCount()

            if row_count == 0:
                height = header_height + frame + 4
                table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            else:
                visible_rows = min(row_count, STATS_TABLE_MAX_VISIBLE_ROWS)
                rows_height = sum(
                    table.rowHeight(row) for row in range(visible_rows)
                )
                height = header_height + rows_height + frame
                if row_count <= STATS_TABLE_MAX_VISIBLE_ROWS:
                    table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                else:
                    table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

                height += self._stats_table_horizontal_scrollbar_height(
                    row_count
                )

            table.setMinimumHeight(height)
            table.setMaximumHeight(height)
        finally:
            self._adjusting_stats_table_height = False
            # One deferred pass after geometry is applied (no recursion).
            if not getattr(self, '_stats_height_relayout_pending', False):
                self._stats_height_relayout_pending = True
                QTimer.singleShot(0, self._adjust_stats_table_height_after_layout)

    def _extract_layer_data_for_attribute(self, attr_name, force_reextract=False):
        """Extract or return cached raw layer data for one parameter."""
        layer = self.input_data_layer.currentLayer()
        if not layer or not attr_name:
            return None

        cached = self._layer_data_by_attribute.get(attr_name)
        if (
            not force_reextract
            and cached is not None
            and cached.get('layer') is layer
        ):
            return cached

        point_ids, coordinates, raw_values, null_count, _well_weights = (
            extract_point_records_from_layer(
                layer,
                attr_name,
                include_fids=self._included_well_fids_for_extract(),
            )
        )
        if (
            point_ids is None
            or coordinates is None
            or raw_values is None
            or len(raw_values) == 0
        ):
            return {
                'error': QCoreApplication.translate(
                    "Tab 2", "No valid data for «{param}»."
                ).format(param=attr_name),
            }

        cached = {
            'layer': layer,
            'point_ids': point_ids,
            'coordinates': coordinates,
            'raw_values': raw_values,
            'null_count': null_count,
        }
        self._layer_data_by_attribute[attr_name] = cached
        return cached

    def _compute_and_store_attribute_statistics(
        self, attr_name, force_reextract=False
    ):
        """
        Compute and store descriptive statistics for one parameter.

        Variogram fitting is intentionally separate (see
        ``_fit_variogram_for_attribute``) so Tab 1 can batch-fit all
        parameters once, and Tab 2 row clicks can reuse stored models.
        """
        layer = self.input_data_layer.currentLayer()
        if not layer:
            return False, QCoreApplication.translate("Tab 2", "No layer selected.")

        layer_data = self._extract_layer_data_for_attribute(
            attr_name, force_reextract=force_reextract
        )
        if not layer_data:
            return False, QCoreApplication.translate(
                "Tab 2", "No valid data for «{param}»."
            ).format(param=attr_name)
        if layer_data.get('error'):
            return False, layer_data['error']

        raw_values = layer_data['raw_values']
        coordinates = layer_data['coordinates']
        point_ids = layer_data['point_ids']
        null_count = layer_data.get('null_count', 0)

        if (
            raw_values is None
            or coordinates is None
            or point_ids is None
            or len(raw_values) == 0
        ):
            return False, QCoreApplication.translate(
                "Tab 2", "No valid data for «{param}»."
            ).format(param=attr_name)

        log_transform = self._get_attribute_log_transform(attr_name)
        transform = 'log' if log_transform else 'none'
        coordinates, transformed, excluded_non_positive, error = (
            align_coordinates_with_transform(coordinates, raw_values, transform)
        )
        if error:
            return False, error

        stats_dict = compute_descriptive_stats(transformed)
        if stats_dict is None:
            return False, QCoreApplication.translate(
                "Tab 2",
                "No valid data after applying the transformation for «{param}».",
            ).format(param=attr_name)

        aligned_point_ids = align_point_ids_with_transform(
            point_ids, raw_values, transform
        )

        self.stats_by_attribute[attr_name] = {
            'stats': stats_dict,
            'coordinates': coordinates,
            'values': transformed,
            'point_ids': aligned_point_ids,
            'null_count': null_count,
            'excluded_non_positive': excluded_non_positive,
            'transform': transform,
            'log_transform': log_transform,
        }

        existing_state = self._get_variogram_state(attr_name) or {}
        synced_state = dict(existing_state)
        synced_state['log_transform'] = log_transform
        if 'weight' not in synced_state:
            synced_state['weight'] = self._equal_parameter_weight()
        self._set_variogram_state(attr_name, synced_state)

        return True, None

    def _fit_variogram_for_attribute(self, attr_name, force_reextract=True):
        """
        Estimate and store the variogram model for one parameter.

        Uses values already held in ``stats_by_attribute``. When
        ``force_reextract`` is True, always auto-fits; otherwise restores the
        stored model when the data fingerprint still matches.
        """
        stored = self.stats_by_attribute.get(attr_name)
        if not stored:
            return False
        self._update_variogram_from_statistics(
            stored['coordinates'],
            stored['values'],
            attr_name,
            stored.get('log_transform', False),
            force_reextract=force_reextract,
        )
        return True

    def _display_stored_variogram(self, attr_name):
        """
        Load the active parameter into the variogram widget without re-fitting.

        Used when the user selects a row in the Tab 2 stats table so switching
        parameters only redraws the already-estimated model from the store.
        """
        if not hasattr(self, 'variogram_widget'):
            return
        stored = self.stats_by_attribute.get(attr_name)
        if not stored:
            self.variogram_widget.clear()
            self._clear_variogram_autofit_status()
            return

        log_transform = stored.get('log_transform', False)
        self.variogram_widget.set_data(
            stored['coordinates'],
            stored['values'],
            attr_name,
            log_transform=log_transform,
            auto_fit=False,
        )
        if self._get_variogram_state(attr_name):
            self._apply_store_to_widget(attr_name)
        self._refresh_variogram_autofit_status(attr_name)

    def _calculate_all_attribute_statistics(self, force_reextract=False):
        """
        Calculate statistics and estimate variograms for every Tab-1 parameter.

        Phase 1 stores descriptive statistics. Phase 2 auto-fits a variogram
        for each successful parameter (progress reported per parameter). Tab 2
        row selection then only displays the stored models.

        Returns:
            bool: True if at least one parameter succeeded; False otherwise.
        """
        layer = self.input_data_layer.currentLayer()
        parameters = self._selected_analysis_parameters()

        if not layer or not parameters:
            self.clear_stats_table()
            return False

        n_params = len(parameters)
        self._set_progress(
            0,
            QCoreApplication.translate(
                "Tab 1", "State: Calculating geostatistics..."
            ),
        )
        try:
            self._sync_stats_table_structure()

            success_count = 0
            error_messages = []
            # Phase 1: descriptive statistics for every selected parameter.
            for index, attr_name in enumerate(parameters):
                percent = int(round((index / max(n_params, 1)) * 40))
                self._set_progress(
                    percent,
                    QCoreApplication.translate(
                        "Tab 1",
                        "State: Calculating statistics for «{param}» "
                        "({current}/{total})...",
                    ).format(
                        param=attr_name,
                        current=index + 1,
                        total=n_params,
                    ),
                )
                ok, error = self._compute_and_store_attribute_statistics(
                    attr_name, force_reextract=force_reextract
                )
                if ok:
                    success_count += 1
                elif error:
                    error_messages.append(
                        QCoreApplication.translate(
                            "Tab 2", "«{param}»: {error}"
                        ).format(param=attr_name, error=error)
                    )

            # Phase 2: auto-fit variograms for parameters with valid stats.
            fitted_names = [
                name for name in parameters if name in self.stats_by_attribute
            ]
            n_fit = len(fitted_names)
            self._batch_fitting_variograms = True
            self._suppress_variogram_progress = True
            try:
                for index, attr_name in enumerate(fitted_names):
                    percent = 40 + int(
                        round((index / max(n_fit, 1)) * 55)
                    )
                    self._set_progress(
                        percent,
                        QCoreApplication.translate(
                            "Tab 1",
                            "State: Estimating variogram for «{param}» "
                            "({current}/{total})...",
                        ).format(
                            param=attr_name,
                            current=index + 1,
                            total=n_fit,
                        ),
                    )
                    self._fit_variogram_for_attribute(
                        attr_name, force_reextract=True
                    )
            finally:
                self._batch_fitting_variograms = False
                self._suppress_variogram_progress = False

            # Re-fitted models invalidate any previous Tab 4 optimization.
            self._invalidate_optimization_after_variogram_change(None)

            self._set_progress(
                95,
                QCoreApplication.translate(
                    "Tab 1", "State: Updating geostatistics views..."
                ),
            )
            self._populate_stats_table_all()
            self._sync_var_params_table_from_store()

            if success_count:
                self.stats_info_label.setText(
                    QCoreApplication.translate(
                        "Tab 2",
                        "Statistics and variograms calculated for {n} "
                        "parameter(s). Click a parameter name to inspect "
                        "plots and the stored variogram.",
                    ).format(n=success_count)
                )
                self.stats_info_label.setStyleSheet(
                    "color: gray; font-style: italic;"
                )
            elif error_messages:
                self.stats_info_label.setText(" · ".join(error_messages))
                self.stats_info_label.setStyleSheet(
                    "color: red; font-style: italic;"
                )

            active_attr = self._current_analysis_attribute()
            if active_attr and active_attr in self.stats_by_attribute:
                self._refresh_tab2_parameter_views(active_attr)
            elif success_count:
                self._refresh_tab2_parameter_views(parameters[0])

            if success_count:
                self._set_progress(
                    100,
                    QCoreApplication.translate("Tab 1", "State: Completed"),
                )
                QTimer.singleShot(1500, self._reset_progress)
            else:
                self._reset_progress(
                    QCoreApplication.translate("Tab 1", "State: Error")
                )
            return success_count > 0
        except Exception:
            self._batch_fitting_variograms = False
            self._suppress_variogram_progress = False
            self._reset_progress(
                QCoreApplication.translate("Tab 1", "State: Error")
            )
            raise

    def _set_stats_view_info_label(self, attr_name, stored):
        """Info label for the active parameter's plots and variogram views."""
        info_parts = [
            QCoreApplication.translate(
                "Tab 2", "Viewing «{param}»"
            ).format(param=attr_name)
        ]
        null_count = stored.get('null_count', 0)
        if null_count > 0:
            info_parts.append(
                QCoreApplication.translate(
                    "Tab 2", "{count} null value(s) omitted"
                ).format(count=null_count)
            )
        transform = stored.get('transform', 'none')
        excluded = stored.get('excluded_non_positive', 0)
        if transform == 'log' and excluded > 0:
            info_parts.append(
                QCoreApplication.translate(
                    "Tab 2", "{count} value(s) ≤ 0 omitted for log"
                ).format(count=excluded)
            )
        if transform == 'log':
            info_parts.append(
                QCoreApplication.translate("Tab 2", "(Log transformation)")
            )
        self.stats_info_label.setText(" · ".join(info_parts))
        self.stats_info_label.setStyleSheet("color: gray; font-style: italic;")

    def _refresh_tab2_parameter_views(self, attr_name):
        """
        Refresh plots, stored variogram and CV for one parameter.

        Does not re-estimate the variogram: models are fitted when leaving
        Tab 1 (or when the transform changes). Row clicks only switch views.
        """
        if not attr_name:
            self._clear_stats_plots()
            self._clear_cross_validation()
            return

        stored = self.stats_by_attribute.get(attr_name)
        if not stored:
            self._clear_stats_plots()
            self._clear_cross_validation()
            return

        self._update_stats_plots(
            stored['coordinates'],
            stored['values'],
            attr_name,
            point_ids=stored.get('point_ids'),
        )
        self._display_stored_variogram(attr_name)
        self._refresh_tab2_cross_validation(attr_name)
        self._set_stats_view_info_label(attr_name, stored)
        self._sync_var_params_table_from_store()

    def _sync_stats_table_structure(self):
        """Resize stats table rows when Tab-1 parameter selection changes."""
        parameters = self._selected_analysis_parameters()
        new_param_set = frozenset(parameters)
        previous_param_set = getattr(
            self, '_tab1_params_for_optimization', frozenset()
        )
        # Any change to Tab 1 selection invalidates all Optimize caches.
        if previous_param_set != new_param_set and getattr(
            self, 'variance_results', None
        ):
            self._clear_all_optimization_results(
                tab5_message=QCoreApplication.translate(
                    "Tab 5",
                    "Selected parameters changed. Run Optimize on tab 4 again.",
                ),
                tab4_message=QCoreApplication.translate(
                    "Tab 4",
                    "Selected parameters changed. Click Optimize again.",
                ),
            )
        self._tab1_params_for_optimization = new_param_set

        self._sync_attr_name_combo()

        if not parameters:
            self.stats_table.setRowCount(0)
            self._adjust_stats_table_height()
            self.stats_by_attribute.clear()
            self._layer_data_by_attribute.clear()
            self.stats_info_label.setText(
                QCoreApplication.translate(
                    "Tab 2",
                    "Select attributes on tab 1, then click Next to calculate "
                    "geostatistics. Click a parameter name in the table to "
                    "inspect plots and the variogram.",
                )
            )
            self.stats_info_label.setStyleSheet(
                "color: gray; font-style: italic;"
            )
            return

        self.stats_by_attribute = {
            name: data
            for name, data in self.stats_by_attribute.items()
            if name in parameters
        }
        self._layer_data_by_attribute = {
            name: data
            for name, data in self._layer_data_by_attribute.items()
            if name in parameters
        }

        self._apply_equal_parameter_weights(parameters)

        self.stats_table.setRowCount(len(parameters))
        self.stats_table.setVerticalHeaderLabels(parameters)
        self.stats_table.verticalHeader().setVisible(True)

        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        for row in range(len(parameters)):
            attr_name = parameters[row]
            self._set_stats_table_weight_cell(row, attr_name)
            self._set_stats_table_transform_cell(row, attr_name)
            for col in range(
                STATS_VALUE_COL_OFFSET, self.stats_table.columnCount()
            ):
                item = self.stats_table.item(row, col)
                if item is None:
                    item = QTableWidgetItem("—")
                    self.stats_table.setItem(row, col, item)
                else:
                    item.setText("—")
                item.setFlags(read_only)
                item.setToolTip("")
                item.setBackground(QBrush())

        self._populate_stats_table_all()
        self._highlight_active_stats_row()

    def _populate_stats_table_row(self, row, attr_name):
        """Fill one stats table row from stored statistics."""
        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled

        self._set_stats_table_weight_cell(row, attr_name)
        self._set_stats_table_transform_cell(row, attr_name)

        stored = self.stats_by_attribute.get(attr_name)

        if stored and stored.get('stats'):
            stats_dict = stored['stats']
            for col, key in enumerate(STAT_KEYS):
                table_col = col + STATS_VALUE_COL_OFFSET
                value = stats_dict[key]
                if key == 'count':
                    text = str(int(value))
                elif np.isnan(value):
                    text = "—"
                else:
                    text = f"{value:.3g}"
                item = self.stats_table.item(row, table_col)
                if item is None:
                    item = QTableWidgetItem(text)
                    self.stats_table.setItem(row, table_col, item)
                else:
                    item.setText(text)
                item.setFlags(read_only)
                apply_distribution_shape_cell_style(item, key, value)
        else:
            for col in range(
                STATS_VALUE_COL_OFFSET, self.stats_table.columnCount()
            ):
                item = self.stats_table.item(row, col)
                if item is None:
                    item = QTableWidgetItem("—")
                    self.stats_table.setItem(row, col, item)
                else:
                    item.setText("—")
                item.setFlags(read_only)
                item.setToolTip("")
                item.setBackground(QBrush())

    def _populate_stats_table_all(self):
        """Fill every stats table row from stats_by_attribute."""
        parameters = self._selected_analysis_parameters()
        if not parameters:
            return
        self._begin_stats_table_sync()
        self.stats_table.blockSignals(True)
        try:
            for row, attr_name in enumerate(parameters):
                self._populate_stats_table_row(row, attr_name)
            self.stats_table.resizeColumnsToContents()
        finally:
            self.stats_table.blockSignals(False)
            self._end_stats_table_sync()
        self._adjust_stats_table_height()
        self._highlight_active_stats_row()

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

        attribute = self._current_analysis_attribute()
        combo = self.var_params_table.cellWidget(row, VAR_COL_MODEL)
        if not attribute or combo is None:
            return

        param_item = self.var_params_table.item(row, VAR_COL_PARAM)
        if not param_item or param_item.text() != attribute:
            return

        model_type = combo.currentText()
        state = dict(self._get_variogram_state(attribute) or {})
        state['model_type'] = model_type
        self._set_variogram_state(attribute, state)

        if attribute != self._current_analysis_attribute():
            return

        widget_data = getattr(self.variogram_widget, 'current_data', None)
        if not widget_data or widget_data.get('attribute') != attribute:
            return

        # Model type is already in the store; auto_fit reads it and writes R² back.
        # Experimental settings (factor/lag) are session-level and are not reset here.
        self.variogram_widget.auto_fit()
        self._refresh_tab2_cross_validation(attribute)
        self._invalidate_optimization_after_variogram_change(attribute)

    def _experimental_variogram_settings(self):
        """Return clamped session experimental variogram settings."""
        settings = getattr(
            self, '_experimental_variogram_settings_store', None
        )
        if not isinstance(settings, ExperimentalVariogramSettings):
            settings = ExperimentalVariogramSettings()
        return ExperimentalVariogramSettings(
            factor_max_dist=clamp_variogram_factor_max_dist(
                settings.factor_max_dist
            ),
            lag_size=float(settings.lag_size),
            n_bins=max(1, int(settings.n_bins)),
        )

    def _set_experimental_variogram_settings_store(self, settings):
        """Persist session experimental settings (clamped copy)."""
        if not isinstance(settings, ExperimentalVariogramSettings):
            settings = ExperimentalVariogramSettings()
        self._experimental_variogram_settings_store = (
            ExperimentalVariogramSettings(
                factor_max_dist=clamp_variogram_factor_max_dist(
                    settings.factor_max_dist
                ),
                lag_size=float(settings.lag_size),
                n_bins=max(1, int(settings.n_bins)),
            )
        )

    def _layer_coordinates_for_ann(self, layer=None):
        """Extract point coordinates from the input layer for ANN."""
        if layer is None and hasattr(self, 'input_data_layer'):
            layer = self.input_data_layer.currentLayer()
        if layer is None:
            return None
        coords = []
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom is None or geom.isEmpty() or geom.isMultipart():
                continue
            point = geom.asPoint()
            coords.append([point.x(), point.y()])
        if len(coords) < 2:
            return None
        return np.asarray(coords, dtype=float)

    @staticmethod
    def _max_dist_from_coordinates(coordinates):
        """Domain diameter used for variogram lag cutoff."""
        from scipy.spatial import distance as scipy_distance

        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] < 2:
            return 1.0
        if len(coordinates) > 100:
            x_range = float(
                np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
            )
            y_range = float(
                np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
            )
            max_dist = float(np.sqrt(x_range ** 2 + y_range ** 2))
        else:
            max_dist = float(np.max(scipy_distance.pdist(coordinates)))
        if not np.isfinite(max_dist) or max_dist <= 0.0:
            return 1.0
        return max_dist

    def _refresh_ann_for_layer(self, layer=None):
        """Compute ANN once on layer select and reset experimental defaults."""
        coordinates = self._layer_coordinates_for_ann(layer)
        if coordinates is None:
            self._nn_stats = None
            self._set_experimental_variogram_settings_store(
                ExperimentalVariogramSettings()
            )
            self._sync_experimental_variogram_controls_from_settings()
            return
        self._nn_stats = compute_nearest_neighbor_stats(coordinates)
        max_dist = self._max_dist_from_coordinates(coordinates)
        self._set_experimental_variogram_settings_store(
            build_default_experimental_variogram_settings(
                max_dist, self._nn_stats
            )
        )
        self._sync_experimental_variogram_controls_from_settings()

    def _sync_experimental_variogram_controls_from_settings(self):
        """Refresh lag combo and hint from session settings."""
        if not hasattr(self, 'var_lag_size_combo'):
            return
        settings = self._experimental_variogram_settings()
        self._syncing_experimental_variogram_controls = True
        try:
            self._rebuild_lag_size_combo(select_lag=settings.lag_size)
        finally:
            self._syncing_experimental_variogram_controls = False
        self._update_experimental_variogram_hint()

    def _rebuild_lag_size_combo(self, select_lag=None):
        """Fill lag combo from current D_o choices (static labels only)."""
        combo = self.var_lag_size_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            d_o = 1.0
            if self._nn_stats is not None:
                d_o = float(self._nn_stats.observed_mean_distance)
            if not np.isfinite(d_o) or d_o <= 0.0:
                d_o = 1.0
            choices = lag_size_choices(d_o)
            labels = (
                QCoreApplication.translate("Tab 2", "Avg D / 3"),
                QCoreApplication.translate("Tab 2", "Avg D / 2"),
                QCoreApplication.translate("Tab 2", "Avg D"),
                QCoreApplication.translate("Tab 2", "1.5 × Avg D"),
                QCoreApplication.translate("Tab 2", "2 × Avg D"),
            )
            for label, value in zip(labels, choices):
                combo.addItem(label, float(value))
            if select_lag is not None:
                best = min(
                    range(combo.count()),
                    key=lambda i: abs(
                        float(combo.itemData(i)) - float(select_lag)
                    ),
                )
                combo.setCurrentIndex(best)
        finally:
            combo.blockSignals(False)

    def _update_experimental_variogram_hint(self):
        """Show Avg D, lag size and max_dist under the variogram plot."""
        if not hasattr(self, 'var_exp_hint_label'):
            return
        d_o = None
        if self._nn_stats is not None:
            d_o = float(self._nn_stats.observed_mean_distance)
        settings = self._experimental_variogram_settings()
        lag_size = float(settings.lag_size)
        max_dist = None
        widget = getattr(self, 'variogram_widget', None)
        if widget is not None and getattr(widget, 'current_data', None):
            max_dist = widget.current_data.get('max_dist')
        if max_dist is None:
            coords = self._layer_coordinates_for_ann()
            if coords is not None:
                max_dist = self._max_dist_from_coordinates(coords)
        if d_o is None and max_dist is None and not np.isfinite(lag_size):
            self.var_exp_hint_label.setText("")
            return
        parts = []
        if d_o is not None and np.isfinite(d_o):
            parts.append(
                QCoreApplication.translate(
                    "Tab 2",
                    "Average distance between neighbor points (Avg D) = {d_o:.4g}",
                ).format(d_o=d_o)
            )
        if np.isfinite(lag_size) and lag_size > 0.0:
            parts.append(
                QCoreApplication.translate(
                    "Tab 2",
                    "Lag size = {lag_size:.4g}",
                ).format(lag_size=lag_size)
            )
        if max_dist is not None and np.isfinite(max_dist):
            parts.append(
                QCoreApplication.translate(
                    "Tab 2",
                    "Max distance in the data = {max_dist:,.0f}",
                ).format(max_dist=max_dist)
            )
        self.var_exp_hint_label.setText("  |  ".join(parts))

    def _recompute_n_bins_for_current_cutoff(self):
        """Derive n_bins from lag_size * n < cutoff."""
        settings = self._experimental_variogram_settings()
        widget = getattr(self, 'variogram_widget', None)
        max_dist = 1.0
        if widget is not None and getattr(widget, 'current_data', None):
            max_dist = float(widget.current_data.get('max_dist', 1.0))
        else:
            coords = self._layer_coordinates_for_ann()
            if coords is not None:
                max_dist = self._max_dist_from_coordinates(coords)
        cutoff = experimental_variogram_cutoff(
            max_dist, settings.factor_max_dist
        )
        settings.n_bins = compute_n_bins(settings.lag_size, cutoff)
        self._set_experimental_variogram_settings_store(settings)
        return settings

    def _on_variogram_lag_size_changed(self, _index=None):
        if getattr(self, '_syncing_experimental_variogram_controls', False):
            return
        lag = self.var_lag_size_combo.currentData()
        if lag is None:
            return
        settings = self._experimental_variogram_settings()
        if abs(float(settings.lag_size) - float(lag)) < 1e-12:
            return
        settings.lag_size = float(lag)
        self._set_experimental_variogram_settings_store(settings)
        self._recompute_n_bins_for_current_cutoff()
        self._update_experimental_variogram_hint()
        self._on_experimental_variogram_settings_committed()

    def _on_experimental_variogram_settings_committed(self):
        """Re-autofit after experimental lag window / binning changes."""
        widget = getattr(self, 'variogram_widget', None)
        if widget is None or getattr(widget, 'current_data', None) is None:
            self._update_experimental_variogram_hint()
            return
        widget.auto_fit()
        self._update_experimental_variogram_hint()

    def _set_experimental_factor_max_dist(self, factor_max_dist):
        """Update cutoff factor from the on-graph handle; skip if unchanged."""
        settings = self._experimental_variogram_settings()
        new_factor = clamp_variogram_factor_max_dist(factor_max_dist)
        if abs(float(settings.factor_max_dist) - float(new_factor)) < 1e-12:
            return False
        settings.factor_max_dist = new_factor
        self._set_experimental_variogram_settings_store(settings)
        self._recompute_n_bins_for_current_cutoff()
        self._sync_experimental_variogram_controls_from_settings()
        self._on_experimental_variogram_settings_committed()
        return True

    def _get_coordinates_and_transformed_values_for_analysis(self):
        """
        Returns coordinates, transformed values and point IDs for tab 2 analysis.

        Prefers stored statistics; otherwise applies the active parameter transform.
        """
        attr_name = self._current_analysis_attribute()
        if not attr_name:
            return None, None, None

        stored = self.stats_by_attribute.get(attr_name)
        if stored is not None:
            return (
                stored['coordinates'],
                stored['values'],
                stored['point_ids'],
            )

        layer_data = self._layer_data_by_attribute.get(attr_name)
        if layer_data is None:
            layer_data = self._extract_layer_data_for_attribute(attr_name)
        if not layer_data or layer_data.get('error'):
            return None, None, None

        transform = (
            'log' if self._get_attribute_log_transform(attr_name) else 'none'
        )
        coordinates, values, _, error = align_coordinates_with_transform(
            layer_data['coordinates'],
            layer_data['raw_values'],
            transform,
        )
        if error or coordinates is None:
            return None, None, None

        point_ids = align_point_ids_with_transform(
            layer_data['point_ids'],
            layer_data['raw_values'],
            transform,
        )
        return coordinates, values, point_ids

    def _get_variogram_state(self, attr_name):
        return self.variogram_models_by_attribute.get(attr_name)

    def _set_variogram_state(self, attr_name, state):
        self.variogram_models_by_attribute[attr_name] = state

    def _apply_store_to_widget(self, attr_name):
        """Redraw the plot from stored parameters (no parallel spin UI)."""
        state = self._get_variogram_state(attr_name)
        if not state or not hasattr(self, 'variogram_widget'):
            return
        widget_data = getattr(self.variogram_widget, 'current_data', None)
        if not widget_data or widget_data.get('attribute') != attr_name:
            return
        self._syncing_variogram = True
        try:
            self.variogram_widget.update_plot(emit_signal=False, params=state)
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
        set_cell(VAR_COL_NUGGET, f"{state.get('nugget', 0):.3g}", editable)
        set_cell(VAR_COL_SILL, f"{state.get('sill', 0):.3g}", editable)
        set_cell(VAR_COL_RANGE, f"{state.get('range', 0):.2g}", editable)

    def _sync_var_params_table_from_store(self, attributes=None):
        """Refresh var_params_table for the active tab-2 parameter only."""
        if not hasattr(self, 'var_params_table'):
            return
        del attributes  # kept for call-site compatibility; table is active-param only

        attr_name = self._current_analysis_attribute()
        state = self._get_variogram_state(attr_name) if attr_name else None

        self._syncing_variogram = True
        self.var_params_table.blockSignals(True)
        try:
            if not attr_name or not state:
                self.var_params_table.setRowCount(0)
            else:
                self.var_params_table.setRowCount(1)
                self._write_var_params_table_row(0, attr_name, state)
            self.var_params_table.resizeColumnsToContents()
        finally:
            self.var_params_table.blockSignals(False)
            self._syncing_variogram = False
        self._refresh_variogram_autofit_status(attr_name)
    def _on_variogram_widget_params_changed(self, params):
        """Save fitted/adjusted widget parameters and sync the params table."""
        if self._syncing_variogram:
            return

        # Prefer the attribute loaded in the widget so batch fitting from Tab 1
        # stores each fit under the correct parameter, not only the active one.
        widget_data = getattr(self.variogram_widget, 'current_data', None) or {}
        attr = widget_data.get('attribute') or self._current_analysis_attribute()
        if not attr:
            return

        log_transform = self._get_attribute_log_transform(attr)
        existing = self._get_variogram_state(attr) or {}
        values = widget_data.get('values')

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
            'r2': float(params.get('r2', 0) or 0),
            'weight': float(existing.get('weight', self._equal_parameter_weight())),
            'log_transform': log_transform,
            'data_fingerprint': fingerprint,
            'autofit_ok': params.get(
                'autofit_ok', existing.get('autofit_ok')
            ),
        }
        self._set_variogram_state(attr, state)

        # During Next-from-Tab-1 batch fitting, only persist the store; views
        # and CV are refreshed once after all parameters are processed.
        if getattr(self, '_batch_fitting_variograms', False):
            return

        self._sync_var_params_table_from_store()
        self._refresh_variogram_autofit_status(attr)
        self._refresh_tab2_cross_validation(attr)
        # Auto-fit / widget edits change the covariance model used by Optimize.
        self._invalidate_optimization_after_variogram_change(attr)

    def _update_store_from_table_cell(self, attr_name, col, text):
        """Update variogram_models_by_attribute from an edited table cell."""
        state = dict(self._get_variogram_state(attr_name))
        if col == VAR_COL_NUGGET:
            state['nugget'] = float(text)
        elif col == VAR_COL_SILL:
            state['sill'] = float(text)
        elif col == VAR_COL_RANGE:
            state['range'] = float(text)
        else:
            return
        self._set_variogram_state(attr_name, state)

    def on_params_table_changed(self, item):
        """Sync manual table edits back to store, plot, and experimental-theoretical R²."""
        if self._syncing_variogram:
            return

        row = item.row()
        col = item.column()
        if col in (
            VAR_COL_PARAM,
            VAR_COL_TRANSFORM,
            VAR_COL_MODEL,
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
                state = self._get_variogram_state(attribute)
                r2 = self.variogram_widget.update_plot(
                    emit_signal=False, params=state
                )
                state = dict(state or {})
                state['r2'] = float(r2)
                self._set_variogram_state(attribute, state)
                if col in (VAR_COL_NUGGET, VAR_COL_SILL, VAR_COL_RANGE):
                    self._refresh_tab2_cross_validation(attribute)
            if col in (VAR_COL_NUGGET, VAR_COL_SILL, VAR_COL_RANGE):
                self._invalidate_optimization_after_variogram_change(attribute)
        except (ValueError, TypeError):
            pass
        finally:
            self._syncing_variogram = False

    def on_attribute_changed(self):
        """Handles the selection of attributes in the data selection tab (supports multi-selection)."""
        self._sync_stats_table_structure()
        self._clear_variogram_widget()
        self._update_tab1_progress_hint()
        if not self.selected_data_parameters.selectedItems():
            self.clear_stats_table()
            return
        attr = self._current_analysis_attribute()
        if attr and attr in self.stats_by_attribute:
            self._refresh_tab2_parameter_views(attr)
        else:
            self._clear_stats_plots()
            self._clear_cross_validation()

    def clear_stats_table(self):
        """Clears statistics table, stored stats and resets the info label."""
        self.stats_by_attribute.clear()
        self._layer_data_by_attribute.clear()
        parameters = self._selected_analysis_parameters()
        self.stats_table.setRowCount(len(parameters))
        if parameters:
            self.stats_table.setVerticalHeaderLabels(parameters)
            self.stats_table.verticalHeader().setVisible(True)
        read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        for row in range(self.stats_table.rowCount()):
            attr_name = (
                parameters[row] if row < len(parameters) else None
            )
            if attr_name:
                self._set_stats_table_weight_cell(row, attr_name)
            else:
                weight_item = self.stats_table.item(row, STATS_COL_WEIGHT)
                default_weight = self._format_parameter_weight(
                    self._equal_parameter_weight()
                )
                if weight_item is None:
                    weight_item = QTableWidgetItem(default_weight)
                    self.stats_table.setItem(
                        row, STATS_COL_WEIGHT, weight_item
                    )
                else:
                    weight_item.setText(default_weight)
            self.stats_table.removeCellWidget(row, STATS_TRANSFORM_COL)
            if attr_name:
                self._set_stats_table_transform_cell(row, attr_name)
            for col in range(
                STATS_VALUE_COL_OFFSET, self.stats_table.columnCount()
            ):
                item = self.stats_table.item(row, col)
                if item is None:
                    item = QTableWidgetItem("—")
                    self.stats_table.setItem(row, col, item)
                else:
                    item.setText("—")
                item.setFlags(read_only)
                item.setToolTip("")
                item.setBackground(QBrush())
        self.stats_info_label.setText(
            QCoreApplication.translate(
                "Tab 2",
                "Select attributes on tab 1, then click Calculate geostatistics.",
            )
        )
        self.stats_info_label.setStyleSheet("color: gray; font-style: italic;")
        self._adjust_stats_table_height()
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

        self._disconnect_point_hover(getattr(self, 'cv_canvas', None))
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

    def _update_cross_validation_plot(
        self, measured, predicted, attr_name, point_ids=None
    ):
        """Measured vs predicted scatter with 1:1 and linear-regression lines."""
        measured = np.asarray(measured, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        self._disconnect_point_hover(getattr(self, 'cv_canvas', None))
        ax = self.cv_ax
        ax.clear()

        ax.scatter(
            measured,
            predicted,
            alpha=0.7,
            color='#4a90d9',
            edgecolors='k',
            linewidths=0.5,
            label=QCoreApplication.translate("Tab 2", "Parameter values"),
        )

        # 1:1 line
        if measured.size > 0:
            lo = float(min(measured.min(), predicted.min()))
            hi = float(max(measured.max(), predicted.max()))
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            ax.plot(
                [lo, hi],
                [lo, hi],
                linestyle='-',
                color='gray',
                linewidth=0.5,
                label=QCoreApplication.translate("Tab 2", "1:1 line"),
            )

            # Ordinary least-squares fit: predicted ≈ slope * measured + intercept.
            finite = np.isfinite(measured) & np.isfinite(predicted)
            if np.count_nonzero(finite) >= 2:
                slope, intercept = np.polyfit(
                    measured[finite], predicted[finite], 1
                )
                x_fit = np.array([lo, hi], dtype=float)
                ax.plot(
                    x_fit,
                    slope * x_fit + intercept,
                    color='gray',
                    linewidth=1.5,
                    solid_capstyle='round',
                    zorder=1,
                    linestyle='dotted',
                    label=QCoreApplication.translate("Tab 2", "Linear-regression line"),
                )
                # Equation text 
                if intercept >= 0:
                    eq_text = "y = {:.4g}x + {:.4g}".format(slope, intercept)
                else:
                    eq_text = "y = {:.4g}x - {:.4g}".format(
                        slope, abs(intercept)
                    )
                ax.text(
                    0.98,
                    0.02,
                    eq_text,
                    transform=ax.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='gray',
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

        if point_ids is not None and len(point_ids) == measured.size:
            hover_xy = np.column_stack([measured, predicted])
            id_labels = [str(pid) for pid in point_ids]
            self._attach_point_hover(
                self.cv_canvas, ax, hover_xy, id_labels
            )

        self.cv_canvas.draw()

    def _reset_stats_map_axes(self):
        """Recreates the map axes (avoids colorbar.remove() issues on refresh)."""
        self.stats_map_figure.clear()
        self.stats_map_ax = self.stats_map_figure.add_subplot(111)
        self.stats_map_colorbar = None

    def _clear_stats_plots(self):
        """Clears histogram and spatial map placeholders."""
        self._disconnect_point_hover(getattr(self, 'stats_hist_canvas', None))
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

        self._disconnect_point_hover(getattr(self, 'stats_map_canvas', None))
        self._reset_stats_map_axes()
        self.stats_map_ax.text(
            0.5, 0.5, no_data,
            ha='center', va='center',
            transform=self.stats_map_ax.transAxes, color='gray',
        )
        self.stats_map_ax.set_xticks([])
        self.stats_map_ax.set_yticks([])
        self.stats_map_canvas.draw()

    def _update_stats_plots(self, coordinates, values, attr_name, point_ids=None):
        """Updates histogram and spatial scatter for the statistics data."""
        values = np.asarray(values, dtype=float)
        coordinates = np.asarray(coordinates, dtype=float)
        id_labels = None
        if point_ids is not None and len(point_ids) == values.size:
            id_labels = [str(pid) for pid in point_ids]

        self._disconnect_point_hover(getattr(self, 'stats_hist_canvas', None))
        ax = self.stats_hist_ax
        ax.clear()

        ax.hist(
            values, bins='auto', color='#4a90d9', edgecolor='white', alpha=0.85,
        )

        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        # Sample std (ddof=1) matches Tab 2 descriptive statistics.
        std_val = (
            float(np.std(values, ddof=1)) if values.size >= 2 else float('nan')
        )

        ax.axvline(
            mean_val, color='#c0392b', linestyle='--', linewidth=1.5,
            label=QCoreApplication.translate(
                "Tab 2", "Mean"
            ).format(value=mean_val),
        )
        ax.axvline(
            median_val, color='#d35400', linestyle='-', linewidth=1.5,
            label=QCoreApplication.translate(
                "Tab 2", "Median"
            ).format(value=median_val),
        )

        # Vertical guides at mean ± 1σ and mean ± 2σ (skip if std is invalid).
        if np.isfinite(std_val) and std_val > 0.0:
            std_styles = (
                (
                    1,
                    'gray',
                    (0, (1, 1)),
                    QCoreApplication.translate(
                        "Tab 2", "±1σ"
                    ).format(std=std_val),
                ),
                (
                    2,
                    'gray',
                    'dotted',
                    QCoreApplication.translate(
                        "Tab 2", "±2σ"
                    ).format(std=2.0 * std_val),
                ),
            )
            for k, color, linestyle, label in std_styles:
                # Label only the +kσ line so the legend lists each band once.
                ax.axvline(
                    mean_val - k * std_val,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                )
                ax.axvline(
                    mean_val + k * std_val,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                    label=label,
                )

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax > ymin else 1.0
        rug_base = ymin + y_range * 0.02
        rug_span = y_range * 0.06
        jitter = np.random.default_rng(0).uniform(0, rug_span, size=values.size)
        rug_y = rug_base + jitter
        ax.scatter(
            values,
            rug_y,
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

        # Hover only on histogram data points (rug), not bars or guide lines.
        if id_labels is not None:
            rug_xy = np.column_stack([values, rug_y])
            self._attach_point_hover(
                self.stats_hist_canvas, ax, rug_xy, id_labels
            )
        self.stats_hist_canvas.draw()

        self._disconnect_point_hover(getattr(self, 'stats_map_canvas', None))
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

        if id_labels is not None and id_labels and coordinates.shape[0] == len(id_labels):
            self._attach_point_hover(
                self.stats_map_canvas,
                self.stats_map_ax,
                coordinates,
                id_labels,
            )
        self.stats_map_canvas.draw()

    def _populate_stats_table(self, stats_dict):
        """Fills the active parameter's statistics row from a stats dictionary."""
        attr_name = self._current_analysis_attribute()
        if not attr_name:
            return
        row = self._stats_table_row_for_attribute(attr_name)
        if row < 0:
            return
        for col, key in enumerate(STAT_KEYS):
            table_col = col + STATS_VALUE_COL_OFFSET
            value = stats_dict[key]
            if key == 'count':
                text = str(int(value))
            elif np.isnan(value):
                text = "—"
            else:
                text = f"{value:.3g}"
            item = self.stats_table.item(row, table_col)
            if item is None:
                item = QTableWidgetItem(text)
                self.stats_table.setItem(row, table_col, item)
            else:
                item.setText(text)
            apply_distribution_shape_cell_style(item, key, value)

    def refresh_attribute_statistics(self, force_reextract=False):
        """Legacy entry point: recalculate all selected parameters."""
        self._calculate_all_attribute_statistics(force_reextract=force_reextract)

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

    def _find_params_table_row(self, attr_name):
        """Return row 0 when the variogram table shows the given parameter."""
        if self.var_params_table.rowCount() != 1:
            return -1
        item = self.var_params_table.item(0, VAR_COL_PARAM)
        if item and item.text() == attr_name:
            return 0
        return -1

    def _set_params_table_cell(self, row, col, text):
        item = self.var_params_table.item(row, col)
        if item is None:
            self.var_params_table.setItem(row, col, QTableWidgetItem(str(text)))
        else:
            item.setText(str(text))

    def _default_node_spacing_for_layer(self, layer=None):
        """
        Default Tab 3 node spacing from ANN observed mean distance (D_o).

        Previously used (max_dist / 2) / 10 from pairwise distances; that
        formula is kept commented below for reference. Falls back to 100 when
        ANN is unavailable.
        """
        if self._nn_stats is not None:
            d_o = float(self._nn_stats.observed_mean_distance)
            if np.isfinite(d_o) and d_o > 0.0:
                return d_o

        # Fallback: try ANN from the layer if cache is empty.
        if layer is None and hasattr(self, 'input_data_layer'):
            layer = self.input_data_layer.currentLayer()
        if layer is not None:
            coords = self._layer_coordinates_for_ann(layer)
            if coords is not None:
                nn_stats = compute_nearest_neighbor_stats(coords)
                if nn_stats is not None:
                    d_o = float(nn_stats.observed_mean_distance)
                    if np.isfinite(d_o) and d_o > 0.0:
                        return d_o

        # Previous default (commented out):
        # if layer is None and hasattr(self, 'input_data_layer'):
        #     layer = self.input_data_layer.currentLayer()
        # if layer is None:
        #     return 100.0
        # try:
        #     coords = []
        #     for feature in layer.getFeatures():
        #         geom = feature.geometry()
        #         if geom is None or geom.isEmpty() or geom.isMultipart():
        #             continue
        #         point = geom.asPoint()
        #         coords.append([point.x(), point.y()])
        #     if len(coords) < 2:
        #         return 100.0
        #     from scipy.spatial.distance import pdist
        #     max_dist = float(np.max(pdist(np.asarray(coords))))
        #     if not np.isfinite(max_dist) or max_dist <= 0:
        #         return 100.0
        #     return (max_dist / 2.0) / 10.0
        # except Exception:
        #     return 100.0
        return 100.0

    def _apply_default_node_spacing(self, layer=None):
        """Set Tab 3 spacing spinbox to D_o (ANN) silently."""
        if not hasattr(self, 'spacing_spin'):
            return
        spacing = self._default_node_spacing_for_layer(layer)
        self.spacing_spin.blockSignals(True)
        try:
            self.spacing_spin.setValue(spacing)
        finally:
            self.spacing_spin.blockSignals(False)

    def _grid_buffer_spacing_factor(self):
        """Buffer setting as a multiple of node spacing (default 1.0)."""
        if not hasattr(self, 'buffer_spin'):
            return 1.0
        return float(self.buffer_spin.value())

    def _effective_grid_buffer_map_units(self):
        """
        Buffer distance applied outside the concave hull, in map units.

        Computed as (buffer × node spacing). For example, factor 1.0 with
        spacing 100 m yields a 100 m outward buffer on the hull.
        """
        if not hasattr(self, 'spacing_spin'):
            return 0.0
        spacing = float(self.spacing_spin.value())
        return self._grid_buffer_spacing_factor() * spacing

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
            buffer = self._effective_grid_buffer_map_units()
            
            if spacing > 0 and width > 0 and height > 0:
                # Rough extent estimate: layer bbox expanded by the hull buffer.
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

        The buffer is applied as a multiple of node spacing (see tab 3 control).
        Uses the QGIS concave hull algorithm (user-defined ALPHA, no holes).
        """
        layer = self.input_data_layer.currentLayer()
        if not layer:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate("Tab 3", "Select a layer first."),
            )
            return

        self._set_progress(
            5,
            QCoreApplication.translate(
                "Tab 3", "State: Generating estimation grid..."
            ),
        )
        try:
            import numpy as np
            from shapely.geometry import Point, Polygon
            from shapely.ops import unary_union

            buffer_factor = self._grid_buffer_spacing_factor()
            buffer = self._effective_grid_buffer_map_units()
            spacing = self.spacing_spin.value()

            # Input well/point coordinates
            self._set_progress(
                15,
                QCoreApplication.translate(
                    "Tab 3", "State: Reading well coordinates..."
                ),
            )
            data_points = extract_layer_coordinates(
                layer, include_fids=self._included_well_fids_for_extract()
            )
            if data_points is None or len(data_points) == 0:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3", "The selected layer has no points."
                    ),
                )
                self._reset_progress()
                return

            # Build hull using the QGIS concave hull algorithm
            self._set_progress(
                35,
                QCoreApplication.translate(
                    "Tab 3", "State: Building boundary hull..."
                ),
            )
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

                # Run concave hull with user ALPHA, without holes
                alg_params = {
                    'INPUT': point_layer,
                    'ALPHA': float(self.alpha_spin.value()),
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
            self._set_progress(
                60,
                QCoreApplication.translate(
                    "Tab 3", "State: Filling grid nodes..."
                ),
            )
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
                'buffer_spacing_factor': buffer_factor,
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
                    self._clear_all_optimization_results(
                        tab5_message=QCoreApplication.translate(
                            "Tab 5",
                            "The estimation grid changed. Re-run optimization on tab 4.",
                        ),
                        tab4_message=QCoreApplication.translate(
                            "Tab 4",
                            "The estimation grid changed. Click Optimize again.",
                        ),
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
                color='#e0d9d8',
                #linestyle='-',
                linewidth=0.5,
                label=QCoreApplication.translate(
                    "Tab 3", "Buffered hull"
                ),
            )

            # Estimation grid nodes
            if n_nodes > 0:
                ax.scatter(
                    estimation_points[:, 0], estimation_points[:, 1],
                    c='gray', #marker='o',
                    label=QCoreApplication.translate("Tab 3", "Estimation grid"),
                    s=30, alpha=0.5, edgecolors='black', linewidths=0.3
                )

            ax.set_xlabel(QCoreApplication.translate("Tab 3", "X"))
            ax.set_ylabel(QCoreApplication.translate("Tab 3", "Y"))
            ax.set_title(
                QCoreApplication.translate(
                    "Tab 3", "Estimation Grid on QGIS Concave Hull"
                )
            )
            ax.legend(loc='best', bbox_to_anchor=(1, 0.5), prop={'size': 8})
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            self.grid_canvas.draw()

            # Keep grid parameters for export
            self.grid_type_saved = 'Generated'
            self.n_nodes_saved = n_nodes
            self.buffer_saved = buffer
            self.buffer_spacing_factor_saved = buffer_factor

            self._update_mn_grid_weight_info()
            self._set_progress(
                100,
                QCoreApplication.translate("Tab 3", "State: Completed"),
            )
            QTimer.singleShot(1500, self._reset_progress)
            self._update_workflow_navigation_state()

        except Exception as e:
            import traceback
            self._reset_progress(
                QCoreApplication.translate("Tab 3", "State: Error")
            )
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
            if grid_info is not None:
                buffer = float(grid_info.get(
                    'buffer', self._effective_grid_buffer_map_units()
                ))
            else:
                buffer = self._effective_grid_buffer_map_units()
            
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
    
    def _draw_imported_estimation_grid(self, xs, ys, weights=None):
        """
        Draws an imported estimation grid on tab 3.

        When ``weights`` is provided, each unique weight gets a discrete legend
        entry. Without weights, nodes are drawn with a single uniform style.
        """
        self.grid_figure.clear()
        ax = self.grid_figure.add_subplot(111)

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if weights is not None:
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
        else:
            ax.scatter(
                xs,
                ys,
                c='#4a90d9',
                s=30,
                alpha=0.9,
                edgecolors='k',
                linewidths=0.3,
                zorder=3,
            )

        layer = (
            self.input_data_layer.currentLayer()
            if hasattr(self, 'input_data_layer')
            else None
        )
        if layer is not None:
            well_coords = extract_layer_coordinates(
                layer, include_fids=self._included_well_fids_for_extract()
            )
            self._plot_wells_on_map_axes(ax, well_coords)

        ax.set_title(
            QCoreApplication.translate("Tab 3", "Imported Estimation Grid")
        )
        ax.set_xlabel(QCoreApplication.translate("Tab 3", "X"))
        ax.set_ylabel(QCoreApplication.translate("Tab 3", "Y"))
        if weights is not None:
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
        Load estimation grid from an .xlsx file and plot on tab 3.

        Uses the first columns in order: ID, X, Y, and optionally weight.
        Rows with invalid ID or coordinates are skipped. Weight is optional;
        when absent or empty for all rows, the grid is imported as unweighted.
        """
        try:
            file_path = self._prompt_open_xlsx(
                QCoreApplication.translate(
                    "Tab 3", "Select Estimation Grid Excel File"
                )
            )
            if not file_path:
                return

            self._set_progress(
                10,
                QCoreApplication.translate(
                    "Tab 3", "State: Importing estimation grid..."
                ),
            )
            try:
                table_rows = read_excel_table_rows(file_path)
            except ImportError as exc:
                self._reset_progress()
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    str(exc),
                )
                return
            except Exception as e:
                self._reset_progress(
                    QCoreApplication.translate("Tab 3", "State: Error")
                )
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3", "Could not read Excel file:\n{error}"
                    ).format(error=str(e)),
                )
                return

            if not table_rows:
                self._reset_progress()
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3", "The selected Excel file is empty."
                    ),
                )
                return

            if len(table_rows[0]) < 3:
                self._reset_progress()
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Invalid File"),
                    QCoreApplication.translate(
                        "Tab 3",
                        "The selected file must have at least 3 columns for "
                        "ID, X, and Y (in that order). Weight is optional.",
                    ),
                )
                return

            self._set_progress(
                50,
                QCoreApplication.translate(
                    "Tab 3", "State: Parsing grid points..."
                ),
            )
            ids = []
            xs = []
            ys = []
            row_weights = []
            for row in table_rows[1:]:
                if row is None or len(row) < 3:
                    continue
                id_val, x_val, y_val = row[0], row[1], row[2]
                if id_val is None or x_val is None or y_val is None:
                    continue
                try:
                    x_f = float(x_val)
                    y_f = float(y_val)
                except (ValueError, TypeError):
                    continue
                if np.isnan(x_f) or np.isnan(y_f):
                    continue

                weight_val = None
                if len(row) >= 4 and row[3] is not None:
                    try:
                        w_f = float(row[3])
                        if not np.isnan(w_f):
                            weight_val = w_f
                    except (ValueError, TypeError):
                        pass

                ids.append(id_val)
                xs.append(x_f)
                ys.append(y_f)
                row_weights.append(weight_val)

            if not ids:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 3", "Error"),
                    QCoreApplication.translate(
                        "Tab 3",
                        "No valid records with ID, X, and Y found in the file.",
                    ),
                )
                return

            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            grid_points = np.column_stack([xs, ys])
            n_nodes = grid_points.shape[0]

            has_weights = any(weight is not None for weight in row_weights)
            weights = None
            if has_weights:
                weights = np.asarray(
                    [weight if weight is not None else 1.0 for weight in row_weights],
                    dtype=float,
                )

            # Save grid information for workflow integration
            self.current_grid_info = {
                'grid_points': grid_points,
                'n_nodes': n_nodes,
                'ids': np.asarray(ids),
                'grid_type': "imported_from_xlsx",
                'buffer': 0.0,
                'n_x': None,  # unknown for imported grid
                'n_y': None,
            }
            if has_weights:
                self.current_grid_info['weights'] = weights
                self.current_ceg_values = weights
            else:
                self.current_ceg_values = None

            self.current_grid_points = grid_points

            self._update_mn_grid_weight_info()

            # Plot on Tab 3 (Estimation Grid Visualization)
            self._draw_imported_estimation_grid(xs, ys, weights)

            if has_weights:
                success_message = QCoreApplication.translate(
                    "Tab 3",
                    "Grid loaded successfully with {count} points "
                    "(columns: ID, X, Y, weight).",
                ).format(count=n_nodes)
            else:
                success_message = QCoreApplication.translate(
                    "Tab 3",
                    "Grid loaded successfully with {count} points "
                    "(columns: ID, X, Y; no node weights).",
                ).format(count=n_nodes)

            self._set_progress(
                100,
                QCoreApplication.translate("Tab 3", "State: Completed"),
            )
            QTimer.singleShot(1500, self._reset_progress)
            self._update_workflow_navigation_state()
            QMessageBox.information(
                self,
                QCoreApplication.translate("Tab 3", "Success"),
                success_message,
            )
        except Exception as e:
            import traceback
            self._reset_progress(
                QCoreApplication.translate("Tab 3", "State: Error")
            )
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 3", "Error"),
                QCoreApplication.translate(
                    "Tab 3",
                    "Failed to load grid:\n{error}\n{details}",
                ).format(error=str(e), details=traceback.format_exc()),
            )

class VariogramWidget(QWidget):
    """Variogram plot and auto-fit; parameters live in the dialog store / table.

    Model params are owned by ``var_params_table`` / store. Experimental lag
    settings come from the dialog session. A red dashed cutoff handle adjusts
    ``factor_max_dist`` interactively.
    """

    params_changed = pyqtSignal(dict)

    def __init__(self, parent=None, dialog=None):
        super().__init__(parent)
        self.dialog = dialog
        self.current_data = None
        self.current_model = None
        self._drag_cids = []
        self._drag_mode = None  # 'cutoff' only
        self._drag_max_dist = None
        self._drag_cutoff = None
        self._drag_factor_before = None
        self._vario_ax = None
        self._cutoff_line = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        # Bottom margin reserved for legend outside the axes.
        self.figure = Figure(figsize=(8, 4.0))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMaximumHeight(300)
        self.canvas.setMinimumHeight(180)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def _session_settings(self):
        if self.dialog is not None and hasattr(
            self.dialog, '_experimental_variogram_settings'
        ):
            return self.dialog._experimental_variogram_settings()
        return ExperimentalVariogramSettings()

    def _resolve_params(self, params=None):
        """Normalize model parameters from an explicit dict or the dialog store."""
        if params is None and self.dialog is not None and self.current_data:
            attr = self.current_data.get('attribute')
            if attr:
                params = self.dialog._get_variogram_state(attr)

        if not params:
            return {
                'model': 'spherical',
                'nugget': 0.0,
                'sill': 0.0,
                'range': 0.0,
            }

        model_type = params.get('model_type', params.get('model', 'spherical'))
        return {
            'model': model_type if model_type in VARIOGRAM_MODEL_TYPES else 'spherical',
            'nugget': float(params.get('nugget', 0) or 0),
            'sill': float(params.get('sill', 0) or 0),
            'range': float(params.get('range', 0) or 0),
        }

    def _experimental_variogram(self):
        """Experimental lag bins from session settings (lag_size spacing)."""
        if self.current_data is None:
            return None, None

        settings = self._session_settings()
        coordinates = self.current_data['coordinates']
        values = self.current_data['values']
        max_dist = self.current_data['max_dist']
        bin_edges = experimental_variogram_bin_edges(max_dist, settings)
        bin_center, gamma = gs.vario_estimate(
            coordinates.T, values, bin_edges=bin_edges
        )
        valid_mask = (
            np.isfinite(bin_center) & np.isfinite(gamma) & (gamma >= 0)
        )
        if np.sum(valid_mask) < 3:
            return None, None
        return bin_center[valid_mask], gamma[valid_mask]

    def _build_model_from_params(self, params):
        """Build a gstools covariance model from resolved parameter dict."""
        return build_covariance_model(
            params['model'],
            params['nugget'],
            params['sill'],
            params['range'],
        )

    def _disconnect_drag(self):
        for cid in self._drag_cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._drag_cids = []
        self._drag_mode = None
        self._cutoff_line = None
        self._vario_ax = None
        self._drag_max_dist = None
        self._drag_cutoff = None
        self._drag_factor_before = None

    def _cutoff_display_point(self):
        if self._vario_ax is None or self._drag_cutoff is None:
            return None
        ylim = self._vario_ax.get_ylim()
        y_mid = 0.5 * (float(ylim[0]) + float(ylim[1]))
        return self._vario_ax.transData.transform(
            (float(self._drag_cutoff), y_mid)
        )

    def _hit_test_cutoff(self, event, pixel_tol=12.0):
        if event.x is None or event.y is None:
            return False
        point = self._cutoff_display_point()
        if point is None:
            return False
        dx = float(event.x) - float(point[0])
        dy = float(event.y) - float(point[1])
        return (dx * dx + dy * dy) <= (pixel_tol * pixel_tol)

    def _on_drag_press(self, event):
        if event.button != 1 or event.inaxes is not self._vario_ax:
            return
        if not self._hit_test_cutoff(event):
            return
        self._drag_mode = 'cutoff'
        if self.dialog is not None:
            settings = self.dialog._experimental_variogram_settings()
            self._drag_factor_before = float(settings.factor_max_dist)

    def _on_drag_motion(self, event):
        if self._drag_mode is None:
            if self._hit_test_cutoff(event):
                self.canvas.setCursor(Qt.SizeHorCursor)
            else:
                self.canvas.unsetCursor()
            return

        if self._drag_max_dist is None or event.xdata is None:
            return

        max_dist = float(self._drag_max_dist)
        # factor in [1.5, 5] → cutoff x in [max_dist/5, max_dist/1.5]
        x_min = max_dist / VARIOGRAM_FACTOR_MAX_DIST_MAX
        x_max = max_dist / VARIOGRAM_FACTOR_MAX_DIST_MIN
        x = max(x_min, min(float(event.xdata), x_max))
        self._drag_cutoff = x
        if self._cutoff_line is not None:
            self._cutoff_line.set_xdata([x, x])
        self.canvas.draw_idle()

    def _on_drag_release(self, event):
        if self._drag_mode is None:
            return
        self._drag_mode = None
        self.canvas.unsetCursor()

        if self._drag_max_dist is None or self._drag_cutoff is None:
            return
        max_dist = float(self._drag_max_dist)
        cutoff = float(self._drag_cutoff)
        if cutoff <= 0:
            return
        factor = clamp_variogram_factor_max_dist(max_dist / cutoff)
        if (
            self._drag_factor_before is not None
            and abs(factor - float(self._drag_factor_before)) < 1e-12
        ):
            # Snap visual line back if user released without a real change.
            if self.dialog is not None:
                settings = self.dialog._experimental_variogram_settings()
                snapped_cutoff = experimental_variogram_cutoff(
                    max_dist, settings.factor_max_dist
                )
                self._drag_cutoff = snapped_cutoff
                if self._cutoff_line is not None:
                    self._cutoff_line.set_xdata(
                        [snapped_cutoff, snapped_cutoff]
                    )
                self.canvas.draw_idle()
            return

        if self.dialog is not None:
            self.dialog._set_experimental_factor_max_dist(factor)

    def _install_cutoff_handle(self, ax, max_dist, cutoff):
        """Install red dashed cutoff line and mouse callbacks."""
        self._disconnect_drag()
        self._cutoff_line = ax.axvline(
            cutoff,
            color='red',
            linestyle='--',
            linewidth=1.2,
            alpha=0.95,
            zorder=4,
            label=QCoreApplication.translate("Tab 2", "Variogram limit (editable)"),
        )

        self._vario_ax = ax
        self._drag_max_dist = float(max_dist)
        self._drag_cutoff = float(cutoff)
        if self.dialog is not None:
            settings = self.dialog._experimental_variogram_settings()
            self._drag_factor_before = float(settings.factor_max_dist)

        self._drag_cids = [
            self.canvas.mpl_connect('button_press_event', self._on_drag_press),
            self.canvas.mpl_connect('motion_notify_event', self._on_drag_motion),
            self.canvas.mpl_connect(
                'button_release_event', self._on_drag_release
            ),
        ]

    def auto_fit(self):
        """
        Automatically fits the variogram model and emits params including R².

        After GSTools convergence, rejects unstable fits when:
          - len_scale > domain max_dist, or
          - total sill > sample variance.
        On rejection or fit exception, applies stable fallback parameters
        (first-bin nugget, sample-variance sill, max_dist/3 range).
        """
        if self.current_data is None:
            return

        main_dialog = self.dialog
        set_progress = getattr(main_dialog, '_set_variogram_progress', None)

        if set_progress is not None:
            set_progress(0, "State: Starting variogram calculation...")

        try:
            model_type = self._resolve_params()['model']

            if set_progress is not None:
                set_progress(
                    10, "State: Computing experimental variogram..."
                )

            bin_center, gamma = self._experimental_variogram()
            if bin_center is None:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("Tab 2", "Error"),
                    QCoreApplication.translate(
                        "Tab 2",
                        "Not enough valid points in the experimental variogram "
                        "to fit the model. Select different lag size or max distance.",
                    ),
                )
                return

            if set_progress is not None:
                set_progress(30, "State: Computing initial values...")

            max_dist = float(self.current_data.get('max_dist', 1.0))
            values = np.asarray(self.current_data['values'], dtype=float).ravel()
            if values.size >= 2:
                data_variance = float(np.var(values, ddof=1))
            else:
                data_variance = 0.0
            if not np.isfinite(data_variance) or data_variance < 0.0:
                data_variance = 0.0

            max_gamma = float(np.max(gamma))
            max_dist_used = float(np.max(bin_center))
            estimated_nugget = float(gamma[0])
            estimated_sill = max_gamma
            estimated_range = max_dist_used
            first_bin_value = estimated_nugget

            if set_progress is not None:
                set_progress(50, "State: Creating variogram model...")

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

            if set_progress is not None:
                set_progress(70, "State: Setting initial parameters...")

            try:
                model.set_arg(nugget=estimated_nugget)
                model.len_scale = estimated_range
                model.var = max(0.01, estimated_sill - estimated_nugget)
            except Exception:
                try:
                    model.nugget = max(0.0, estimated_nugget)
                    model.len_scale = max(0.01, estimated_range)
                    model.var = max(0.01, estimated_sill - estimated_nugget)
                except Exception:
                    pass

            if set_progress is not None:
                set_progress(80, "State: Fitting variogram model...")

            autofit_ok = False
            use_fallback = False
            try:
                try:
                    model.fit_variogram(
                        bin_center, gamma, return_r2=True, max_eval=5000
                    )
                except TypeError:
                    model.fit_variogram(bin_center, gamma, return_r2=True)

                fitted_sill = float(max(0.0, model.var + model.nugget))
                fitted_range = float(max(0.01, model.len_scale))
                ok, _reasons = validate_variogram_autofit(
                    fitted_range,
                    fitted_sill,
                    max_dist,
                    data_variance,
                )
                if ok:
                    autofit_ok = True
                else:
                    use_fallback = True
            except Exception:
                use_fallback = True

            if use_fallback:
                fallback = variogram_autofit_fallback_params(
                    first_bin_value,
                    data_variance,
                    max_dist,
                )
                try:
                    model.nugget = float(fallback['nugget'])
                    model.len_scale = float(fallback['range'])
                    model.var = max(
                        1e-6,
                        float(fallback['sill']) - float(fallback['nugget']),
                    )
                except Exception:
                    pass
                autofit_ok = False

            if set_progress is not None:
                set_progress(95, "State: Updating controls...")

            r2 = compute_variogram_lag_r2(bin_center, gamma, model)
            self.current_model = model

            params = {
                'model': model_type,
                'nugget': float(max(0.0, model.nugget)),
                'sill': float(max(0.0, model.var + model.nugget)),
                'range': float(max(0.01, model.len_scale)),
                'r2': float(r2),
                'autofit_ok': bool(autofit_ok),
            }
            self.params_changed.emit(params)
            self.update_plot(emit_signal=False, params=params)
            if self.dialog is not None:
                hint = getattr(
                    self.dialog, '_update_experimental_variogram_hint', None
                )
                if hint is not None:
                    hint()

            if set_progress is not None:
                if autofit_ok:
                    set_progress(100, "State: Completed successfully")
                else:
                    set_progress(
                        100,
                        "State: Auto-fit rejected; using fallback parameters",
                    )
                if not getattr(
                    main_dialog, '_suppress_variogram_progress', False
                ):
                    from qgis.PyQt.QtCore import QTimer
                    QTimer.singleShot(
                        2000, lambda: self._hide_progress_elements()
                    )
        except Exception as e:
            if set_progress is not None:
                set_progress(0, "State: Error")
            QMessageBox.warning(
                self,
                QCoreApplication.translate("Tab 2", "Error"),
                QCoreApplication.translate(
                    "Tab 2", "Automatic fitting failed: {error}"
                ).format(error=str(e)),
            )

    def _hide_progress_elements(self):
        """Reset progress bar after variogram auto-fit completes."""
        main_dialog = self.dialog
        if main_dialog is not None and getattr(
            main_dialog, '_suppress_variogram_progress', False
        ):
            return
        reset_progress = getattr(main_dialog, '_reset_variogram_progress', None)
        if reset_progress is not None:
            reset_progress()

    def clear(self):
        """Clears data and plot when input layer/attributes change."""
        self._disconnect_drag()
        self.current_data = None
        self.current_model = None
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

    def set_data(
        self, coordinates, values, attribute_name, log_transform=False, auto_fit=False
    ):
        """Loads point data for the experimental variogram; optionally auto-fits."""
        from scipy.spatial import distance

        coordinates = np.asarray(coordinates, dtype=float)
        values = np.asarray(values, dtype=float).ravel()
        if coordinates.shape[0] < 2:
            self.clear()
            return

        if len(coordinates) > 100:
            x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
            y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
            max_dist = float(np.sqrt(x_range ** 2 + y_range ** 2))
        else:
            max_dist = float(np.max(distance.pdist(coordinates)))
        if not np.isfinite(max_dist) or max_dist <= 0:
            max_dist = 1.0

        self.current_data = {
            'coordinates': coordinates,
            'values': values,
            'attribute': attribute_name,
            'max_dist': max_dist,
            'log_transform': log_transform,
        }
        # Ensure n_bins matches current lag/factor and this max_dist.
        if self.dialog is not None:
            recompute = getattr(
                self.dialog, '_recompute_n_bins_for_current_cutoff', None
            )
            if recompute is not None:
                recompute()
            hint = getattr(
                self.dialog, '_update_experimental_variogram_hint', None
            )
            if hint is not None:
                hint()

        if auto_fit:
            self.auto_fit()

    def update_plot(self, emit_signal=True, params=None):
        """Redraw experimental + theoretical curves and cutoff handle.

        X-axis is fixed to ``[0, max_dist/1.5 + max_dist*0.05]``. Experimental
        bins and the theoretical curve use cutoff = max_dist / factor_max_dist.
        """
        self._disconnect_drag()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        r2 = 0.0

        if self.current_data is not None:
            try:
                max_dist = float(self.current_data['max_dist'])
                settings = self._session_settings()
                cutoff = experimental_variogram_cutoff(
                    max_dist, settings.factor_max_dist
                )
                values = np.asarray(
                    self.current_data['values'], dtype=float
                ).ravel()
                if values.size >= 2:
                    data_variance = float(np.var(values, ddof=1))
                else:
                    data_variance = 0.0

                bin_center, gamma = self._experimental_variogram()
                if bin_center is None:
                    raise ValueError("insufficient experimental variogram bins")

                ax.scatter(
                    bin_center,
                    gamma,
                    marker='+',
                    color='black',
                    label=QCoreApplication.translate(
                        "Tab 2", "Experimental Variogram"
                    ),
                    alpha=0.7,
                    zorder=2,
                )

                resolved = self._resolve_params(params)
                model = self._build_model_from_params(resolved)
                self.current_model = model
                r2 = compute_variogram_lag_r2(bin_center, gamma, model)

                if emit_signal:
                    self.params_changed.emit({
                        'model': resolved['model'],
                        'nugget': resolved['nugget'],
                        'sill': resolved['sill'],
                        'range': resolved['range'],
                        'r2': float(r2),
                    })

                x_axis_max = max_dist / 1.5 + max_dist * 0.05
                x_model = np.linspace(0.0, max(cutoff, 1e-6), 100)
                y_model = model.variogram(x_model)
                ax.plot(
                    x_model,
                    y_model,
                    color='#3944d7',
                    linewidth=0.8,
                    label=QCoreApplication.translate(
                        "Tab 2", "Model: {model}"
                    ).format(model=resolved['model']),
                    zorder=3,
                )

                if np.isfinite(data_variance) and data_variance > 0:
                    ax.axhline(
                        data_variance,
                        color='#7f8c8d',
                        linestyle=':',
                        linewidth=1.0,
                        alpha=0.7,
                        zorder=1,
                        label=QCoreApplication.translate(
                            "Tab 2", "Data variance"
                        ),
                    )

                ax.annotate(
                    f"R² = {r2:.3f}",
                    xy=(float(np.max(x_model)), float(np.min(y_model))),
                    ha='right',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='#3944d7',
                )

                ax.set_xlim(0.0, x_axis_max)
                ax.set_xlabel(QCoreApplication.translate("Tab 2", "Distance"))
                ax.set_ylabel(
                    QCoreApplication.translate("Tab 2", "Semivariance")
                )
                ax.set_ylim(bottom=0)
                title = QCoreApplication.translate(
                    "Tab 2", "Variogram – {param}"
                ).format(param=self.current_data["attribute"])
                if self.current_data.get('log_transform', False):
                    title += QCoreApplication.translate(
                        "Tab 2", " (Log transform)"
                    )
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

                self._install_cutoff_handle(ax, max_dist, cutoff)

                # Legend below the axes (outside the drawn data area).
                self.figure.subplots_adjust(
                    left=0.12, right=0.98, top=0.88, bottom=0.28
                )
                handles, labels = ax.get_legend_handles_labels()
                self.figure.legend(
                    handles,
                    labels,
                    loc='lower center',
                    ncol=3,
                    fontsize=7,
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.02),
                )

            except Exception as e:
                print(f"Error updating plot: {str(e)}")

        self.canvas.draw()
        return float(r2)

class StatsTableHeightFilter(QObject):
    """Recalculate stats_table height when the table is resized."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self._dialog = dialog

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            self._dialog._adjust_stats_table_height()
        return False

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
