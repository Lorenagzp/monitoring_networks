# Monitoring Networks

**Goal:** optimize a monitoring network of wells so you sample fewer points while retaining as much spatial information as possible (variance reduction via a Kalman-style ranking).

The tool appears on the QGIS toolbar **Monitoring networks** and opens the dialog **Monitoring Networks Prioritization**.

### What it can do

- Compute descriptive statistics and experimental/theoretical **variograms** per parameter
- Run leave-one-out **cross-validation** (OK) to assess model fit
- Build or import an **estimation grid** (optional node weights), from an Excel file or from a previously loaded point layer
- Use optional **well weights** and **parameter weights** (multi-parameter)
- Rank wells by information gain and show **prioritization order**
- Map **ordinary kriging (OK)** and optional **standard error** surfaces for all wells vs the first *N* prioritized wells
- Download the selected-wells network as a layer, its OK interpolation and kriging standard-error surfaces as layers, and its cross-validation details as Excel

---

## Workflow overview
* Generated automatically with IA

Work through the five tabs **in order**. Tabs 2–5 stay locked until **Next →** unlocks them. Changing the point layer (or other key inputs) can clear downstream results and lock later tabs again.

| Step | Tab | Purpose | You finish when… |
|------|-----|---------|------------------|
| 1 | **Input Data** | Choose the well layer, numeric parameters, and which wells to include | **Next →** runs geostatistics successfully |
| 2 | **Geostatistics** | Review stats, fit variograms, check cross-validation | Models look acceptable; **Next →** opens the grid tab |
| 3 | **Estimation Grid** | Create or import the spatial grid used for variance reduction | A grid is calculated or loaded; **Next →** is enabled |
| 4 | **Prioritization** | Configure weights and run **Optimize** | Ranking + variance-reduction curve exist; **Next →** opens maps |
| 5 | **Map** | Compare OK (and SE) maps and CV for all wells vs the first *N* wells | Export layers if needed (**Next →** is disabled here) |

**Why order matters**

1. Statistics and variograms need a valid point sample (Tab 1).
2. Optimization needs those variograms (Tab 2) and a full estimation grid (Tab 3).
3. Maps and network CV need a Kalman ranking from Tab 4.

Shared footer on every tab: progress bar (**State: …**), **← Previous**, **Next →**, **Close**.

---

## Installation

### 1. Install the plugin (ZIP)

1. Download the plugin ZIP folder which the archive contains `monitoring_networks/`.
2. In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP**.
3. Enable **Monitoring Networks** and open it from the **Monitoring networks** toolbar.

The ZIP only installs the plugin code. Python dependencies must be installed separately (step 2).

### 2. Install Python dependencies

Extra packages are listed in [`requirements.txt`](requirements.txt):

| Package | Needed for |
|---------|------------|
| **gstools** | Variograms and kriging |
| **shapely** | Estimation-grid hull / node filtering |
| **openpyxl** | Import estimation grid from Excel |
| **XlsxWriter** | Export prioritization and cross-validation details to Excel |

**NumPy**, **SciPy**, and **Matplotlib** are normally already included with QGIS.

**gstools** and **shapely** are imported when the plugin loads, so they must be installed before you open the dialog. **openpyxl** and **XlsxWriter** are only needed when you actually import or export an `*.xlsx` file (Tab 3 grid import, Tab 4 **Download prioritization**, Tab 5 **Download cross-validation details**); if either is missing, the plugin tries to install it automatically into the QGIS Python environment the first time it is needed, so most users never have to do this by hand. If that automatic install doesn't succeed, you'll see a warning asking you to install it manually, using the same command below — check the QGIS **Log Messages** panel, **"Monitoring Networks"** tab, for the specific reason (pip output, missing permissions, etc.) if you want to understand why.

Install into the **QGIS Python environment** (not a separate system Python):

**Windows (OSGeo4W Shell)** — open *OSGeo4W Shell* from the Start menu, then:

```bat
python -m pip install gstools shapely openpyxl XlsxWriter
```

After installing, restart QGIS if the plugin was already loaded.

**Alternative:** use a QGIS plugin such as **qpip** to install the same packages from inside QGIS.

<details>
<summary>Note for developers: why the automatic install locates its own python.exe on Windows</summary>

On the Windows standalone QGIS installer, the embedded Python interpreter runs inside the QGIS application executable itself (`qgis-bin.exe` / `qgis-ltr-bin.exe`), so `sys.executable` resolves to that launcher rather than to a real `python.exe`. Running `qgis-ltr-bin.exe -m pip install ...` doesn't invoke pip — it starts another instance of QGIS, which never exits on its own, so a naive auto-install hangs until a subprocess timeout kills it.

`_find_python_executable()` in `monitoring_networks_analysis.py` works around this by looking for `python.exe` inside `sys.exec_prefix` (and related `sys.*_prefix` values) first — that directory is the real bundled Python install (e.g. `...\QGIS 3.40.4\apps\Python312\`) and does contain a standalone `python.exe`. `sys.executable` is used only as a last resort, and only if its filename doesn't look like the QGIS application itself.
</details>

---

## Requirements & input specifications

### Software

- **QGIS** ≥ 3.22 "Białowieża" (LTR) — declared in `metadata.txt`; see the note below for why this is the realistic floor
- Extra Python packages: see **Installation** and `requirements.txt` above

> **Note on the declared minimum:** `qgisMinimumVersion` only controls which QGIS installations the Plugin Manager will offer this plugin to — QGIS does not otherwise enforce it, so the real constraint is Python, not QGIS itself. `requirements.txt`'s **gstools** requires **Python ≥ 3.8**. QGIS bundles its own Python (via the OSGeo4W installer on Windows, or the system/Homebrew Python on Linux/macOS), and that bundled version only reached **Python 3.9** starting around the **3.22 LTR** line (released February 2022) — earlier 3.x releases (3.16–3.20 and older) shipped Python 3.7, which cannot install `gstools`. From 3.22 onward every subsequent release (3.28, 3.34, 3.40, 3.44…) bundles Python 3.9 or newer, so `qgisMinimumVersion=3.22` is set as the realistic minimum where the plugin's dependencies install and the plugin runs without the Qt6/Python-version issues seen on older or newer setups. If you need to support an older QGIS install anyway, pin `gstools` (and the other packages in `requirements.txt`) to a version compatible with that install's Python, and test the full workflow there — it is not expected to work out of the box.

### Point layer (wells)

| Requirement | Detail |
|-------------|--------|
| Geometry | **Point** layer only |
| CRS | Shown on Tab 1 (`authid`). Distances are **Euclidean in map units** (not geoidal) |
| Parameters | Numeric fields only (`Int` / `Double` / `LongLong`) |
| Inclusion | Tab 1 **Include** column (default: all wells included) |
| IDs in plots/tables/exports | Sequential **1..N** well number — see **Well identifiers (Well_ID)** below |
| Nulls | Null / non-numeric / NaN / Inf values are **omitted** per parameter |
| Log transform | Only values **> 0** are kept; ≤ 0 are omitted |

**Typical failures**

- No point layer or no selected parameter → cannot leave Tab 1
- After log transform, no positive values → recalculation error for that parameter
- Cross-validation needs **≥ 3** valid points
- Combined multi-parameter optimize needs the **same valid wells** for every parameter (nulls/log can break alignment)

#### Well identifiers (Well_ID)

Every plot, table and export that needs to identify a well (histogram/spatial-distribution hover labels, the Tab 2/Tab 5 cross-validation "Details" tables, the Tab 4 prioritization order table, Tab 5 map markers, and the **Download prioritization**, **Download selected wells as layer** and **Download cross-validation details** outputs) uses the same identifier: a plain sequential number from **1** to **N**, where **N** is the **total** number of wells loaded in the layer.

This number:

- Never depends on the layer having an `id`-like field (`id`, `clave`, `cve`, `pozo`, `well`, `name`, `nombre`, …) — a layer with no such field, or with one that has gaps or duplicates, is numbered exactly the same way.
- Never depends on the provider's internal feature ID (`fid`), which can start at 0, skip numbers, or differ between providers/formats. `fid` is used only internally to sort features into a reproducible order before numbering; it is never shown.
- Is assigned **once**, from the full layer, and reused everywhere — the same well is always "3" on every plot, table and export, even for a parameter where that well's value happens to be null and it drops out of that parameter's own plots/stats.
- Does **not** renumber when wells are checked/unchecked on Tab 1. Unchecking a well simply leaves a gap: its number stops appearing anywhere while it is excluded, and every other well keeps its own number unchanged. Re-checking it brings it back with the same original number. This way "well 7" always refers to the same physical well, whatever else is currently selected — deselecting still invalidates and recalculates every downstream result as usual (see **Tab 1 — Buttons** below), it just never changes what number a well carries.

Any `id`-like field the layer already has (CVE, POZO, NAME, …) is still exported as a plain data column — in **Download selected wells as layer**'s original-fields block and in `records_by_point_id`-based exports — it is simply no longer used as the *label* for a well.

### Estimation grid (Tab 3)

**Generate in the plugin**

- Spacing and buffer use the **layer CRS map units**
- Temporary / saved grid layer uses the same CRS as the wells layer
- Generated grids have **uniform node weight = 1** (Tab 4 grid-weight option stays unavailable)

**Import `*.xlsx`**

| Spec | Detail |
|------|--------|
| Columns (order) | **ID**, **X**, **Y**; optional 4th **weight** |
| Sheet | First sheet (via openpyxl) |
| Header | Expected on row 1; data from row 2 |
| CRS | Plugin does **not** reproject Excel XY — use the same CRS / map units as the wells |
| Weights | If any weight column exists, missing cells default to `1.0`; no weights → unweighted grid |

**Typical failures**

- Empty file, fewer than 3 columns, or no valid ID/X/Y rows
- openpyxl / XlsxWriter missing in the QGIS Python environment and the automatic install (see **Installation**) could not complete — install the package manually and try again
- **Next →** on Tab 3 disabled until a grid exists

**Use a previously loaded point layer**

- Any point layer already in the project can be used directly as the estimation grid: every feature becomes a grid node.
- An optional numeric field on that layer can supply per-node weights (same convention as the Excel import: if any feature has a weight, cells without one default to `1.0`).
- The plugin does **not** reproject the layer — it must already share the wells' CRS / map units.
- A grid imported this way enables the Tab 4 **Use estimation grid node weights** option exactly like an Excel import with a weight column does, when a weight field was picked.

### Optional well-weight field (Tab 4)

Auto-detected (case-insensitive) if the field name matches one of:

`peso_pozo`, `well_weight`, `w_pozo`, `peso_w`, `well_w`, `weight_well`, `pozo_peso`, `w`

Analysis attributes and ID-like fields are ignored for detection.

---

## Interactive elements reference

### Shared controls

| Control | What it does |
|---------|----------------|
| **← Previous** | Go to the previous unlocked tab |
| **Next →** | Advance the workflow (runs Tab 1 geostatistics; gated on Tabs 3–4) |
| **Close** | Close the dialog |
| Progress bar | Shows **State: Ready**, **State: Completed**, **State: Error**, or step-specific messages |

---

### Tab 1 — Input Data

#### Layer & attributes

| Control | Type | Notes |
|---------|------|--------|
| **Point layer:** | Dropdown (`QgsMapLayerComboBox`) | Point layers in the project only. Changing layer resets later tabs |
| Attribute list | Multi-select list | Numeric fields usable as monitoring parameters |
| Layer info panel | Read-only | `Name`, feature count, CRS |
| Attribute values table | Table | Column **Include** (checkbox) + all layer fields. Preview capped at **500** rows (label warns when truncated). Select/Deselect still apply to **all** features |

#### Buttons

| Button | When to click | Result |
|--------|---------------|--------|
| **Select all wells** | After loading a layer | Checks every **Include** box; clears downstream Tab 2/4/5 results |
| **Deselect all wells** | To exclude everyone | Unchecks all; clears downstream results |

Unchecking any well (or Select/Deselect all) fully invalidates every downstream result, not just the raw statistics: it clears stats, variograms, CV, and optimization caches; it also **drops the Tab 3 estimation grid** (its hull boundary was built from well positions, so a grid calculated before the change would still extend to cover deselected wells' locations until regenerated) and refreshes the nearest-neighbor stats (Avg D) that drive the Tab 2 default lag size and the Tab 3 default node spacing, so those defaults reflect only the wells still included. Tabs 2–5 **lock again** exactly as when the input layer changes — recalculate geostatistics via **Next**, regenerate the grid, and re-run **Optimize** before trusting any later tab. Attribute selection on Tab 1 itself is kept.

#### Hints

- No attribute selected: *Select an attribute of the layer to use as optimization parameter*
- Ready: *Click Next to run the geostatistics*
- Next failure: *Select a point layer and at least one parameter on tab 1, then try again.*

---

### Tab 2 — Geostatistics

Click a **parameter name** in the stats table vertical header to drive plots, variogram, and CV for that parameter.

**Tooltip (parameter names):**  
`Click a parameter name to show its geostatistics, variogram and cross-validation.`

#### Basic statistics table

| Column | Editable? | Meaning |
|--------|-----------|---------|
| **Weight** | Yes | Relative importance when combining parameters on Tab 4. Default **1/N** (sum = 1). Must be **> 0** |
| **Transformation** | Combo | **None** (default) or **Logarithmic**. Recalculates stats + variogram for that parameter only |
| **Count** … **Kurtosis** | No | Descriptive stats (Asymmetry = skewness) |

The table always resizes to show **every** analyzed parameter as its own row — there is no internal scrollbar and no cap on how many rows are shown at once, so with several parameters selected on Tab 1 (e.g. multiple hydrogeochemical species) you always see all of them together, never only a partial subset. If the table (plus the plots below it) ends up taller than the window, the tab's own outer scrollbar handles that — scroll the tab, not the table.

**Tooltip — Weight:**  
`Relative importance when combining parameters on tab 4. Default: equal share (1/N) so weights sum to 1.`

**Tooltip — Transformation:**  
`Data transform for this parameter. Changing it recalculates statistics and variogram for this parameter only.`

**Tooltip — Asymmetry (skewness):**  
`Skewness measures the lack of symmetry in the data distribution.`

- Between −0.5 and 0.5: Symmetric  
- From −1 to −0.5 or 0.5 to 1: Close to symmetric  
- Less than −1 or greater than 1: High skewness  

**Tooltip — Kurtosis:**  
`Excess kurtosis measures how peaked or flattened a distribution is.`

- Between −2 and 2: Normal  
- Greater than 2: Sharper peak / heavy tails  
- Less than −2: Flatter peak / light tails  

Cells for asymmetry/kurtosis are color-coded to match those bands.

#### Plots

- **Histogram** — value distribution with mean, median, ±1σ / ±2σ guides  
- **Spatial distribution** — X/Y scatter of wells (hover shows IDs)

#### Variogram parameters

| Control | Type | Options / range | Effect |
|---------|------|-----------------|--------|
| **Model** | Combo | `spherical` (typical default), `exponential`, `gaussian`, `stable`, `matern` | Changing model re-runs auto-fit |
| **Nugget** / **Sill** / **Range** | Editable cells | Continuous | Manual edits refresh the plot and CV. **Range** is GSTools `len_scale` (not always the "practical range") |
| **Lag size:** | Combo | **Avg D / 3**, **Avg D / 2**, **Avg D**, **1.5 × Avg D**, **2 × Avg D** | Bin spacing for the experimental variogram. Default: **Avg D** if nearest-neighbor index ≥ 1, else **Avg D / 2** |
| Variogram limit handle | Drag handle on plot | Factor **1.5–5.0**, step **0.5**, default **3.0** | Red dashed vertical line. Cutoff ≈ `max_dist / factor`. Drag to re-bin and re-autofit |
| Practical range line | Read-only marker on plot | — | Green dash-dot vertical line at the distance where the **fitted model** reaches 95% of its total sill. Recomputed on every redraw, so it moves whenever **Model**, **Nugget**, **Sill**, or **Range** change |

**Avg D** = average nearest-neighbor distance among included wells (map units). Hint under the lag combo shows Avg D, lag size, and max pairwise distance in the data.

**Tooltip — Lag size:**  
`Lag spacing for the experimental variogram, based on the observed mean nearest-neighbor distance (Avg D).`

**Tooltip — Range:**  
`Range shown is GSTools length scale (len_scale). It is not necessarily the practical range.`

The **red dashed "Variogram limit"** handle and the **green dash-dot "Practical range"** line are two different things and move independently:
- **Variogram limit** (red, draggable) only depends on the data's max distance and the lag-size factor — it controls experimental-variogram binning, not the model.
- **Practical range** (green, read-only) is the distance where the *fitted model itself* reaches 95% of its sill, computed directly from GSTools for whichever model is active — spherical, exponential, gaussian, stable, or matérn — so it always reflects the current **Model** and **Range** (not a fixed per-model multiplier of `len_scale`).

**Autofit status**

- Success: *Variogram parameters auto-fit converged…*  
- Failure: *Auto-fit did not converge; check outliers, variogram limit, lag size…*

#### Cross-validation (leave-one-out OK)

| Block | Contents |
|-------|----------|
| **Cross-Validation Summary** | Min/Max/Mean error, **MAE**, **RMSE**, **ASE**, **MSE**, **RMSSE** |
| **Cross-Validation Details** | **ID**, **Included?**, **Measured**, **Predicted**, **Error**, **SE**, **Standardized Error** |
| Scatter plot | Measured vs predicted with 1:1 and regression lines |

**Tooltips (summary headers)**

| Metric | Exact tooltip |
|--------|----------------|
| **ASE** | `Average Standard Error (ASE): root mean square of the kriging standard errors. Ideally close to RMSE.` |
| **MSE** | `Mean Standardized Error (MSE): mean of error/SE. Ideally close to 0 (unbiased standardized residuals).` |
| **RMSSE** | `Root-Mean-Square Standardized Error (RMSSE): root mean square of error/SE. Ideally close to 1.` |

Prefer CV diagnostics over a "pretty" experimental-vs-model curve alone when judging fit.

---

### Tab 3 — Estimation Grid

The estimation grid is the set of locations where kriging variance is evaluated during Optimize. No grid → cannot continue to prioritization.

#### Calculate the estimation grid

| Control | Range / default | Meaning |
|---------|-----------------|---------|
| **Spacing between nodes (map units):** | 10–10000; default ≈ **Avg D** (fallback 100) | Node spacing in layer CRS units |
| **Estimated total points:** | Read-only | Live estimate `{count} points ({n_x} × {n_y})` |
| **Boundary buffer (x times node spacing):** | 0.0–3.0, step 0.5; default **1.0** | Expands the hull by `factor × spacing` |
| **Boundary tightness (tight ↔ smooth)** | 0.1–1.0, step 0.1; default **0.7** | QGIS concave-hull **alpha** (lower → tighter/more concave; higher → smoother/more convex). UI minimum is 0.1 |

**Tooltip — buffer:**  
`Expands the concave hull outward by this many times the node spacing. Default 1.0 adds one spacing interval beyond the hull.`

**Tooltip — tightness:**  
`Alpha parameter for the QGIS hull around the wells. 0 = concave , 1 = convex .`

#### Buttons

| Button | When | Result |
|--------|------|--------|
| **Calculate grid** | Layer selected; spacing/buffer/alpha set | Builds buffered hull + nodes, stores grid, enables **Next →**, shows preview |
| **Save Grid as Temporary Layer** / **Save grid as layer** | After a grid exists | Adds a memory point layer in the wells CRS |
| **Select *.XLSX file** / **Load grid as layer** | Optional import | Loads ID, X, Y [, weight]; preview plot |
| **Load layer as grid** | Optional import | Uses every feature of a previously loaded point layer as a grid node, with an optional numeric field as per-node weight; preview plot |

Excel import group title: **Or Upload the estimation grid from *.XLSX file [Optional]**  
Hint: *Required columns: ID, X, Y. Optional column: weight.*

Point-layer import group title: **Or Select a Previously Loaded Point Layer [Optional]**  
Controls: **Point layer:** (point layers in the project only) and **Node weight field (optional):** (numeric fields of the chosen layer, or none).

**Next →** idle texts

- No grid: *Calculate or import an estimation grid to continue.*  
- Has grid: *Click Next to continue to well prioritization.*

---

### Tab 4 — Prioritization

#### Configure monitoring network

| Control | Type | Default | Notes |
|---------|------|---------|--------|
| **Select parameter to optimize:** | Combo | Disabled until params exist | One Tab-1 attribute, or **Parameters combined (Weighted)** if ≥ 2 parameters |
| **Use personalized well weight** | Checkbox | Off | Enabled only if a weight field is detected |
| **Use estimation grid node weights** | Checkbox | Off | Enabled only when the grid was imported (Excel or point layer) with a weight column/field |
| **Optimize** | Button | — | Runs Kalman ranking + variance-reduction curve |
| **Download prioritization** | Button | — | Excel export after a successful Optimize |

**Tooltip — well weight:**  
`Detected on the input point layer when a field name matches (case-insensitive): peso_pozo, well_weight, w_pozo, peso_w, well_w, weight_well, pozo_peso, or w. The analysis attribute and ID-like fields are ignored.`

**Tooltip — grid weights:**  
`Available after importing an estimation grid on tab 3 with a weight column, either from an Excel file (columns in order: ID, X, Y, weight) or from a point layer with a numeric weight field. Generated grids or grids without a weight column/field cannot use this option.`

Toggling either weight checkbox **clears** the current Optimize result — click **Optimize** again.

#### Variance Reduction

| Control | Notes |
|---------|--------|
| Variance plot | **Prioritized wells** vs **Adverse order**; axes = number of points vs remaining **Total variance (%)**. Reference lines at **90%** and **95%** of max reduction |
| **Optimization order** table | **Priority**, **ID**, **Well weight**, **Total variance (%)**, **Adverse order**, **Adverse variance (%)** (read-only, sortable) |

**Optimize (high level)**

1. Validates parameter, Tab-3 grid, optional weights  
2. Uses Tab-2 variogram(s); for combined mode, Tab-2 **Weight** column  
3. Ranks wells by greedy variance reduction (optional well × grid weights)  
4. Builds remaining-variance curve; sets Tab 5 default *N* near **95%** of max reduction  

**Next →** requires Optimize results for the **currently selected** parameter.  
Hints: *Run Optimize to compute well prioritization before continuing.* / *Click Next to view the optimization map.*

**Download prioritization** writes sheets such as prioritization order + variogram/CV settings (`prioritization_{param}.xlsx` or `prioritization_combined.xlsx`). The prioritization-order sheet starts with a `Well_ID` column (the sequential **1..N** identifier — see **Well identifiers (Well_ID)**), followed by the source layer's own fields, then `Priority`, `Variance`, `Adverse_Order`, `Adverse_Variance`, `Use_Well_Weight`, `Use_Grid_Weight`. Needs **XlsxWriter**, which the plugin tries to install automatically the first time you use this button if it isn't already present (see **Installation**).

---

### Tab 5 — Map

Uses the parameter(s) selected on Tab 4 for well ranking. When Tab 4 is in **Parameters combined (Weighted)** mode, the **Parameter shown on maps** selector lets you pick which one of the analyzed (combined) parameters the O.K./S.E. maps and cross-validation are built from — the well ranking itself does not change, only which parameter's values and variogram are interpolated. In single-parameter mode the selector just shows that one parameter. Compares **all wells** (left) vs the **first N prioritized wells** (right).

#### Controls

| Control | Type | Default | Effect |
|---------|------|---------|--------|
| **Parameter shown on maps:** | Dropdown | First analyzed parameter (combined mode); the selected parameter (single mode) | Switches which analyzed parameter's O.K./S.E. maps and CV are shown. Only enabled/populated once Tab 4 has a parameter (or combined set) selected |
| **Number of monitoring wells:** | Spin box | Min **3**; default ≈ wells at **95%** of max variance reduction | Rebuilds selected-network OK/SE maps and CV |
| **Download selected wells as layer** | Button | — | Temporary point layer for the selected-wells network (see below for its fields) |
| **Download interpolation as layer** | Button | — | Temporary GeoTIFF of the OK surface for the selected *N* wells, styled with the same continuous viridis-like ramp shown on the in-app O.K. map |
| **Download kriging standard error as layer** | Button | — | Temporary GeoTIFF of the kriging standard-error surface for the selected *N* wells, for the parameter shown on maps, styled with the same continuous Reds-like ramp shown on the in-app S.E. map |
| **Download cross-validation details** | Button | — | Excel export (summary + details) of the leave-one-out cross-validation for the selected-wells network, for the parameter shown on maps |
| **Show kriging standard error maps** | Checkbox | **Off** | Shows/hides SE map panels |
| **Show monitoring wells on maps** | Checkbox | **On** | Overlay well markers (no re-krige) |
| **Color O.K. wells by value** | Checkbox | **On** | Same color ramp as the OK surface; off → all wells red |

**Tooltip — parameter shown on maps:**  
`Choose which analyzed parameter drives the O.K./S.E. maps and cross-validation below. The well ranking stays the same; only the interpolated values and variogram change.`

**Tooltip — download interpolation:**  
`Download the kriging interpolation map as a temporary layer. The raster resolution is based on the estimation grid spacing.`

**Tooltip — download kriging standard error:**  
`Download the kriging standard error surface for the selected-wells network as a temporary raster layer. The raster resolution is based on the estimation grid spacing.`

**Tooltip — download cross-validation details:**  
`Export the leave-one-out cross-validation summary and details for the selected-wells network of the parameter shown on maps, as an Excel file.`

**Tooltip — color wells:**  
`When checked, well markers on O.K. interpolation maps use the same color ramp as the surface. When unchecked, all wells are drawn in red.`

##### Download selected wells as layer — field order

The temporary point layer's attribute table starts with **every field the source (input) point layer already has** — the same layer used to generate the well prioritization — followed by:

1. `well_id` — the well's sequential **1..N** identifier (see **Well identifiers (Well_ID)** above); the same number shown for that well everywhere else in the plugin
2. `prioritization_rank` — 1-based rank in the selected-wells network
3. `total_variance_pct` — remaining total variance (%) after this well joins the network
4. `predicted_<parameter>` — one column per analyzed parameter (a single column in single-parameter mode; one per combined parameter in **Parameters combined (Weighted)** mode), holding the leave-one-out cross-validation predicted value for that well within the selected-wells network (same values as the Tab 5 "Selected wells" cross-validation table)

#### Maps & CV

| Panel | Content |
|-------|---------|
| OK maps | Titles like **O.K. – {param}** for all wells and for *N* wells, where `{param}` is whatever is picked in **Parameter shown on maps** |
| SE maps | **S.E. – {param}** when the SE checkbox is on |
| **Cross-Validation Summary / Details** | Same metrics as Tab 2; selected-network CV uses the first *N* wells as the network. **Included?** = Yes/No |

**Combined-parameter note:** in combined mode, OK/SE maps and CV on Tab 5 are built with **one** parameter's variogram at a time (never a true multivariate model) — the **Parameter shown on maps** selector picks which one; it defaults to the first analyzed parameter until you choose otherwise. The **Download selected wells as layer** button is the exception: it adds a `predicted_<parameter>` column for **every** combined parameter, not just the one shown on maps.

If Optimize is cleared (variogram/weights/grid change), Tab 5 prompts to run Optimize on Tab 4 again.

---

## Tips & common workflows

### Typical single-parameter run

1. Tab 1 — Select wells layer + one numeric parameter → **Next →**  
2. Tab 2 — Check histogram/skewness; try **Logarithmic** if strongly skewed and all values > 0  
3. Tab 2 — Adjust **Lag size** / drag the **variogram limit**; confirm **ASE ≈ RMSE**, **MSE ≈ 0**, **RMSSE ≈ 1**  
4. Tab 3 — Keep default spacing ≈ Avg D and buffer **1.0** → **Calculate grid** → **Next →**  
5. Tab 4 — **Optimize** → inspect variance curve (90%/95% lines) → **Next →**  
6. Tab 5 — Set *N* near the 95% line; toggle SE maps if you need uncertainty surfaces; download layers/files as needed  

### Multi-parameter (weighted)

1. Select ≥ 2 numeric attributes on Tab 1  
2. On Tab 2, edit **Weight** so relative importance sums to 1  
3. On Tab 4 choose **Parameters combined (Weighted)** → **Optimize**  
4. Remember: each parameter has its **own** variogram; cross-correlation between parameters is **not** modeled  

### Using weights

| Goal | What to do |
|------|------------|
| Prefer certain wells | Detected weight field → enable **Use personalized well weight** → Optimize again |
| Prefer certain areas | Import a grid (Excel or point layer) with a **weight** column/field → enable **Use estimation grid node weights** → Optimize again |
| Generated grid only | Grid-weight checkbox stays disabled (all nodes weight 1) |

### Gotchas

- **Fit vs prediction:** a high R² on the variogram cloud is not enough — trust **cross-validation** (Tab 2 / Tab 5).  
- **Euclidean distances** in the layer CRS: for geographic CRS (degrees), results are not true ground distances — prefer a projected CRS.  
- Editing Tab 2 model/nugget/sill/range or Tab 3 grid / Tab 4 weight toggles often **invalidates** Optimize — re-run **Optimize** before trusting Tab 5.  
- Unchecking a well on Tab 1 locks tabs 2–5 again and drops the Tab 3 grid — re-run **Next** → **Calculate grid** → **Optimize** so the excluded well is actually gone from the variogram, the grid's boundary, and the ranking, not just from the raw stats.  
- Attribute table preview shows at most **500** rows, but Include still applies to the full layer.  
- Excel grid XY, and a point layer used as the grid, must already match the wells CRS (no reprojection).  
- Alpha spinbox cannot go below **0.1** even though the tooltip mentions 0 = fully concave.

---

## Important notes

- More important than a visually good experimental vs theoretical variogram (or a high R²) is a sound **cross-validation** assessment of how the model predicts held-out wells.
- After changing lag, cutoff handle, transform, or model, re-check CV metrics before Optimize.

---

## Limitations

- In multi-parameter mode, each contaminant/parameter has its **own** variogram; the tool does **not** model correlation between parameters.
- Average / pairwise distances use **Euclidean** geometry in map units, not geoidal (ellipsoidal) distance.
- Experimental plugin (v0.1): expect iterative improvements; report issues via the project tracker in `metadata.txt`.
