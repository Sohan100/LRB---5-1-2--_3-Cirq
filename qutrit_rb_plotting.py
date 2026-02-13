"""
qutrit_rb_plotting.py
=====================

Centralized plotting and comparison utilities for qutrit RB studies.

Purpose
-------
This module decouples analysis/visualization from simulation scripts.
Simulation
files produce checkpoints and `.npy` summaries, while this module reads those
artifacts and generates publication-style figures.

Supported data sources
----------------------
Logical RB checkpoint directories produced by either:
- `qutrit_logical_rb_logical_noise_sim.py` (sampled logical-noise model), or
- `qutrit_logical_rb_terminal_check_sim.py` (terminal-check local-noise model).

Physical RB baseline files produced by:
- `qutrit_physical_rb_sim.py` (`physicalRB_SimulationResults.npy`).

Compatibility strategy
----------------------
Checkpoint and result loaders are tolerant to both legacy camelCase field names
and newer snake_case field names so old and new simulation runs can be plotted
without manual conversion.

Typical workflow
----------------
1. Instantiate `LogicalRbPlotter(logical_checkpoint_dir, experiment_name)`.
2. Select error rates explicitly or infer via `get_error_rates()`.
3. Generate one or more of:
   - fidelity curves
   - survival-ratio curves
   - logical-success curves
   - logical-vs-physical infidelity threshold plot
   - logical-vs-physical RB overlay grid
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class LogicalRbPlotter:
    """
    This class provides plotting/analysis utilities for logical RB experiment
    outputs and logical-vs-physical comparisons.

    It is intentionally experiment-agnostic at the plotting layer: the same
    instance works with checkpoint directories produced by either logical-noise
    LRB or terminal-check local-noise LRB, and it can optionally overlay
    physical RB baselines from saved `.npy` payloads.

    Attributes:
        logical_checkpoint_dir (str): Directory containing `logicalRB_p*.pkl`
            checkpoints and optionally `final_logical_error_rate.pkl`.
        experiment_name (str): Human-readable and filename-safe label used in
            plot titles/output names (e.g., `logical_noise`,
            `terminal_check`).
        output_dir (str): Directory where generated figures are written.

    Methods:
        __init__(logical_checkpoint_dir, experiment_name, output_dir=None):
            Initializes plotting context and output subdirectories.
        infer_error_rates():
            Infers available physical rates from checkpoint filenames.
        load_final_logical_error_curve():
            Loads final physical-vs-logical infidelity curve data.
        get_error_rates(error_rates=None):
            Resolves explicit or inferred plotting rate list.
        plot_fidelity_curves(error_rates=None):
            Plots expectation-value RB depth curves per physical rate.
        plot_survival_ratios(error_rates=None):
            Plots post-selection survival-ratio depth curves.
        plot_logical_success(error_rates=None):
            Plots logical-success depth curves.
        plot_logical_vs_physical_infidelity():
            Plots logical vs physical infidelity with optional threshold fit.
        overlay_with_physical_rb(physical_results_path, error_rates=None,
                                 show_plot=True, fit_curves=True,
                                 y_range=None):
            Overlays logical and physical RB expectation data for each rate.
            Fit curves can be enabled/disabled.
        overlay_terminal_check_with_physical_rb_no_fit(
            physical_results_path, error_rates=None, show_plot=True,
            y_range=None):
            Convenience wrapper for terminal-check overlays without fit
            models.

    Internal helper methods:
        _fit_func, _fit_func_with_offset, _sigmoid, _load_pickle,
        _metric_key_candidates, _get_metric_dict, _sorted_depth_data,
        _plot_metric_grid, _lookup_physical_result_for_p

    Data compatibility:
        This class accepts both snake_case and camelCase key conventions in
        historical checkpoints/results (`expectation_values` vs
        `expectationValues`, etc.).
    """

    def __init__(
        self,
        logical_checkpoint_dir: str,
        experiment_name: str,
        output_dir: Optional[str] = None,
    ):
        """
        Configure plotting context and output directories.
        
        Args:
            logical_checkpoint_dir (str):
                Directory containing logical RB checkpoints
                                          and summary files.
            experiment_name (str):
                Experiment label appended to output filenames and
                                   figure titles.
            output_dir (Optional[str]):
                Optional root directory where generated plot
                                        artifacts are saved.
        
        Returns:
        None: `__init__` updates internal object state and returns no value.
        
        Raises:
            ValueError: If supplied argument values violate this method's input
                        assumptions.
        """
        self.logical_checkpoint_dir = logical_checkpoint_dir
        self.experiment_name = experiment_name
        self.output_dir = output_dir or logical_checkpoint_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir,
                                 "Fidelity Curves"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir,
                                 "Survival Probability Curves"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir,
                                 "Logical Success Curves"), exist_ok=True)

    @staticmethod
    def _fit_func(x, amplitude, decay):
        """
RB exponential model `A * f^x` used for expectation metrics.

Args:
    x (Any): Independent-variable values (for example RB depth values) used by
             the model function.
    amplitude (Any): Model amplitude parameter used in the fitted decay
                     expression.
    decay (Any): Model decay parameter used in the fitted RB expression.

Returns:
    object: Output produced by this routine according to the behavior described
            above (rb exponential model `a * f^x` used for expectation
            metrics.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return amplitude * (decay ** x)

    @staticmethod
    def _fit_func_with_offset(x, amplitude, decay, offset):
        """
Offset RB model `A * f^x + B` used for survival-ratio fitting.

Args:
    x (Any): Independent-variable values (for example RB depth values) used by
             the model function.
    amplitude (Any): Model amplitude parameter used in the fitted decay
                     expression.
    decay (Any): Model decay parameter used in the fitted RB expression.
    offset (Any): Additive offset parameter used by offset-enabled fit models.

Returns:
    object: Output produced by this routine according to the behavior described
            above (offset rb model `a * f^x + b` used for survival-ratio
            fitting.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return amplitude * (decay ** x) + offset

    @staticmethod
    def _sigmoid(x, x_0, steepness, amplitude, offset):
        """
Log-domain sigmoid used for threshold-shape fitting in a selected ROI.

Args:
    x (Any): Independent-variable values (for example RB depth values) used by
             the model function.
    x_0 (Any): Sigmoid center location in physical-error-rate space.
    steepness (Any): Sigmoid slope parameter controlling transition sharpness.
    amplitude (Any): Model amplitude parameter used in the fitted decay
                     expression.
    offset (Any): Additive offset parameter used by offset-enabled fit models.

Returns:
    object: Output produced by this routine according to the behavior described
            above (log-domain sigmoid used for threshold-shape fitting in a
            selected roi.).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        return amplitude / \
            (1 + np.exp(-steepness * (np.log10(x) - np.log10(x_0)))) + offset

    @staticmethod
    def _load_pickle(path: str) -> Optional[dict]:
        """
Load pickle file if it exists.

Args:
    path (str): Filesystem path used for loading or saving serialized data.

Returns:
    Optional[dict]: Output produced by this routine according to the behavior
                    described above (load pickle file if it exists.).

Raises:
    FileNotFoundError: If the requested input file cannot be found at the
                       provided path.
    pickle.UnpicklingError: If the serialized payload is malformed or
                            incompatible with the expected schema.
"""
        if not os.path.exists(path):
            return None
        with open(path, "rb") as file_obj:
            return pickle.load(file_obj)

    @staticmethod
    def _metric_key_candidates(metric_name: str) -> List[str]:
        """
        Return accepted key aliases for a metric.

        Args:
            metric_name: Canonical metric id (`expectation_values`,
                `survival_ratios`, or `logical_successes`).

        Returns:
            Ordered list of candidate dictionary keys.

        Raises:
            ValueError: If `metric_name` is unknown.
        """
        mapping = {
            "expectation_values": ["expectation_values", "expectationValues"],
            "survival_ratios": ["survival_ratios", "survivalRatios"],
            "logical_successes": ["logical_successes", "logicalSuccesses"],
        }
        if metric_name not in mapping:
            raise ValueError(f"Unknown metric_name: {metric_name}")
        return mapping[metric_name]

    def _get_metric_dict(self, checkpoint_data: dict,
                         metric_name: str) -> Dict[str, List[float]]:
        """
Extract a metric dictionary from checkpoint payload with alias
fallback.

Args:
    checkpoint_data (dict): Deserialized checkpoint payload for one physical
                            error rate.
    metric_name (str): Canonical metric key identifying which logical-RB
                       quantity to extract or plot.

Returns:
    Dict[str, List[float]]: Output produced by this routine according to the
                            behavior described above (extract a metric
                            dictionary from checkpoint payload with alias).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        for candidate in self._metric_key_candidates(metric_name):
            if candidate in checkpoint_data:
                return checkpoint_data[candidate]
        return {}

    @staticmethod
    def _sorted_depth_data(
            metric_dict: Dict[str, List[float]]) -> Tuple[
                List[int], List[float], List[List[float]]]:
        """
Convert depth-keyed metric dict into sorted plotting arrays.

Args:
    metric_dict (Dict[str, List[float]]): Depth-indexed metric samples loaded
                                          from checkpoint data.

Returns:
    Tuple[
                List[int], List[float], List[List[float]]]: Output
                                                                       produced
                                                                       by this
                                                                       routine 
                                                                       accordin
                                                                       g to the
                                                                       behavior
                                                                       describe
                                                                       d above
                                                                       (convert
                                                                       depth-
                                                                       keyed
                                                                       metric
                                                                       dict
                                                                       into
                                                                       sorted
                                                                       plotting
                                                                       arrays.)
                                                                       .

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if not metric_dict:
            return [], [], []

        sorted_items = sorted(metric_dict.items(),
                              key=lambda item: int(item[0]))
        x_data = [int(depth_key) for depth_key, _ in sorted_items]
        violin_data = [depth_values for _, depth_values in sorted_items]
        y_data = [float(np.mean(depth_values)) for depth_values in violin_data]
        return x_data, y_data, violin_data

    def infer_error_rates(self) -> List[float]:
        """
        Infer available physical error rates from checkpoint filenames.
        
        Args:
        None: `infer_error_rates` relies on object state and accepts no
        additional inputs.
        
        Returns:
            List[float]:
                Inferred value set computed from available checkpoint data.
        
        Raises:
        ValueError: If `infer_error_rates` receives inputs that are
        incompatible with its expected configuration.
        """
        error_rates = []
        for file_name in os.listdir(self.logical_checkpoint_dir):
            if not (file_name.startswith("logicalRB_p")
                    and file_name.endswith(".pkl")):
                continue
            value_str = file_name[len("logicalRB_p"): -len(".pkl")]
            try:
                error_rates.append(float(value_str))
            except ValueError:
                continue
        return sorted(error_rates)

    def load_final_logical_error_curve(
            self) -> Tuple[List[float], List[float]]:
        """
        Load (physical_error_rates, logical_error_rates) from final checkpoint.
        
        Args:
        None: `load_final_logical_error_curve` relies on object state and
        accepts no additional inputs.
        
        Returns:
            Tuple[List[float], List[float]]:
                Requested data object loaded or assembled
                                             by this method.
        
        Raises:
            FileNotFoundError: If the requested input file does not exist.
            pickle.UnpicklingError: If serialized data cannot be decoded.
        """
        path = os.path.join(self.logical_checkpoint_dir,
                            "final_logical_error_rate.pkl")
        data = self._load_pickle(path)
        if data is None:
            return [], []

        if "physical_error_rates" in data and "logical_error_rates" in data:
            return list(
                data["physical_error_rates"]), list(
                data["logical_error_rates"])

        if "physicalErrorRates" in data and "logicalErrorRates" in data:
            return list(
                data["physicalErrorRates"]), list(
                data["logicalErrorRates"])

        if isinstance(data, dict):
            points = []
            for key, value in data.items():
                try:
                    points.append((float(key), float(value)))
                except (TypeError, ValueError):
                    continue
            points.sort(key=lambda item: item[0])
            return [
                point[0] for point in points], [
                point[1] for point in points]

        return [], []

    def get_error_rates(
            self, error_rates: Optional[List[float]] = None) -> List[float]:
        """
Resolve the set of physical rates to plot.

Args:
    error_rates (Optional[List[float]]): Optional physical-error-rate list.
                                         When omitted, available rates are
                                         inferred from checkpoint files.

Returns:
    List[float]: Requested data object loaded or assembled by this method.

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if error_rates:
            return error_rates
        inferred = self.infer_error_rates()
        return inferred

    def _plot_metric_grid(
        self,
        error_rates: List[float],
        metric_name: str,
        title: str,
        y_label: str,
        color: str,
        output_subdir: str,
        output_file_name: str,
        fit_with_offset: bool = False,
    ):
        """
        Generic grid plot generator for depth-resolved RB metrics.
        
        Plot composition per panel: - violin plot of per-depth distributions, -
        scatter overlay of all raw points, - optional fit curve (`A*f^x` or
        `A*f^x+B`), - optional average-fidelity text box for non-offset fits.
        
        Args:
            error_rates (List[float]):
                Optional physical-error-rate list. When omitted,
                                       available rates are inferred from
                                       checkpoint
                                       files.
            metric_name (str):
                Canonical metric key identifying which logical-RB
                               quantity to extract or plot.
            title (str):
                Input argument consumed by `_plot_metric_grid` to perform this
                         operation.
            y_label (str):
                Input argument consumed by `_plot_metric_grid` to perform
                           this operation.
            color (str):
                Input argument consumed by `_plot_metric_grid` to perform this
                         operation.
            output_subdir (str):
                Input argument consumed by `_plot_metric_grid` to
                                 perform this operation.
            output_file_name (str):
                Input argument consumed by `_plot_metric_grid` to
                                    perform this operation.
            fit_with_offset (bool):
                Input argument consumed by `_plot_metric_grid` to
                                    perform this operation.
        
        Returns:
        None: `_plot_metric_grid` updates internal object state and returns no
        value.
        
        Raises:
            ValueError: If supplied argument values violate this method's input
                        assumptions.
        """
        if not error_rates:
            print("No error rates were provided or inferred; skipping plot.")
            return

        num_plots = len(error_rates)
        cols = 3
        rows = (num_plots - 1) // cols + 1

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        fig.suptitle(title, fontsize=16, y=0.95)
        axes_flat = np.atleast_1d(axes).flatten()

        for index, physical_error_rate in enumerate(error_rates):
            # One subplot per physical depolarizing rate.
            ax = axes_flat[index]
            file_name = os.path.join(
                self.logical_checkpoint_dir,
                f"logicalRB_p{physical_error_rate}.pkl")
            checkpoint_data = self._load_pickle(file_name)

            if checkpoint_data is None:
                ax.set_title(f"p={physical_error_rate} (missing)")
                ax.axis("off")
                continue

            metric_dict = self._get_metric_dict(checkpoint_data, metric_name)
            x_data, y_data, violin_data = self._sorted_depth_data(metric_dict)

            if not x_data:
                ax.set_title(f"p={physical_error_rate} (empty)")
                ax.axis("off")
                continue

            try:
                if fit_with_offset:
                    # Survival-ratio curves may include non-zero asymptote.
                    fit_parameters, _ = curve_fit(
                        self._fit_func_with_offset,
                        x_data,
                        y_data,
                        p0=[1.0, 0.9, float(np.min(y_data))],
                        bounds=([0, 0, 0], [np.inf, 1, np.inf]),
                        maxfev=10000,
                    )
                    fit_curve_fn = self._fit_func_with_offset
                else:
                    # Standard RB expectation fit without additive offset.
                    fit_parameters, _ = curve_fit(
                        self._fit_func,
                        x_data,
                        y_data,
                        p0=[0.5, 0.5],
                        bounds=([0, 0], [1, 1]),
                    )
                    fit_curve_fn = self._fit_func
            except Exception as exc:
                print(f"Fit failed for p={physical_error_rate}: {exc}")
                fit_parameters = None
                fit_curve_fn = None

            violins = ax.violinplot(violin_data, x_data, showmedians=True)
            for violin_body in violins["bodies"]:
                violin_body.set_facecolor(color)
                violin_body.set_edgecolor(color)
                violin_body.set_alpha(0.45)
            violins["cmedians"].set_color("black")

            for depth_index, depth_values in enumerate(violin_data):
                # Overlay raw points so distribution shape is visible beyond
                # violin summary.
                xs = [x_data[depth_index]] * len(depth_values)
                ax.scatter(xs, depth_values, color=color, alpha=0.4, s=5)

            if fit_parameters is not None and fit_curve_fn is not None:
                x_fit = np.linspace(min(x_data), max(x_data), 100)
                y_fit = fit_curve_fn(x_fit, *fit_parameters)
                ax.plot(x_fit, y_fit, "k-", linewidth=1.5)

                if not fit_with_offset:
                    decay = fit_parameters[1]
                    avg_gate_fidelity = (1 + 2 * decay) / 3
                    ax.text(
                        0.97,
                        0.95,
                        (
                            f"F={avg_gate_fidelity:.6f}\n"
                            f"1-F={1 - avg_gate_fidelity:.6f}"
                        ),
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        bbox=dict(
                            facecolor="white",
                            alpha=0.75),
                    )

            ax.set_xlabel("Circuit depth")
            ax.set_ylabel(y_label)
            ax.set_title(f"p = {physical_error_rate}")
            ax.set_xticks(range(0, max(x_data) + 1, 4))
            ax.set_xlim(-0.1, max(x_data) + 0.5)

        for index in range(num_plots, len(axes_flat)):
            axes_flat[index].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(
            self.output_dir, output_subdir, output_file_name)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")

    def plot_fidelity_curves(self, error_rates: Optional[List[float]] = None):
        """
Plot expectation-value RB curves for each physical error rate.

Args:
    error_rates (Optional[List[float]]): Optional physical-error-rate list.
                                         When omitted, available rates are
                                         inferred from checkpoint files.

Returns:
    None: None. This method writes figure artifacts to disk as its primary side
          effect.

Raises:
    OSError: If output directories/files for plots cannot be created or
             written.
    ValueError: If plotting inputs are inconsistent with expected data shapes
                or keys.
"""
        selected_rates = self.get_error_rates(error_rates)
        self._plot_metric_grid(
            error_rates=selected_rates,
            metric_name="expectation_values",
            title=f"Logical RB Fidelity Curves ({self.experiment_name})",
            y_label="Expectation value",
            color="steelblue",
            output_subdir="Fidelity Curves",
            output_file_name=f"all_fidelity_{self.experiment_name}_curves.png",
            fit_with_offset=False,
        )

    def plot_survival_ratios(self, error_rates: Optional[List[float]] = None):
        """
Plot post-selection survival-ratio curves for each physical error rate.

Args:
    error_rates (Optional[List[float]]): Optional physical-error-rate list.
                                         When omitted, available rates are
                                         inferred from checkpoint files.

Returns:
    None: None. This method writes figure artifacts to disk as its primary side
          effect.

Raises:
    OSError: If output directories/files for plots cannot be created or
             written.
    ValueError: If plotting inputs are inconsistent with expected data shapes
                or keys.
"""
        selected_rates = self.get_error_rates(error_rates)
        self._plot_metric_grid(
            error_rates=selected_rates,
            metric_name="survival_ratios",
            title=f"Survival Ratios ({self.experiment_name})",
            y_label="Survival ratio",
            color="seagreen",
            output_subdir="Survival Probability Curves",
            output_file_name=(
                f"all_survival_probability_{self.experiment_name}_curves.png"
            ),
            fit_with_offset=True,
        )

    def plot_logical_success(self, error_rates: Optional[List[float]] = None):
        """
Plot logical-success curves for each physical error rate.

Args:
    error_rates (Optional[List[float]]): Optional physical-error-rate list.
                                         When omitted, available rates are
                                         inferred from checkpoint files.

Returns:
    None: None. This method writes figure artifacts to disk as its primary side
          effect.

Raises:
    OSError: If output directories/files for plots cannot be created or
             written.
    ValueError: If plotting inputs are inconsistent with expected data shapes
                or keys.
"""
        selected_rates = self.get_error_rates(error_rates)
        self._plot_metric_grid(
            error_rates=selected_rates,
            metric_name="logical_successes",
            title=f"Logical Success ({self.experiment_name})",
            y_label="Logical success",
            color="indianred",
            output_subdir="Logical Success Curves",
            output_file_name=(
                f"all_logical_success_{self.experiment_name}_curves.png"
            ),
            fit_with_offset=False,
        )

    def plot_logical_vs_physical_infidelity(self):
        """
        Plot logical vs physical infidelity and fit a threshold sigmoid in ROI.
        
        Args:
        None: `plot_logical_vs_physical_infidelity` relies on object state and
        accepts no additional inputs.
        
        Returns:
            None: None. This method primarily writes figure artifacts to disk.
        
        Raises:
            OSError: If plot outputs cannot be written to disk.
            ValueError: If plotting inputs have incompatible structure.
        """
        physical_rates, logical_rates = self.load_final_logical_error_curve()
        if not physical_rates or not logical_rates:
            print(
                "No final logical-error curve data found; "
                "skipping threshold plot."
            )
            return

        physical_rates = np.array(physical_rates, dtype=float)
        logical_rates = np.array(logical_rates, dtype=float)

        # Restrict sigmoid fit to a region where threshold-like behavior is
        # expected and data are sufficiently dense.
        roi_start, roi_end = 1e-2, 1e-1
        roi_mask = (physical_rates >= roi_start) & (physical_rates <= roi_end)
        x_roi = physical_rates[roi_mask]
        y_roi = logical_rates[roi_mask]

        fit_parameters = None
        threshold = None

        if len(x_roi) >= 4:
            try:
                fit_parameters, _ = curve_fit(
                    self._sigmoid,
                    x_roi,
                    y_roi,
                    p0=[5e-2, 50, float(max(y_roi) - min(y_roi)),
                        float(min(y_roi))],
                )

                x_dense = np.logspace(
                    np.log10(roi_start), np.log10(roi_end), 1000)
                y_dense = self._sigmoid(x_dense, *fit_parameters)
                index = int(np.argmin(np.abs(y_dense - x_dense)))
                candidate = float(x_dense[index])
                threshold = (
                    candidate
                    if roi_start <= candidate <= roi_end
                    else None
                )
            except Exception as exc:
                print(f"Sigmoid fit failed: {exc}")

        plt.figure(figsize=(10, 8))
        plt.loglog(physical_rates, logical_rates, "o",
                   color="royalblue", markersize=4, label="Data")
        plt.loglog(physical_rates, physical_rates, "r--", label="y = x")

        if fit_parameters is not None:
            x_fit = np.logspace(np.log10(roi_start), np.log10(roi_end), 1000)
            plt.loglog(x_fit, self._sigmoid(x_fit, *fit_parameters),
                       "g-", label="Sigmoid fit")

        if threshold is not None:
            plt.loglog(threshold, threshold, "go", markersize=8,
                       label=f"Threshold ~ {threshold:.3e}")
            plt.axvline(threshold, color="g", linestyle="--", alpha=0.5)
            plt.axhline(threshold, color="g", linestyle="--", alpha=0.5)

        plt.xlabel("Average physical gate infidelity")
        plt.ylabel("Average logical gate infidelity")
        plt.title(f"Logical vs Physical Infidelity ({self.experiment_name})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir,
            f"logical_vs_physical_infidelity_{self.experiment_name}.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")

    @staticmethod
    def _lookup_physical_result_for_p(
            physical_results_dict: dict,
            physical_error_rate: float):
        """
Resolve physical RB payload for a target `p` with tolerant float
matching.

Args:
    physical_results_dict (dict): Dictionary of physical RB payload entries
                                  keyed by physical error rate.
    physical_error_rate (float): Physical depolarizing probability used for the
                                 current simulation or analysis call.

Returns:
    object: Output produced by this routine according to the behavior described
            above (resolve physical rb payload for a target `p` with tolerant
            float).

Raises:
    ValueError: If supplied argument values violate this method's input
                assumptions.
"""
        if physical_error_rate in physical_results_dict:
            return physical_results_dict[physical_error_rate]

        for key in physical_results_dict.keys():
            try:
                if np.isclose(
                        float(key),
                        float(physical_error_rate),
                        rtol=0,
                        atol=1e-15):
                    return physical_results_dict[key]
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _fidelity_from_decay(decay: float, dimension: int = 3) -> float:
        """
        Convert RB decay parameter to average gate fidelity.

        Args:
            decay (float): Exponential RB decay parameter `f`.
            dimension (int): Local Hilbert-space dimension.

        Returns:
            float: Average gate fidelity derived from `f`.
        """
        return float((1 + (dimension - 1) * float(decay)) / dimension)

    def _fit_decay_and_fidelity(
            self,
            x_data: List[float],
            y_data: List[float]) -> Tuple[np.ndarray, float]:
        """
        Fit RB decay model and return fitted parameters with fidelity.

        Args:
            x_data (List[float]): RB depths.
            y_data (List[float]): Mean expectation values at each depth.

        Returns:
            Tuple[np.ndarray, float]: `(fit_params, fidelity)` where
                `fit_params = [A, f]` and fidelity is computed from `f`.
        """
        fit_params, _ = curve_fit(
            self._fit_func,
            x_data,
            y_data,
            p0=[0.5, 0.5],
            bounds=([0, 0], [1, 1]),
        )
        fidelity = self._fidelity_from_decay(float(fit_params[1]))
        return fit_params, fidelity

    def overlay_with_physical_rb(self,
                                 physical_results_path: str,
                                 error_rates: Optional[List[float]] = None,
                                 show_plot: bool = True,
                                 fit_curves: bool = True,
                                 y_range: Optional[
                                     Tuple[float, float]] = None):
        """
Overlay logical RB and physical RB expectation-value curves.

Args:
    physical_results_path (str): Path to the physical-RB `.npy` results payload
                                 used for overlay plots.
    error_rates (Optional[List[float]]): Optional physical-error-rate list.
                                         When omitted, available rates are
                                         inferred from checkpoint files.
    show_plot (bool): If `True`, display the overlay figure with `plt.show()`.
    fit_curves (bool): If `True`, fit exponential RB models and draw fit
                       curves/annotations. If `False`, plot only raw
                       depth-averaged traces.
    y_range (Optional[Tuple[float, float]]): Fixed y-axis range
                                             `(y_min, y_max)`. If `None`,
                                             use adaptive range
                                             `[-0.1, max(y)+0.45]`.

Returns:
    Dict[float, Dict[str, float]]: Per-`p` fidelity summary with
                                   `logical_fidelity` and `physical_fidelity`.

Raises:
    OSError: If output directories/files for plots cannot be created or
             written.
    ValueError: If plotting inputs are inconsistent with expected data shapes
                or keys.
"""
        if not os.path.exists(physical_results_path):
            print(
                f"Physical RB results file not found: {physical_results_path}")
            return {}

        physical_raw = np.load(physical_results_path, allow_pickle=True)
        physical_results = physical_raw.item() if hasattr(
            physical_raw, "item") else physical_raw
        selected_rates = self.get_error_rates(error_rates)
        return self._overlay_from_physical_results(
            physical_results=physical_results,
            error_rates=selected_rates,
            show_plot=show_plot,
            fit_curves=fit_curves,
            y_range=y_range,
        )

    def overlayLogicalPhysicalRB(self,
                                 physicalResults: np.ndarray,
                                 errorRates: List[float],
                                 show_plot: bool = True,
                                 fit_curves: bool = True,
                                 y_range: Optional[
                                     Tuple[float, float]] = None):
        """
        Backward-compatible overlay helper using in-memory physical results.

        Args:
            physicalResults (np.ndarray): Physical RB results payload loaded
                                          from `.npy`.
            errorRates (List[float]): Physical error rates to include.
            show_plot (bool): If `True`, display the figure with `plt.show()`.
            fit_curves (bool): If `True`, fit/plot exponential models; if
                `False`, plot only depth-averaged traces without fit
                overlays.
            y_range (Optional[Tuple[float, float]]): Fixed y-axis range
                `(y_min, y_max)`. If `None`, use adaptive range
                `[-0.1, max(y)+0.45]`.

        Returns:
            Dict[float, Dict[str, float]]: Per-`p` fidelity summary containing
                                           `logical_fidelity` and
                                           `physical_fidelity`.
        """
        physical_results = (
            physicalResults.item()
            if hasattr(physicalResults, "item")
            else physicalResults
        )
        return self._overlay_from_physical_results(
            physical_results=physical_results,
            error_rates=errorRates,
            show_plot=show_plot,
            fit_curves=fit_curves,
            y_range=y_range,
        )

    def overlay_terminal_check_with_physical_rb_no_fit(
            self,
            physical_results_path: str,
            error_rates: Optional[List[float]] = None,
            show_plot: bool = True,
            y_range: Optional[Tuple[float, float]] = None):
        """
Overlay terminal-check logical RB against physical RB without fit models.

This helper is intended for the terminal-check experiment where logical
decay is not expected to follow a clean exponential model. It plots only
depth-averaged expectation traces and intentionally omits fit curves and
fit-derived fidelity annotations.

Args:
    physical_results_path (str): Path to physical RB `.npy` results.
    error_rates (Optional[List[float]]): Optional rate list. If omitted,
                                         rates are inferred from logical
                                         checkpoints.
    show_plot (bool): If `True`, display figure with `plt.show()`.
    y_range (Optional[Tuple[float, float]]): Optional fixed y-limits.

Returns:
    Dict[float, Dict[str, float]]: Per-`p` summary dictionary. Fidelity fields
                                   are `nan` when fitting is disabled.

Raises:
    OSError: If output files cannot be written.
    ValueError: If inputs are inconsistent with expected data schema.
"""
        return self.overlay_with_physical_rb(
            physical_results_path=physical_results_path,
            error_rates=error_rates,
            show_plot=show_plot,
            fit_curves=False,
            y_range=y_range,
        )

    def _overlay_from_physical_results(
            self,
            physical_results: dict,
            error_rates: List[float],
            show_plot: bool = True,
            fit_curves: bool = True,
            y_range: Optional[Tuple[float, float]] = None):
        """
        Internal implementation shared by file-based and in-memory overlays.
        """
        if not error_rates:
            print("No error rates available for overlay plot.")
            return {}

        fidelity_summary: Dict[float, Dict[str, float]] = {}

        num_plots = len(error_rates)
        cols = 3
        rows = (num_plots - 1) // cols + 1

        figure_title = (
            "Logical RB vs Physical RB under Logical Depolarizing Channel"
            if self.experiment_name == "logical_noise"
            else (
                "Terminal-Check Logical RB vs Physical RB "
                "(No Exponential Fits)"
            )
            if self.experiment_name == "terminal_check" and not fit_curves
            else f"Logical vs Physical RB ({self.experiment_name})"
        )
        fig = plt.figure(figsize=(6 * cols, 5 * rows))
        fig.suptitle(figure_title, fontsize=16, y=0.95)

        for index, physical_error_rate in enumerate(error_rates):
            # Build side-by-side logical/physical panel for each p value.
            ax = fig.add_subplot(rows, cols, index + 1)

            logical_file = os.path.join(
                self.logical_checkpoint_dir,
                f"logicalRB_p{physical_error_rate}.pkl")
            logical_checkpoint = self._load_pickle(logical_file)
            logical_x: List[int] = []
            logical_y: List[float] = []
            logical_fit = None
            logical_fidelity = None

            if logical_checkpoint is not None:
                logical_metric = self._get_metric_dict(
                    logical_checkpoint, "expectation_values")
                logical_x, logical_y, _ = self._sorted_depth_data(
                    logical_metric)

                if logical_x:
                    if fit_curves:
                        try:
                            logical_fit, logical_fidelity = (
                                self._fit_decay_and_fidelity(
                                    logical_x, logical_y)
                            )
                        except Exception as exc:
                            print(
                                "Logical fit failed for "
                                f"p={physical_error_rate}: {exc}")
                    ax.plot(
                        logical_x,
                        logical_y,
                        "o-",
                        color="tab:blue",
                        markersize=3,
                        linewidth=1.8,
                        label="Logical LRB",
                    )
                    if fit_curves and logical_fit is not None:
                        # Smooth fitted logical decay curve over plotted range.
                        x_fit = np.linspace(min(logical_x), max(logical_x),
                                            200)
                        y_fit = self._fit_func(x_fit, *logical_fit)
                        ax.plot(
                            x_fit,
                            y_fit,
                            "--",
                            color="tab:blue",
                            linewidth=1.2,
                            label="Logical fit",
                        )

            physical_entry = self._lookup_physical_result_for_p(
                physical_results, physical_error_rate)
            physical_x: List[int] = []
            physical_y: List[float] = []
            physical_fit = None
            physical_fidelity = None

            if physical_entry is not None:
                physical_metric = physical_entry.get(
                    "expectationValues", physical_entry.get(
                        "expectation_values", {}))
                physical_x, physical_y, _ = (
                    self._sorted_depth_data(physical_metric)
                )

                if physical_x:
                    # Always attempt exponential fit directly from plotted
                    # physical data, then fallback to stored parameters.
                    infidelity_raw = physical_entry.get("infidelity")

                    if fit_curves:
                        try:
                            physical_fit, physical_fidelity = (
                                self._fit_decay_and_fidelity(
                                    physical_x,
                                    physical_y,
                                )
                            )
                        except Exception as exc:
                            print(
                                f"Physical fit failed for p="
                                f"{physical_error_rate}: {exc}")
                            fit_params_raw = physical_entry.get(
                                "fitParams", physical_entry.get("fit_params"))
                            if fit_params_raw is not None and len(
                                    fit_params_raw) >= 2:
                                physical_fit = np.array(fit_params_raw,
                                                        dtype=float)
                                physical_fidelity = self._fidelity_from_decay(
                                    float(physical_fit[1]))
                            elif infidelity_raw is not None:
                                physical_fidelity = float(
                                    1.0 - float(infidelity_raw))
                                physical_fit = None
                            else:
                                physical_fit = None
                                physical_fidelity = None

                        if physical_fit is None and infidelity_raw is not None:
                            # Keep fidelity value available even when fit
                            # curve is unavailable.
                            physical_fidelity = float(
                                1.0 - float(infidelity_raw))

                        if physical_fit is None and infidelity_raw is None:
                            try:
                                physical_fit, physical_fidelity = (
                                    self._fit_decay_and_fidelity(
                                        physical_x, physical_y)
                                )
                            except Exception as exc:
                                print(
                                    f"Physical fit failed for p="
                                    f"{physical_error_rate}: {exc}")

                    ax.plot(
                        physical_x,
                        physical_y,
                        "s-",
                        color="tab:red",
                        markersize=3,
                        linewidth=1.8,
                        label="Physical RB",
                    )
                    if fit_curves and physical_fit is not None:
                        # Smooth fitted physical decay curve over
                        # plotted range.
                        x_fit = np.linspace(
                            min(physical_x), max(physical_x), 200)
                        y_fit = self._fit_func(x_fit, *physical_fit)
                        ax.plot(
                            x_fit,
                            y_fit,
                            "--",
                            color="tab:red",
                            linewidth=1.2,
                            label="Physical fit",
                        )

            # Keep full curve domain in view (no zoom/ROI cropping).
            x_min = None
            x_max = None
            if logical_x:
                x_min = (
                    min(logical_x)
                    if x_min is None
                    else min(x_min, min(logical_x))
                )
                x_max = (
                    max(logical_x)
                    if x_max is None
                    else max(x_max, max(logical_x))
                )
            if physical_x:
                x_min = min(physical_x) if x_min is None else min(
                    x_min, min(physical_x))
                x_max = max(physical_x) if x_max is None else max(
                    x_max, max(physical_x))
            if x_min is not None and x_max is not None:
                ax.set_xlim(float(x_min) - 0.1, float(x_max) + 0.5)

            # Fidelity annotation.
            text_lines = []
            if fit_curves:
                if logical_fidelity is not None:
                    text_lines.append(f"F_LRB={logical_fidelity:.16f}")
                if physical_fidelity is not None:
                    text_lines.append(f"F_RB={physical_fidelity:.16f}")
            if text_lines:
                ax.text(
                    0.98,
                    0.96,
                    "\n".join(text_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.78),
                )

            fidelity_summary[float(physical_error_rate)] = {
                "logical_fidelity": (
                    float(logical_fidelity)
                    if logical_fidelity is not None
                    else float("nan")
                ),
                "physical_fidelity": (
                    float(physical_fidelity)
                    if physical_fidelity is not None
                    else float("nan")
                ),
            }

            ax.set_xlabel("Circuit depth")
            ax.set_ylabel("Expectation value")
            ax.set_title(f"p = {physical_error_rate}")
            if y_range is not None:
                # User-specified fixed y range (for consistent panel scaling).
                ax.set_ylim(float(y_range[0]), float(y_range[1]))
            else:
                # Adaptive y range: preserve headroom while avoiding compressed
                # data traces.
                y_candidates = []
                if logical_y:
                    y_candidates.append(float(max(logical_y)))
                if physical_y:
                    y_candidates.append(float(max(physical_y)))
                y_top = (max(y_candidates) + 0.45) if y_candidates else 1.2
                ax.set_ylim(-0.1, y_top)
            ax.set_autoscaley_on(False)
            ax.legend(fontsize="x-small", loc="lower left")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(
            self.output_dir,
            "Fidelity Curves",
            f"logical_vs_physical_rb_{self.experiment_name}.pdf",
        )
        legacy_path = os.path.join(
            self.output_dir,
            "Fidelity Curves",
            f"logicalVsPhysicalRB_{self.experiment_name}_Comparison.pdf",
        )
        png_path = os.path.join(
            self.output_dir,
            "Fidelity Curves",
            f"logical_vs_physical_rb_{self.experiment_name}.png",
        )

        def save_or_fallback(path: str):
            """Save figure to `path`, with fallback name if file is locked."""
            try:
                plt.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved plot: {path}")
                return path
            except PermissionError:
                # Windows/PDF viewers can lock files; save with suffix instead.
                root, ext = os.path.splitext(path)
                fallback = f"{root}_new{ext}"
                plt.savefig(fallback, dpi=300, bbox_inches="tight")
                print(f"Saved plot (fallback): {fallback}")
                return fallback

        save_or_fallback(output_path)

        # Compatibility filename matching legacy plotting output.
        save_or_fallback(legacy_path)
        save_or_fallback(png_path)
        if show_plot:
            plt.show()
        plt.close()
        return fidelity_summary
