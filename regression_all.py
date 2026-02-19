import os
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import pickle
from swerve import config, cli, plt_config, savefig, regress, write_eqn_and_fname

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])
results_dir = os.path.join(CONFIG['dirs']['data'], '_results', 'regression')

args = cli('all_plot.py') 
error_type = 'automated_error'

plt_config()

if args['event'] is None or args['event'] == '2024-05-10': # save to 2024-05-10 results dir
    save_results_dir = results_dir

output_name = 'gic_max'

labels = {
    'mag_lat': '\\lambda',
    'mag_lon': 'Magnetic Longitude [deg]',
    'geo_lat': 'Geographic Latitude [deg]',
    'interpolated_beta': '\\beta',
    'log_beta': '\\log_{10} (\\beta)',
    'alpha': '\\alpha',
    'gic_std': '\\sigma_\\text{GIC} (A)',
    'gic_max': '\\vert{\\text{GIC}\\vert_\\text{max}}',
    'mag_lat*mag_lon': 'Mag. Lat. \\cdot Mag. Long.',
    'alpha*interpolated_beta': '\\alpha \\cdot \\beta',
    'mag_lat*interpolated_beta': '\\lambda \\cdot \\beta',
    'slope*alpha*interpolated_beta': '\\text{M} \\cdot \\alpha \\cdot \\beta'
}

fname = CONFIG['files']['regression_results'][output_name]
if os.path.exists(fname):
    with open(fname, 'rb') as f:
        fit_models = pickle.load(f)
else:
    logger.error(f"File {fname} does not exist. Rerun regression.py with reparse=True to create it.")

def plot_regression_all(fit_models, output_name=output_name, save_results_dir=save_results_dir, slope_model=None, all_residual=False):

    plt.figure(figsize=(10,6))
    # Sort events by model slope (coefficient) in descending order
    sorted_events = sorted(fit_models.keys(), key=lambda e: fit_models[e][error_type]['model'].coef_[0] if error_type in fit_models[e] else 0, reverse=True)

    for event_name in sorted_events:
        if error_type not in fit_models[event_name]:
            logger.warning(f"  No {error_type} data for event {event_name}. Skipping.")
            continue
        y = fit_models[event_name][error_type][output_name]
        x = fit_models[event_name][error_type]['alpha_beta']
        model = fit_models[event_name][error_type]['model']
        equation = fit_models[event_name][error_type]['equation']

        if slope_model is not None:
            # Adjust model to have the same slope as slope_model
            slope_calculated = np.exp(slope_model.predict(np.array([np.abs(fit_models[event_name]['min_dst'])]).reshape(1,-1))[0]) # set slope based on min_dst
            model.coef_[0] = (model.coef_[0]/slope_calculated)
            x = x*slope_calculated
            #plt.axline((0, 0), slope=1, color='gray', linestyle=':')
            slope_model_m_text = f'$^{{{slope_model.coef_[0]:+.3g}}}$'
            slope_model_b_text = f'$^{{{slope_model.intercept_:+.3g}}}$'
            plt.text(0.98, -0.15, '$\\text{M} = e$'+slope_model_m_text+'$^{*|\\text{Dst}_\\text{min}|}$'+slope_model_b_text, transform=plt.gca().transAxes, 
                     ha='left', va='bottom', fontsize=20)
            equation = write_eqn_and_fname(['slope*alpha*interpolated_beta'], output_name, model, labels)[0]

        x_range = np.linspace(x.min(),x.max(), 100)
        y_model = model.predict(np.column_stack([x_range]))

        if fit_models[event_name].keys().__contains__('min_dst'): # plotting individual events
            plt.plot(x, y, marker='o', linestyle='', label=f'{event_name} (min Dst={fit_models[event_name]['min_dst']} nT)')
        else:
            plt.plot(x, y, marker='o', linestyle='', label=f'{event_name} data')
        plt.plot(x_range, y_model, linewidth=2, linestyle='--', color=plt.gca().lines[-1].get_color(), label=f'${equation}$')    
    # running regression for all events combined
    if slope_model is not None:
        all_x = np.concatenate([fit_models[event_name][error_type]['alpha_beta']*np.exp(slope_model.predict(np.array([np.abs(fit_models[event_name].get('min_dst', 0))]).reshape(1,-1))[0]) 
                                for event_name in fit_models.keys() if error_type in fit_models[event_name]])
    else:
        all_x = np.concatenate([fit_models[event_name][error_type]['alpha_beta'] for event_name in fit_models.keys() if error_type in fit_models[event_name]])
    all_y = np.concatenate([fit_models[event_name][error_type][output_name] for event_name in fit_models.keys() if error_type in fit_models[event_name]])
    all_model, _, all_metrics = regress(all_x, all_y)
    if slope_model is not None:
        all_equation = write_eqn_and_fname(['slope*alpha*interpolated_beta'], output_name, all_model, labels)[0]
    else:
        all_equation = write_eqn_and_fname(['alpha*interpolated_beta'], output_name, all_model, labels)[0]
    x_range = np.linspace(all_x.min(),all_x.max(), 100)
    y_model = all_model.predict(np.column_stack([x_range]))
    # add linear model for all events combined
    plt.plot(x_range, y_model, color='k', linewidth=3, linestyle='-', label=f'${all_equation}$')
    if all_metrics is not None:
        text = (
          f"r = ${all_metrics['r']:.2f}$ ± ${all_metrics['r_2se']:.2f}$  |  "
          f"RMSE = ${all_metrics['rmse']:.1f}$"
        )
        plt.scatter([], [], facecolors='none', edgecolors='none', label=text) # Adds metrics to legend while keeping legend marker alined w eqn
    plt.xlabel(f'${labels.get('alpha*interpolated_beta', 'alpha*interpolated_beta')}$')
    if slope_model is not None: plt.xlabel(f'$\\text{{M}} \\cdot {labels.get("alpha*interpolated_beta", "alpha*interpolated_beta")}$')
    plt.ylabel(f'${labels.get(output_name, output_name)}$ (A)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    if save_results_dir is not None:
        logger.info(f"Saving all events fit plot to results dir: {save_results_dir}")
        if slope_model is None:
            savefig(save_results_dir, f'all_events_fit_alpha_beta_{output_name}_{error_type}', logger)
        else:
            savefig(save_results_dir, f'all_events_fit_alpha_beta_{output_name}_dst_slope', logger)
    if all_residual:
        def create_residual_plots(plot_type='residuals', event_colors={}):
            plt.figure(figsize=(10,6))
            for event_name in sorted_events:
                if error_type not in fit_models[event_name]:
                    continue
                y = fit_models[event_name][error_type][output_name]
                if slope_model is not None:
                    x = fit_models[event_name][error_type]['alpha_beta']*np.exp(slope_model.predict(np.array([np.abs(fit_models[event_name].get('min_dst', 0))]).reshape(1,-1))[0])
                else: 
                    x = fit_models[event_name][error_type]['alpha_beta']
                predictions = all_model.predict(np.column_stack([x]))
                residuals = y - predictions

                if plot_type == 'residuals':
                    plt.plot(predictions, residuals, marker='o', alpha=.7, linestyle='', label=f'{event_name}')
                    event_colors[event_name] = plt.gca().lines[-1].get_color()  # Store color for this event
                else:  # qq plot
                    from statsmodels.graphics.gofplots import qqplot
                    color = event_colors.get(event_name, 'k')  # Use stored color for this event, default to 'k' if not found
                    qqplot(residuals, line='45', ax=plt.gca(), marker='o', markerfacecolor=color, markeredgecolor=color, alpha=.7, label=f'{event_name}', fit=True)
                    reference_line = plt.gca().lines[-1] 
                    reference_line.set_color('k')
                    reference_line.set_linestyle('--')
            
            if plot_type == 'residuals':
                plt.axhline(0, color='k', linestyle='--', label=f'${all_equation}$')
                plt.xlabel(f'Predicted ${labels.get(output_name, output_name)}$ [A]')
                plt.ylabel('Residuals [A]')
                fname_suffix = 'residuals'
            else:
                plt.xlabel('Theoretical Quantiles')
                plt.ylabel('Sample Quantiles')
                fname_suffix = 'qq'
            
            if all_metrics is not None and plot_type == 'residuals':
                text = (f"r = ${all_metrics['r']:.2f}$ ± ${all_metrics['r_2se']:.2f}$  |  "
                    f"RMSE = ${all_metrics['rmse']:.1f}$")
                plt.scatter([], [], facecolors='none', edgecolors='none', label=text)
            
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            
            if save_results_dir is not None:
                logger.info(f"Saving all events {fname_suffix} plot to results dir: {save_results_dir}")
                slope_suffix = 'dst_slope' if slope_model else f'{error_type}'
                savefig(save_results_dir, f'all_events_{fname_suffix}_alpha_beta_{output_name}_{slope_suffix}', logger)
            plt.close()
            return event_colors
        
        event_colors = create_residual_plots('residuals')
        create_residual_plots('qq', event_colors=event_colors)

# plotting all events regression
plot_regression_all(fit_models, output_name=output_name, save_results_dir=save_results_dir)

# plotting slope vs min Dst
plt.figure(figsize=(10,6))
dst_values = [fit_models[event_name].get('min_dst', 0) for event_name in fit_models.keys()]
slopes = [fit_models[event_name][error_type]['model'].coef_[0] 
            for event_name in fit_models.keys()]
plt.scatter(dst_values, slopes, marker='o', color='k')
for event_name in fit_models.keys():
    dst = fit_models[event_name].get('min_dst', 0)
    plt.annotate(event_name, (dst, fit_models[event_name][error_type]['model'].coef_[0]), 
        xytext=(0, -15), textcoords='offset points', ha='center')
plt.xlabel('min Dst (nT)')
plt.ylabel('Model Slope')
plt.grid(True)
plt.tight_layout()
savefig(save_results_dir, f'all_events_slope_vs_dst_{output_name}_{error_type}', logger)
plt.close()

# create subplots for slope vs each non-error key
non_error_keys = [key for event_name in fit_models.keys() 
            for key in fit_models[event_name].keys() 
            if 'error' not in key.lower() and key not in ['manual_error', 'automated_error']]
non_error_keys = list(set(non_error_keys))  # Remove duplicates
non_error_keys = [k for k in non_error_keys if k != error_type]  # Remove error_type itself

n_plots = len(non_error_keys)
n_cols = 2
n_rows = (n_plots + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
axes = axes.flatten()
vals = {}

for idx, key in enumerate(non_error_keys):
    events_with_key = [(event_name, fit_models[event_name][key]) 
                for event_name in fit_models.keys() 
                if key in fit_models[event_name]]
    if events_with_key:
        values = [val for _, val in events_with_key]
        if all(v <0 for v in values): values = [-v for v in values]  # make all positive for log scale
        if key == 'min_dst' or key == 'min_symh':
            vals[key] = values
        slopes = [fit_models[event_name][error_type]['model'].coef_[0] 
                for event_name, _ in events_with_key]
        axes[idx].scatter(values, slopes, marker='o', color='k')
        for event_name, val in events_with_key:
            axes[idx].annotate(event_name, (np.abs(val), fit_models[event_name][error_type]['model'].coef_[0]), 
                xytext=(0, -15), textcoords='offset points', ha='center')
        axes[idx].set_xlabel(f'|{key}|')
        axes[idx].set_ylabel('Model Slope')
        axes[idx].grid(True)

for idx in range(n_plots, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
savefig(save_results_dir, f'all_events_slope_vs_params_{output_name}_{error_type}', logger)

for ax in axes: # make semilog plot
    ax.set_yscale('log')
    if ax.get_xlabel() == ('|min_dst|') or ax.get_xlabel() == '|min_symh|':
        if 'dst' in ax.get_xlabel().lower():
            x_name = '$\\text{Dst}_\\text{min}$'
            key = 'min_dst'
            save_model = True
        else:
            x_name = '$\\text{SymH}_\\text{min}$'
            key = 'min_symh'
            save_model = False
        x = np.array(vals[key]).reshape((-1,1))
        y = np.array(slopes)
        model,_,metrics = regress(x, np.log(y))
        if save_model:
            slope_model = model
        equation = f'm = exp(${model.coef_[0]:+.3g}$*|'+x_name+f'| ${model.intercept_:+.3g}$)'
        x_range = np.linspace(x.min(),x.max(), 10)
        y_model = np.exp(model.predict(np.column_stack([x_range])))
        ax.plot(x_range, y_model, color='m', linestyle='--', label=equation)
        if metrics is not None:
            text = (f"r = ${metrics['r']:.2f}$ ± ${metrics['r_2se']:.2f}$  |  "
                        f"RMSE = ${metrics['rmse']:.1f}$"
                    )
            ax.scatter([], [], facecolors='none', edgecolors='none', label=text)
        ax.legend()
savefig(save_results_dir, f'all_events_semilog_slope_vs_params_{output_name}_{error_type}', logger)

for ax in axes: # make log-log plot
    if np.any(np.array([val for _, val in events_with_key]) < 0):
        ax.set_xscale('symlog')
    else:
        ax.set_xscale('log')
savefig(save_results_dir, f'all_events_log_slope_vs_params_{output_name}_{error_type}', logger)

plt.close()

# plotting all regression with dst modeled slope
plot_regression_all(fit_models, output_name=output_name, save_results_dir=save_results_dir, slope_model=slope_model, all_residual=True)