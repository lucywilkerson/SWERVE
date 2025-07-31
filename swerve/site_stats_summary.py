def site_stats_summary(stats, logger=None):
  import pandas as pd

  for sid in stats.keys():
    for result in stats[sid].keys():
      if result.startswith('B') and 'calculated' in result and 'metrics' in stats[sid][result]:
        logger.info(f"  {sid}/{result} metrics: {stats[sid][result]['metrics']['pe'][0]:.3f}")
        rows.append({
          'site_id': sid,
          'model': result.split('/')[2],
          'pex': stats[sid][result]['metrics']['pe'][3],
          'ccx': stats[sid][result]['metrics']['cc'][3],
        })

  df = pd.DataFrame(rows)
  print(df)
  models = df['model'].unique()
  for model in models:
    model_df = df[df['model'] == model]
    mean_pex = model_df['pex'].mean()
    mean_ccx = model_df['ccx'].mean()
    mean_pex_se = model_df['pex'].std() / (len(model_df) ** 0.5)
    mean_ccx_se = model_df['ccx'].std() / (len(model_df) ** 0.5)
    logger.info(f"  Model: {model}, n = {len(model_df)}; Mean PE H: {mean_pex:.3f} +/- {mean_pex_se:0.3f}, Mean CC H: {mean_ccx:.3f} +/- {mean_ccx_se:0.3f}")
