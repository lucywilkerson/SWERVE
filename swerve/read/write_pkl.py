import pickle

def _write_pkl(fname, data, logger, indent=''):

  import os
  from swerve import config
  CONFIG = config()
  fname = os.path.join(CONFIG['dirs']['data'], fname)
  if not os.path.exists(os.path.dirname(fname)):
    os.makedirs(os.path.dirname(fname))

  with open(fname, 'wb') as f:
    logger.info(f"{indent}Writing {fname}")
    pickle.dump(data, f)
