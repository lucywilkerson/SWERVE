import os
import pickle


def write_pkl(fname, data, logger, indent=''):
    from swerve import config

    CONFIG = config()
    fname = os.path.join(CONFIG['dirs']['data'], fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as file_handle:
        logger.info(f"{indent}Writing {fname}")
        pickle.dump(data, file_handle)