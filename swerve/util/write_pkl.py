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

    # Save pickle named fname.meta.pkl with data arrays excluded
    base, ext = os.path.splitext(fname)
    meta_fname = f"{base}.meta{ext}"
    meta = _exclude_arrays(data)
    with open(meta_fname, 'wb') as file_handle:
        logger.info(f"{indent}Writing {meta_fname}")
        pickle.dump(meta, file_handle)


def write_pkl_getstate(fname, data, logger, indent=''):
    """Same as write_pkl, but the meta pickle uses __getstate__ to exclude
    data arrays at pickling time instead of pre-building a filtered copy."""
    from swerve import config

    CONFIG = config()
    fname = os.path.join(CONFIG['dirs']['data'], fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as file_handle:
        logger.info(f"{indent}Writing {fname}")
        pickle.dump(data, file_handle)

    # Save pickle named fname.meta.pkl with data arrays excluded
    base, ext = os.path.splitext(fname)
    meta_fname = f"{base}.meta{ext}"
    with open(meta_fname, 'wb') as file_handle:
        logger.info(f"{indent}Writing {meta_fname}")
        pickle.dump(_MetaDict(data), file_handle)


class _MetaDict(dict):
    """dict that pickles as a plain dict with numpy arrays excluded.

    The state produced by __getstate__ wraps nested dicts in _MetaDict so
    the exclusion is applied recursively as pickle descends the structure.
    Unpickling yields plain dicts.
    """

    def __reduce__(self):
        return (dict, (self.__getstate__(),))

    def __getstate__(self):
        import numpy

        def wrap(value):
            if isinstance(value, dict):
                return _MetaDict(value)
            if isinstance(value, (list, tuple)):
                return type(value)(
                    wrap(item) for item in value
                    if not isinstance(item, numpy.ndarray)
                )
            return value

        return {
            key: wrap(value)
            for key, value in self.items()
            if not isinstance(value, numpy.ndarray)
        }


def _exclude_arrays(obj):
    import numpy

    if isinstance(obj, dict):
        return {
            key: _exclude_arrays(value)
            for key, value in obj.items()
            if not isinstance(value, numpy.ndarray)
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _exclude_arrays(value)
            for value in obj
            if not isinstance(value, numpy.ndarray)
        )
    return obj
