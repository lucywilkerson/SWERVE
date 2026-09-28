def _output_error(d, logger):
  msgo = "Not computing modified"
  if 'error' in d:
    logger.error(f"    {msgo} due to error: {d['error']}")
    return True

  if len(d['data'].shape) != 2:
    logger.error(f"    {msgo} b/c data array is not 2D")
    return True

  if len(d['time'].shape) != 1:
    logger.error(f"    {msgo} b/c time array is not 1D")
    return True

  if d['data'].shape[0] != len(d['time']):
    msg = f"    {msgo} b/c d['data'].shape[0] = {d['data'].shape[0]} != len(d['time'])"
    msg += f" = {len(d['time'])}"
    logger.error(msg)
    return True

  return False

