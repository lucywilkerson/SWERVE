def cli(script, defs=False):

  clkws = {
    "event": {
      "help": "ID or pattern for dataset IDs to include (prefix with ^ to use pattern match, e.g., '^A|^B') (default: ^.*)",
      "_used_by_all": True,
    },
    "sites": {
      "help": "",
      "default": None,
      "_used_by": ['main.py']
    },
    "log-level": {
      "help": "Log level",
      "default": 'info',
      "choices": ['debug', 'info', 'warning', 'error', 'critical'],
      "_used_by_all": True
    },
    "debug": {
      "action": "store_true",
      "help": "Same as --log-level debug",
      "default": False,
      "_used_by_all": True
    }

  }

  for key, val in clkws.copy().items():
    keep = '_used_by_all' in val and val['_used_by_all']
    keep = keep or ('_used_by' in val and script in val['_used_by'])
    if not keep:
      del clkws[key]
    if '_used_by_all' in val:
      del val['_used_by_all']
    if '_used_by' in val:
      del val['_used_by']

  if defs:
    return clkws

  import argparse
  parser = argparse.ArgumentParser()
  for k, v in clkws.items():
    parser.add_argument(f'--{k}', **v)

  # Note that hyphens are converted to underscores when parsing
  args = vars(parser.parse_args())

  if args['debug']:
    args['log_level'] = 'debug'
  del args['debug']

  return args
