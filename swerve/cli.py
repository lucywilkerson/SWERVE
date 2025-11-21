def cli(script, defs=False):

  clkws = {
    "event": {
      "help": ""
    },
    "sites": {
      "help": "",
      "default": None
    },
    "log-level": {
      "help": "Log level",
      "default": 'info',
      "choices": ['debug', 'info', 'warning', 'error', 'critical']
    },
    "debug": {
      "action": "store_true",
      "help": "Same as --log-level debug",
      "default": False
    },
    "error-type": {
      "help": "Type of error removed (manual or automated)",
      "default": 'manual'
    }

  }

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
