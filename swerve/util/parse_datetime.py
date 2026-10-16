def parse_datetime(value):
    import datetime
    if value is None:
      raise ValueError("Datetime value is None")
    if isinstance(value, datetime.datetime):
      return value
    if isinstance(value, datetime.date):
      return datetime.datetime.combine(value, datetime.time.min)

    formats = [
      '%Y-%m-%d/%H:%M:%S',
      '%m/%d/%Y %I:%M:%S %p',
      '%m/%d/%Y %H:%M:%S',
      '%m/%d/%Y %I:%M:%S',
      '%Y-%m-%d %H:%M:%S',
      '%Y/%m/%d %H:%M:%S',
      '%Y-%m-%d %I:%M:%S %p',
      '%Y-%m-%d %H:%M:%S.%f',
    ]

    for fmt in formats:
      try:
        return datetime.datetime.strptime(value, fmt)
      except ValueError:
        continue

    raise ValueError(f"Unrecognized datetime format: {value!r}")