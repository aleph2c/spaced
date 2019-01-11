from collections import deque

class GenExperiment1():

  def __init__(self, maxlen=None):

    self.maxlen = maxlen
    if self.maxlen is None:
      self.maxlen = 100

    # We use deques to limit the upper bound of memory
    # needed for this object.  It is intended to provide
    # infinite scheduling, but we don't have infinite memory
    self.dates = deque(maxlen=self.maxlen)
    self.enclosure_1s = deque(maxlen=self.maxlen)
    self.enclosure_2s = deque(maxlen=self.maxlen)

    self.dates.append(0)
    self.g_schedule = self._g_schedule()

  def latest_date(self):
    return self.dates[-1]

  def enclosure_1(self, val):
    def fn():
      return val + 0.01
    return fn

  def enclosure_2(self, val):
    def fn():
      return val + 0.02
    return fn

  def _g_complicated_fun(self, k=None):
    fn1 = self.enclosure_1(k)
    fn2 = self.enclosure_2(k)
    self.enclosure_1s.append(fn1)
    self.enclosure_2s.append(fn2)
    k = yield k
    k += fn1() + fn2()
    while True:
      fn1 = self.enclosure_1(k)
      fn2 = self.enclosure_2(k)
      k = yield k
      self.enclosure_1s.append(fn1)
      self.enclosure_2s.append(fn2)
      k += fn1() + fn2()

  def _g_schedule(self):
    k = self.latest_date()
    self.g_complicated_fun = self._g_complicated_fun(k)
    k = next(self.g_complicated_fun)
    yield k
    k = self.g_complicated_fun.send(k)

    while True:
      yield k
      self.dates.append(k)
      k = self.g_complicated_fun.send(k)

  def schedule(self, stop, now=None):
    if now is None:
      now = 0

    while stop > self.latest_date():
      next(self.g_schedule)

    return self.dates

  def next_schedule_date(self, now=None):
    if self.latest_data() < now:
      next(self.g_schedule)
    return self.latest_date()

  def query_result(self, when):
    result = None
    index_for_query = None

    for index, date in enumerate(self.dates):
      if date <= when and self.dates[index+1] > when:
        index_for_query = index
        break

    if index_for_query is not None:
      result = self.dates[index_for_query]
      result += self.enclosure_1s[index_for_query]()
      result += self.enclosure_2s[index_for_query]()

    return result

  def reset(self):
    self.g_complicated_fun.close()
    self.g_schedule.close()
    self.dates = deque(maxlen=self.maxlen)
    self.enclosure_1s = deque(maxlen=self.maxlen)
    self.enclosure_2s = deque(maxlen=self.maxlen)
    self.dates.append(0)
    self.g_schedule = self._g_schedule()

if __name__ == '__main__':

  e = GenExperiment1()

  # it should output a schedule
  print(e.schedule(30))

  # there should be one function for each schedule 
  assert len(e.schedule(30)) == len(e.enclosure_1s)

  # we should be able to re-build a value given the saved functions
  fn1_3 = e.enclosure_1s[2]
  fn2_3 = e.enclosure_2s[2]
  result = e.dates[2] + fn1_3() + fn2_3()
  assert e.dates[3] == result

  # we should be able to reset the object and start over
  old_schedule = e.schedule(30)
  e.reset()
  new_schedule = e.schedule(3)
  assert len(new_schedule) < len(old_schedule)
  assert new_schedule[0] == old_schedule[0]
  assert new_schedule[-1] < old_schedule[-1]
  for n_d, o_d in zip(new_schedule, old_schedule):
    assert n_d == o_d

  # our schedule should never exceed it's maxlen
  e = GenExperiment1(maxlen=10)
  print(e.schedule(50000000))
  assert len(e.dates) == 10 
  assert e.dates[0] != 0

  # we should be able to conduct a query if the information is held within the
  # schedule
  e = GenExperiment1(maxlen=10)
  e.schedule(50)
  query_date = e.dates[2] + 0.00000001
  print(e.query_result(query_date))
  assert e.dates[3] == e.query_result(query_date)

  # wrap the deques so we can see if the code will work forever
  for when in range(2, 200):
    e = GenExperiment1(maxlen=4)
    e.schedule(when)
    query_date = e.dates[1] + 0.00000001
    assert abs(e.dates[2] - e.query_result(query_date)) < 0.0001


