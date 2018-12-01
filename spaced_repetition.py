# spaced_repetition
import os
import json
import pprint
import warnings
import functools
import matplotlib
import numpy as np
from pid import PID
from graph import ErrorPlot
from functools import reduce
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from collections import OrderedDict
from scipy.optimize import curve_fit
from graph import SpaceRepetitionPlot
import matplotlib.animation as animation
from animate import LearningTrackerAnimation

# from graph import PlotPaneData
# from graph import ErrorPlotFromZero
# from graph import ErrorPlotFromEpoch
# from graph import SpaceRepetitionBasePlotClass
# from graph import SpaceRepetitionPlotDaysFromZero
# from graph import SpaceRepetitionPlotDaysFromEpoch

matplotlib.use("Agg")

ppp = pprint.PrettyPrinter(indent=2)
def pp(thing):
  ppp.pprint(thing)

class QueryTimeBeforeCurve(Exception):
  '''A query time has been made before a forgetting curve has been activated'''
  def __init__(self, moment, curve, curve_start_datetime):
    message = "The query to find p at "
    message += "moment {} for curve {} is too soon, it must occur after {}". \
                 format(moment, curve, curve_start_datetime)
    super().__init__(message)

class SpaceRepetitionDataBuilder(object):
  def __init__(self, *args, **kwargs):
    self.dictionary = {}

  def create_add_x_y_fn_for(self, data_dict, dictionary_name):
    data_dict[dictionary_name] = {}

    def add_x_y(name, list_x, list_y):
      data_dict[dictionary_name][name] = []
      for x_, y_ in zip(list_x, list_y):
        forgetting = {}
        forgetting['x'] = x_
        forgetting['y'] = y_
        data_dict[dictionary_name][name].append(forgetting)

    return add_x_y

  def append_to_base(self, base, data_dict):
    key = list(data_dict.keys())[0]
    if type(data_dict[key]) is dict:
      base[key] = dict(data_dict[key])
      data_dict.clear()
    elif type(data_dict[key]) is list:
      base[key] = data_dict[key][:]
    return base

class SpaceRepetition(object):

  Title           = "Spaced Memory Repetition Strategy\n"
  Horizontal_Axis = ""
  Vertical_Axis   = ""
  Default_Samples = 100

  def __init__(self, *args, **kwargs):
    self.datetime      = None
    self.plot          = None
    self.range         = None
    self.domain        = None
    self.x_and_y       = None
    self.vertical_bars = None
    self.samples       = 0
    self.ref_events_x  = []  # recommended events x
    self.ref_events_y  = []  # recommended events y
    self.a_events_x    = []  # actual events x
    self.a_events_y    = []
    self.control_x     = 0

    if("datetime" in kwargs):
      self.epoch = kwargs['epoch']
    else:
      self.epoch = datetime.now()

  def plot_graph(self, **kwargs):
    pass

  def forgetting_curves(self, scale, offset):
    def fn(x):
      """
      This is the falling exponential forgetting curve
      given a x value, you return a y value """

      return np.exp(scale * (x - offset))

    def fn0(x, y):
      return y - np.exp(scale * (x - offset))

    return [fn, fn0]

  def generate_equations(self, ffn0, fn0_2):
    def equations(xy):
      x, y = xy
      return(ffn0(x, y), fn0_2(x, y))

    return equations

  def days_from_epoch(self, time):

    if isinstance(time, datetime):
      time_since_start  = time
      time_since_start -= self.epoch
      time_in_seconds   = time_since_start.total_seconds()
      c_time            = float(time_in_seconds / 86400.0)
    else:
      raise TypeError("only datetime objects supported")

    c_time = 0 if c_time < 0 else c_time
    return c_time

  def days_to_time(self, days):
    if isinstance(days, datetime):
      raise TypeError("datetime objects not supported")
    else:
      time              = self.epoch
      time             += timedelta(seconds=(days * 86400))
    return time

  def longterm_potentiation_curve(self, factor):
    def is_array(var):
      return isinstance(var, (np.ndarray))

    """
    This is the slow improvement, recollection curve function in a regular and
    zero'd form """
    def fn(x, shift=0):
      with np.errstate(all="ignore"):
        result = np.power(x, 1.0 / factor)
        result /= np.power(x + 1, 1.0 / factor)
      result = np.clip(result, 0, 1)

      if is_array(x):
        for x_index in range(0, len(x)):
          x_item = x[x_index]
          if x_item <= shift:
           result[x_index] = 0
      else:
        if x <= shift:
          result = 0

      return result

    def invfn(y):
      assert 0 <= y <= 1, "only 0 <= y <=1 supported by this function"
      result = -1
      result *= np.power(y, factor)
      result /= (np.power(y, factor) - 1)
      return result

    def fn0(x, y):
      return y - np.power(x, 1.0 / factor) / (np.power(x + 1, 1.0 / factor))

    return [fn, fn0, invfn]

  def forgetting_decay_curve(self, initial, tau):
    """
    This is the forgetting curve function in a regular and
    zero'd form """

    def fn(x):
      exponential  = -1
      exponential *= x
      exponential /= tau
      result  = initial
      result *= np.exp(exponential)
      result *= -1
      return result

    return fn

  def y_data(self, range_x, fn):
    y1 = [fn(x) for x in range_x]
    y1 = [0 if y > 1 else y for y in y1]
    return y1

  def x_data(self, samples, range_=1):
    x1 = np.linspace(0, range_, samples)
    return x1

  def make_data(self, set_, samples, fn, range_, solution_x=None):
    return_set = [[], []]
    range_x2   = self.x_data(samples, range_)

    if solution_x is not None:
      target_x, nearest_index = self.find_nearest(range_x2, solution_x)
      range_x2[nearest_index] = solution_x

    domain_y2  = self.y_data(range_x2, fn)
    set2       = [range_x2, domain_y2]

    if len(set_[0]) == 0:
      empty_x = range_x2[:]
      empty_y = list(map(lambda x: 0, range_x2))
      set_ = [empty_x, empty_y]

    for index in range(len(set_[0])):
      if(set_[1][index] > set2[1][index]):
        return_set[0].append(set_[0][index])
        return_set[1].append(set_[1][index])
      else:
        return_set[0].append(set2[0][index])
        return_set[1].append(set2[1][index])
    return return_set

  def times(self):
    return self.ref_events_x

  def outcome(self, time):
    pass

  def datetime_to_days_offset_from_epoch(self, datetime_):
    '''convert a datetime into a float, representing the number of days
    difference between that datetime and when epoch'''
    result = (datetime_ - self.epoch).total_seconds()
    result /= (60 * 60 * 24)
    return result

  def days_offset_from_epoch_to_datetime(self, offset_):
    result = self.epoch
    result += timedelta(days=offset_)
    return result

class SpaceRepetitionReference(SpaceRepetition):
  Title           = "Spaced Memory Reference\n"
  Horizontal_Axis = ""
  Vertical_Axis   = "recommendation"
  Default_Samples = 1000
  Default_Range   = 40

  StickleBackColor          = 'xkcd:orangered'
  LongTermPotentiationColor = 'xkcd:blue'

  # fdecay0
  Forgetting_Decay_Initial_Value = 1.4
  # fdecaytau
  Forgetting_Decay_Tau           = 1.2
  # plasticity
  PlasticityRate                 = 1.8

  def __init__(self, *args, **kwargs):
    # Run our super class
    SpaceRepetition.__init__(self, *args, **kwargs)

    # initialization and defaults
    self.domain                             = 1
    self.vertical_bars                      = []
    self.x_and_y                            = [[], []]
    self.forgetting_properties              = {}
    self.forgetting_properties["fdecay"]    = {}
    self.forgetting_properties["fdecaytau"] = {}
    self.range                              = SpaceRepetitionReference.Default_Range
    self.plasticity_rate                    = SpaceRepetitionReference.PlasticityRate
    self.samples                            = SpaceRepetitionReference.Default_Samples
    self.forgetting_decay_tau               = SpaceRepetitionReference.Forgetting_Decay_Tau
    self.forgetting_decay_initial_value     = SpaceRepetitionReference.Forgetting_Decay_Initial_Value

    self.forgetting_functions = []

    # Let the caller over-ride some of our default values
    if("range" in kwargs):
      self.range = kwargs['range']
    if("samples" in kwargs):
      self.samples = kwargs['samples']
    if("plasticity" in kwargs):
      self.plasticity_rate = kwargs['plasticity']
    if("fdecaytau" in kwargs):
      self.forgetting_decay_tau = kwargs['fdecaytau']
    if("fdecay0" in kwargs):
      self.forgetting_decay_initial_value = kwargs['fdecay0']

    if("ifo" in kwargs):
      self.initial_forgetting_offset = kwargs['ifo']
    else:
      self.initial_forgetting_offset = 0.0

    self.rfn, self.rfn0, self.invrfn = self.longterm_potentiation_curve(self.plasticity_rate)

    self.stickleback(fdecaytau = self.forgetting_decay_tau,
                     fdecay0   = self.forgetting_decay_initial_value,
                     ifo       = self.initial_forgetting_offset)

    if("plot" in kwargs and kwargs["plot"] is True):
      self.plot_graph()

  def schedule(self):
    schedule = []
    for target_x in self.ref_events_x:
      schedule.append(self.epoch + timedelta(days=target_x))
    return schedule

  def schedule_as_offset_of_days_from_epoch(self):
    return self.ref_events_x

  def datetime_for(self, curve=None):
    '''get the datetime stampe of a curve in the stickleback.  Curves are
    indexed from zero'''
    if curve is None or curve <= 0:
      index = 0
    else:
      index = curve - 1
    schedule_ = self.schedule()
    result  = schedule_[index]
    return result

  def range_for(self, curve=None, range=None, day_step_size=1):
    if curve is None or curve < 1:
      curve = 1
    datetime_of_curve = self.datetime_for(curve=curve)
    end_date  = self.epoch 
    #end_date += (timedelta(days=self.range) - (datetime_of_curve - self.epoch))
    end_date += (timedelta(days=self.range))
    result = list(np.arange(datetime_of_curve, end_date,
      timedelta(days=day_step_size)).astype(datetime))
    return result

  def next_lesson(self):
    return self.datetime_for(curve=1)

  def find_nearest(self, array, value):
      idx = (np.abs(array - value)).argmin()
      return array[idx], idx

  def stickleback(self, **kwargs):

    forgetting_decay_tau           = kwargs['fdecaytau']
    forgetting_decay_initial_value = kwargs['fdecay0']

    if("ifo" in kwargs):
      initial_forgetting_offset = kwargs['ifo']
    else:
      initial_forgetting_offset = 0.0

    # salt away our results
    self.forgetting_properties["fdecaytau"][initial_forgetting_offset] = forgetting_decay_tau
    self.forgetting_properties["fdecay"][initial_forgetting_offset]    = forgetting_decay_initial_value

    # describes how the student gets worse at forgetting after a refresh
    fdfn = self.forgetting_decay_curve(forgetting_decay_initial_value, forgetting_decay_tau)

    # construct our first forgetting curve, they haven't see this information before.
    ffn, ffn0 = self.forgetting_curves(fdfn(initial_forgetting_offset), initial_forgetting_offset)


    # Our first training schedule is set at time self.initial_forgetting_offset
    self.ref_events_x.append(initial_forgetting_offset)
    self.forgetting_functions.append(ffn)
    self.ref_events_y.append(1)
    self.x_and_y = self.make_data(self.x_and_y, self.samples, ffn, self.range)
    # The function used to generate the next set of forgetting curves at their
    # prescribed locations in time

    # create a reasonable x_data range to start with
    self.recollection_x = self.x_data(self.samples, self.range)

    def generate_targets(fn, fn0, fdfn, index):

      # Solve fn and rfn with their given tuning parameters
      equation = self.generate_equations(fn0, self.rfn0)
      solution_x, solution_y = fsolve(equation, (1.0, 1.0))

      # we an intersection between our rfn curve and our latest forgetting
      # curve.
      if(solution_x <= self.range):
        # get an x value in our data close to solution_x
        
        target_x, nearest_index = self.find_nearest(self.recollection_x, solution_x)
        self.recollection_x[nearest_index] = solution_x
        # we want to replace an item in self.recollection_x with our solution,
        # yet we want to keep the same number of samples.


        #self.ref_events_x.append(target_x)
        self.ref_events_x.append(solution_x)
        self.ref_events_y.append(solution_y)

        # generate our next forgetting curve
        fn_next, ffn0_next = self.forgetting_curves(fdfn(index), solution_x)

        # save the forgetting function for the ref_events_x identified.  This
        # will be needed to make predictions about a student's ability to
        # recollect something at a given time in the future
        self.forgetting_functions.append(fn_next)

        # eclipse the previous forgetting curve with our new one
        self.x_and_y = self.make_data(
            self.x_and_y,
            self.samples,
            fn_next,
            self.range,
            solution_x=solution_x)

        # this is where we know the function
      else:
        # if we didn't find a solution our search is complete
        fn_next, ffn0_next = [None, None]
      return [fn_next, ffn0_next]

    # get ready to generate the second forgetting curve
    index, ffn_next, ffn0_next = [1 + initial_forgetting_offset, ffn, ffn0]

    # generate as many forgetting curves that we need for the time over which we
    # are looking
    while(ffn_next is not None):
      ffn_next, ffn0_next = generate_targets(ffn_next, ffn0_next, fdfn, index)
      index += 1

    # Draw our first forgetting curve across our schedule
    self.recollection_y = list(map(lambda x: self.rfn(x), self.recollection_x))

    # parse our ref_events into vertical bar information for graphing
    self.vertical_bar_information()

  def vertical_bar_information(self):
    self.ref_events_x = self.ref_events_x[:]
    self.ref_events_y = self.ref_events_y[:]

    for target_x, target_y in zip(self.ref_events_x, self.ref_events_y):
      self.vertical_bars.append([target_x, target_x])
      self.vertical_bars.append([0, target_y])

  #def schedule(self):
  #  schedule = []
  #  for target_x in self.ref_events_x:
  #    schedule.append(self.datetime + timedelta(days=target_x))
  #  return schedule

  def plot_graph(self, **kwargs):
    x      = self.x_and_y[0]
    y      = self.x_and_y[1]
    rx     = self.recollection_x
    ry     = self.recollection_y
    if("epoch" in kwargs):
      epoch = kwargs['epoch']
    else:
      if self.epoch is None:
        epoch = None
      else:
        epoch = self.epoch

    if("plot_pane_data" in kwargs):
      plot_pane_data = kwargs['plot_pane_data']
    else:
      plot_pane_data = None

    if("panes" in kwargs):
      panes = kwargs['panes']
    else:
      panes = None

    data_dict = {}
    add_x_y = SpaceRepetitionDataBuilder().create_add_x_y_fn_for(data_dict, "recommendation")
    add_x_y("forgetting", x, y)
    add_x_y("long_term_potentiation", rx, ry)
    add_x_y("moments", self.ref_events_x, self.ref_events_y)

    data_args = [rx, ry, x, y]
    data_args += self.vertical_bars
    vertical_bars = {'vertical_bars': self.vertical_bars,
                     'colour': "orange"}
    # reference
    self.plot = SpaceRepetitionPlot(
                    *data_args,
                     title              = SpaceRepetitionReference.Title,
                     x_label            = SpaceRepetition.Horizontal_Axis,
                     y_label            = SpaceRepetitionReference.Vertical_Axis,
                     first_graph_color  = SpaceRepetitionReference.LongTermPotentiationColor,
                     second_graph_color = SpaceRepetitionReference.StickleBackColor,
                     scheduled          = vertical_bars,
                     x_range            = self.range,
                     y_domain           = self.domain + 0.01,
                     epoch              = epoch,
                     panes              = panes,
                     plot_pane_data     = plot_pane_data
                     )

    return self.plot, data_dict

  def show(self):
    plt.show()

  def recollect_scalar(self, moment, curve=None):
    '''
    This will return a scalar representing the probability that a student can
    correctly recall something being tracked by the space repetition algorithm.

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    The ``moment`` must occur after the ``curve`` number has been activated

    '''
    if(isinstance(moment, datetime)):
      assert(moment + timedelta(days=0.01) > self.epoch)
      moment_as_datetime = moment
    elif(isinstance(moment, timedelta)):
      moment_as_datetime = self.epoch + timedelta
    elif(isinstance(moment, float)):
      moment_as_datetime = self.epoch + timedelta(days=moment)
    else:
      raise

    if curve is None:
      curve = 1

    recollection_function = self.recollect_function(curve)
    index = curve-1
    time_offset_in_days = self.schedule_as_offset_of_days_from_epoch()[index]
    curve_start_datetime = self.epoch + timedelta(days=time_offset_in_days)

    y = recollection_function(moment_as_datetime-timedelta(days=time_offset_in_days) )
    #curve_start_datetime = self.epoch + \
    #                       timedelta(self.control_x) + \
    #                       timedelta(days=time_offset_in_days)
         
    #if(moment_as_datetime < curve_start_datetime - timedelta(days=0.01)):
    #  raise QueryTimeBeforeCurve(moment_as_datetime, curve, curve_start_datetime)

    return y

  def recollect_function(self, curve=None):
    if curve is None:
      curve = 1

    index = curve - 1
    curves_start_since_epoch = self.schedule_as_offset_of_days_from_epoch()[index]
    forgetting_function = self.forgetting_functions[index]

    if index > 0:
      previous_offset = self.schedule_as_offset_of_days_from_epoch()[index-1]
    else:
      previous_offset = 0

    def _tuned_after_feedback(moment, forgetting_function, offset, epoch, control_x):
      moment = moment - timedelta(control_x)
      query_time = self.datetime_to_days_offset_from_epoch(moment)
      recollection_scalar = forgetting_function(query_time+offset)
      if recollection_scalar > 1:
        recollection_scalar = 1
      return recollection_scalar

    recollection_function = functools.partial(_tuned_after_feedback,
        forgetting_function=forgetting_function,
        offset=curves_start_since_epoch,
        epoch=self.epoch,
        control_x=self.control_x+previous_offset)

    return recollection_function


class SpaceRepetitionFeedback(SpaceRepetition):
  Title = "Spaced Memory Feedback\n"
  Vertical_Axis = "observed"
  LongTermPotentiationColor = 'xkcd:teal'
  StickleBackColor          = 'xkcd:red'

  def __init__(self, *args, **kwargs):
    SpaceRepetition.__init__(self, *args, **kwargs)
    self.range = 0

    if(len(args[0]) == 0):
      self.add_event(0, 1)
    else:
      for event_x, event_y in zip(args[0], args[1]):
        conditioned_x = None
        if isinstance(event_x, datetime) is False:
          conditioned_x = self.days_offset_from_epoch_to_datetime(event_x)
        else:
          conditioned_x = event_x
        self.add_event(conditioned_x, event_y)

    if("range" in kwargs):
      self.range = kwargs['range']
    else:
      self.range = 10

    self.domain = 1.01
    self.rfn    = self.longterm_potentiation_curve()

  def recollection_curve_profile(self, x, adder_, pdiv_):
    with np.errstate(all="ignore"):
      result  = np.power(x, 1.0 / pdiv_)
      result /= np.power(x + adder_, 1.0 / pdiv_)
    return result

  def recollection_line_profile(self, x, m, b):
    result = m * x + b
    return result

  def add_event(self, event_x, event_y):
    if isinstance(event_x, datetime):
      c_event_x = self.days_from_epoch(event_x)
    else:
      c_event_x = event_x
    self.a_events_x.append(c_event_x)
    self.a_events_y.append(event_y)
    if c_event_x > self.range:
      self.range = c_event_x
    self.rfn = self.longterm_potentiation_curve()

  def fitting_parameters(self, fn, xdata, ydata, weights):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      popt, pcov = curve_fit(fn, xdata, ydata, sigma=weights, method='dogbox')
    return [popt, pcov]

  def longterm_potentiation_curve(self):
    rx = self.a_events_x
    ry = self.a_events_y
    weights = np.linspace(0.1, 1.0, len(rx))
    if len(weights) >= 1:
      weights[0] = 0.2
      if len(weights) >= 2:
        pass
        #weights[-3] = 0.85
        weights[-2] = 0.1
        weights[-1] = 0.1

    rparams, rcov = self.fitting_parameters(self.recollection_curve_profile, rx, ry, weights)

    def fn(x):
      return self.recollection_curve_profile(x, *rparams)

    return fn

  def plot_graph(self, **kwargs):
    observed_events = [[], []]

    rfn   = self.longterm_potentiation_curve()
    x     = self.a_events_x
    fillx = np.linspace(0, self.range, 50)
    rx    = np.union1d(fillx, x)
    ry    = rfn(rx)

    if("epoch" in kwargs):
      epoch = kwargs['epoch']
    else:
      epoch = self.epoch

    if("panes" in kwargs):
      panes = kwargs['panes']
    else:
      panes = 1

    if("plot_pane_data" in kwargs):
      plot_pane_data = kwargs['plot_pane_data']
    else:
      plot_pane_data = None

    vertical_bars = []
    vertical_bars.append([0, 0])
    vertical_bars.append([0, 1])
    observed_events[0].append(self.a_events_x[0])
    observed_events[1].append(self.a_events_y[0])

    for target_x, target_y in zip(self.a_events_x[1:], self.a_events_y[1:]):
      vertical_bars.append([target_x, target_x])
      vertical_bars.append([0, target_y])
      observed_events[0].append(target_x)
      observed_events[1].append(target_y)
      observed_events[0].append(target_x)
      observed_events[1].append(1)

    data_dict = {}
    add_x_y = SpaceRepetitionDataBuilder().create_add_x_y_fn_for(data_dict, "feedback")
    add_x_y("long_term_potentiation", rx, ry)
    add_x_y("moments", self.a_events_x, self.a_events_y)
    add_x_y("forgetting", observed_events[0], observed_events[1])

    data_args = observed_events[:]
    data_args += [rx, ry]
    self.vertical_bars = vertical_bars[:]
    data_args += vertical_bars
    # feedback

    self.plot = SpaceRepetitionPlot(
                    *data_args,
                     x_label=SpaceRepetition.Horizontal_Axis,
                     y_label=SpaceRepetitionFeedback.Vertical_Axis,
                     Second_graph_color=SpaceRepetitionFeedback.LongTermPotentiationColor,
                     first_graph_color=SpaceRepetitionFeedback.StickleBackColor,
                     title=SpaceRepetitionFeedback.Title,
                     x_range=self.range,
                     y_domain=self.domain + 0.01,
                     epoch=epoch,
                     plot_pane_data=plot_pane_data
                     )
    return self.plot, data_dict

  def show(self):
    plt.show()

class ControlData(object):
  def __init__(self, object_handle, *args, **kwargs):
    self._long_term_potentiation = None
    self._obj = object_handle

    if("rfn" in kwargs):
      self._long_term_potentiation_graph = kwargs["rfn"]

  @property
  def long_term_potentiation_graph(self):
    return self._long_term_potentiation_graph

  @long_term_potentiation_graph.setter
  def long_term_potentiation_graph(self, fn):
    self._long_term_potentiation_graph = fn

  @long_term_potentiation_graph.deleter
  def long_term_potentiation_graph(self):
    del self._long_term_potentiation_graph

  @property
  def o(self):
    return self._obj

  @o.setter
  def o(self, object_handle):
    self._obj = object_handle

  @property
  def range(self):
    return self._range

  @range.setter
  def range(self, fn):
    self._range = fn

  @range.deleter
  def range(self):
    del self._range

  @property
  def domain(self):
    return self._domain

  @domain.setter
  def domain(self, fn):
    self._domain = fn

  @domain.deleter
  def domain(self):
    del self._domain

class SpaceRepetitionController(SpaceRepetition):

  def __init__(self, *args, **kwargs):
    # Run our super class
    SpaceRepetition.__init__(self, *args, **kwargs)
    self.vertical_bars = []

    self.reference        = ControlData(kwargs['reference'])
    self.reference.fn     = self.reference.o.rfn
    self.reference.invfn  = self.reference.o.invrfn
    self.reference.range  = self.reference.o.range
    self.reference.domain = self.reference.o.domain

    self.initialize_feedback(kwargs['feedback'])
    #self.feedback         = ControlData(kwargs['feedback'])
    #self.feedback.fn      = self.feedback.o.rfn
    #self.feedback.range   = self.feedback.o.range
    #self.feedback.domain  = self.feedback.o.domain
    #self.range            = self.feedback.range+0.1*(self.feedback.range)
    #self.input_x          = self.feedback.o.a_events_x[:]
    #self.input_y          = self.feedback.o.a_events_y[:]

    if("control_x" in kwargs):
      control_x = kwargs["control_x"]
    else:
      control_x = self.input_x[-1]

    if("range" in kwargs):
      self.range = kwargs['range']

    # not tuned or used yet
    self.pid_forgetting_decay_tau             = PID(1.2, 0.3, 0.01)
    self.pid_forgetting_decay_initial_value   = PID(1.1, 0.1, 0.01)
    self.updated_reference, x_reference_shift = self.control(control_x = control_x)
    self._schedule                            = []

    for day in self.updated_reference.ref_events_x:
      self._schedule.append(self.days_to_time(day))

  def initialize_feedback(self, feedback, *args, **kwargs):
    self.feedback         = ControlData(feedback)
    self.feedback.fn      = self.feedback.o.rfn
    self.feedback.range   = self.feedback.o.range
    self.feedback.domain  = self.feedback.o.domain
    self.range            = self.feedback.range
    self.input_x          = self.feedback.o.a_events_x[:]
    self.input_y          = self.feedback.o.a_events_y[:]
    self.control_x        = self.input_x[-1]

  def error(self, x):
    # a positive error means we need to push harder
    # a negative error means we are pushing too hard
    error = self.reference.fn(x) - self.feedback.fn(x)
    return error

  def get_domain(self, error):
    domain = reduce((lambda y1, y2: y1 if abs(y1) >= abs(y2) else y2), error)
    return domain

  def plot_error(self, **kwargs):
    x1     = np.linspace(0, self.feedback.range, SpaceRepetition.Default_Samples)
    y1     = self.error(x1)
    bars   = self.error_vertical_bars()

    if("epoch" in kwargs):
      epoch = kwargs['epoch']
    else:
      epoch = None

    if("plot_pane_data" in kwargs):
      plot_pane_data = kwargs['plot_pane_data']
    else:
      plot_pane_data = None

    if("panes" in kwargs):
      panes = kwargs['panes']
    else:
      panes = None

    data_dict = {}
    data_dict["error"] = []
    for x_, y_ in zip(x1, y1):
      error = {}
      error['x'] = x_
      error['y'] = y_
      data_dict["error"].append(error)

    # error
    plt = ErrorPlot(x1,
                    y1,
                    *bars,
                    x_range=self.range,
                    y_domain=0.4,
                    y_label="error signal",
                    x_lable="time",
                    epoch=epoch,
                    panes=panes,
                    plot_pane_data=plot_pane_data
                    )

    return self.plot, data_dict

  def error_vertical_bars(self):
    x1 = self.feedback.o.a_events_x
    y1 = self.error(np.array(x1))
    vertical_bars = []
    for target_x, target_y in zip(x1, y1):
      vertical_bar = []
      vertical_bar.append([target_x, target_x])
      vertical_bar.append([0, target_y])
      vertical_bars.append(vertical_bar[:])
    return vertical_bars

  def control(self, control_x=None):
    x = np.linspace(0, self.range, 500)
    rx = x[:]

    if(control_x is None):
      control_x = self.input_x[-1]

    self.control_x = control_x
    # If there is not a specified location to control, use the last viewed
    # location
    #
    # We take the x_input location, determine the corresponding y_input
    # then find the corresponding x location on the reference rfn function.
    #
    # We shift the rfn function on the x axis so that it will intersect where
    # the feedback rfn function was at the x_input, then redraw our reference
    # graph at that location, to generate new recommendation times.
    x_reference_feedback_overlap = 0
    y_input                      = self.feedback.fn(control_x)
    x_reference_feedback_overlap = self.reference.invfn(y_input)
    x_reference_shift            = control_x - x_reference_feedback_overlap
    self.x_reference_shift       = x_reference_shift

    fdecay0         = self.reference.o.forgetting_decay_initial_value
    fdecay0        += self.pid_forgetting_decay_initial_value.feedback(self.error(control_x))
    self.fdecay0    = fdecay0
    fdecaytau       = self.reference.o.forgetting_decay_tau
    fdecaytau      += self.pid_forgetting_decay_tau.feedback(self.error(control_x))
    self.fdecaytau  = fdecaytau
    updated_reference = SpaceRepetitionReference(ifo=x_reference_feedback_overlap,
                                                  fdecay0=fdecay0,
                                                  fdecaytau=fdecaytau)
    self._schedule_as_offset = [ref_as_offset + x_reference_shift for ref_as_offset in updated_reference.schedule_as_offset_of_days_from_epoch()]
    self._schedule = [self.epoch + timedelta(days=ref_schedule_item) for ref_schedule_item in self._schedule_as_offset]
    return [updated_reference, x_reference_shift]

  def schedule(self):
    return self._schedule

  def schedule_as_offset_of_days_from_epoch(self):
    return self._schedule_as_offset

  def plot_control(self, **kwargs):
    control_x = self.control_x
    updated_reference, x_reference_shift = self.control(control_x=control_x)
    x_ref = updated_reference.x_and_y[0]
    y_ref = updated_reference.x_and_y[1]

    if("epoch" in kwargs):
      epoch = kwargs['epoch']
    else:
      epoch = None

    if("panes" in kwargs):
      panes = kwargs['panes']
    else:
      panes = None

    if("plot_pane_data" in kwargs):
      plot_pane_data = kwargs['plot_pane_data']
    else:
      plot_pane_data = None

    control_ref_x = []
    control_ref_y = []
    for x_data, y_data in zip(x_ref, y_ref):
      x_new = x_reference_shift + x_data
      if(x_new <= self.range):
        if(x_new >= control_x):
          control_ref_x.append(x_new)
          control_ref_y.append(y_data)

    self.schedule_vertical_bars = []
    schedule_x = []
    schedule_y = []
    for target_x, target_y in zip(updated_reference.ref_events_x, updated_reference.ref_events_y):
      new_x = target_x + x_reference_shift
      if(new_x <= self.range):
        schedule_x.append(target_x + new_x)
        schedule_y.append(target_y)
        self.schedule_vertical_bars.append([new_x, new_x])
        self.schedule_vertical_bars.append([0, target_y])

    x = np.linspace(0, self.range, 50)
    rx = x[:]
    ry = self.reference.fn(rx - x_reference_shift)
    feedback_x = x[:]
    feedback_y = self.feedback.fn(x)
    data_args = [feedback_x, feedback_y, rx, ry, control_ref_x, control_ref_y]
    data_args += self.feedback.o.vertical_bars
    data_args += self.schedule_vertical_bars
    vertical_bars = {'vertical_bars': self.schedule_vertical_bars, 'colour': "orange"}

    data_dict = {}
    add_x_y = SpaceRepetitionDataBuilder().create_add_x_y_fn_for(data_dict, "control")
    add_x_y("feedback_potentiation", feedback_x, feedback_y)
    add_x_y("reference_potentiation_with_offset", rx, ry)
    add_x_y("reference_forgetting_with_offset", control_ref_x, control_ref_y)
    add_x_y("schedule", schedule_x, schedule_y)
    add_x_y("moments", self.feedback.o.a_events_x, self.feedback.o.a_events_y)

    # control
    self.plot = SpaceRepetitionPlot(*data_args, x_range=self.range, y_domain=1 + 0.01, y_label="control", x_label="",
                  scheduled=vertical_bars,
                  first_graph_color=SpaceRepetitionFeedback.LongTermPotentiationColor,
                  second_graph_color=SpaceRepetitionReference.LongTermPotentiationColor,
                  third_graph_color=SpaceRepetitionReference.StickleBackColor,
                  epoch=epoch,
                  plot_pane_data=plot_pane_data,
                  #range=self.range - x_reference_shift,
                  range=self.range,
                  panes=panes)
    return self.plot, data_dict

  def control_schedule(self):
    pass

  def plot_graphs(self):
    base = {}
    db   = SpaceRepetitionDataBuilder()

    hdl, data_dict = self.reference.o.plot_graph(panes=4, epoch=self.epoch)
    db.append_to_base(base, data_dict)

    h, data_dict = self.feedback.o.plot_graph(epoch=self.epoch, plot_pane_data=hdl.ppd)
    db.append_to_base(base, data_dict)

    h, data_dict = self.plot_error(epoch=self.epoch, plot_pane_data=hdl.ppd)
    db.append_to_base(base, data_dict)

    h, data_dict = self.plot_control(epoch=self.epoch, plot_pane_data=hdl.ppd)
    db.append_to_base(base, data_dict)
    graph_handle = h.ppd
    #graph_handle = hdl.ppd
    return graph_handle, base

  def show(self):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

  def save_figure(self, filename="spaced.pdf"):
    plt.savefig(filename, dpi=300)

  def open_figure(self, filename="spaced.pdf"):
    os.system(filename)

  def recollect_scalar(self, moment, curve=None):
    '''
    This will return a scalar representing the probability that a student can
    correctly recall something being tracked by the space repetition algorithm.

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    The ``moment`` must occur after the ``curve`` number has been activated

    '''
    if(isinstance(moment, datetime)):
      assert(moment + timedelta(days=0.01) > self.epoch)
      moment_as_datetime = moment
    elif(isinstance(moment, timedelta)):
      moment_as_datetime = self.epoch + timedelta
    elif(isinstance(moment, float)):
      moment_as_datetime = self.epoch + timedelta(days=moment)
    else:
      raise

    moment_as_datetime = moment

    if curve is None:
      curve = 1

    recollection_function = self.recollect_function(curve)
    index = curve-1
    time_offset_in_days = self.schedule_as_offset_of_days_from_epoch()[index]
    curve_start_datetime = self.epoch + \
                           timedelta(self.control_x) + \
                           timedelta(days=time_offset_in_days)
    y = recollection_function(moment_as_datetime)
    return y

  def recollect_function(self, curve=None):
    if curve is None:
      curve = 1

    index = curve - 1
    forgetting_function = self.updated_reference.forgetting_functions[index]
    curves_start_since_epoch = \
      self.datetime_to_days_offset_from_epoch(self._schedule[index])

    def _tuned_after_feedback(moment, forgetting_function, offset):
      moment = self.datetime_to_days_offset_from_epoch(moment) - offset
      recollection_scalar = forgetting_function(moment)
      return recollection_scalar

    recollection_function = functools.partial(_tuned_after_feedback,
        forgetting_function=forgetting_function,
        offset=self.x_reference_shift)

    return recollection_function

  def datetime_for(self, curve=None):
    '''get the datetime stampe of a curve in the stickleback.  Curves are
    indexed from zero'''
    if curve is None or curve <= 0:
      index = 0
    else:
      index = curve - 1
    return self.schedule()[index]

  def range_for(self, curve=None, range=None, day_step_size=1):
    if curve is None or curve < 1:
      curve = 1
    datetime_of_curve = self.datetime_for(curve=curve)
    end_date  = self.epoch + (timedelta(days=self.range))
    result = list(np.arange(datetime_of_curve, end_date,
      timedelta(days=day_step_size)).astype(datetime))
    return result

  def next_lesson(self):
    return self.schedule()[0]

class LearningTracker(object):
  def __init__(self, *args, **kwargs):

    self.data_file     = "test_run.json"
    self.base          = {}
    self.base["frame"] = {}
    self.start_time    = datetime.now()

    if("datetime" in kwargs):
      self.epoch = kwargs['epoch']
    else:
      self.epoch = datetime.now()

    if("feedback_data" in kwargs):
      self.feedback_data = kwargs['feedback_data']

    self.plasticity_rate = SpaceRepetitionReference.PlasticityRate
    self.samples = SpaceRepetitionReference.Default_Samples
    self.forgetting_decay_tau = SpaceRepetitionReference.Forgetting_Decay_Tau
    self.forgetting_decay_initial_value = SpaceRepetitionReference.Forgetting_Decay_Initial_Value

    if("plasticity" in kwargs):
      self.plasticity_rate = kwargs['plasticity']
    if("fdecaytau" in kwargs):
      self.forgetting_decay_tau = kwargs['fdecaytau']
    if("fdecay0" in kwargs):
      self.forgetting_decay_initial_value = kwargs['fdecay0']

    if("ifo" in kwargs):
      self.initial_forgetting_offset = kwargs['ifo']
    else:
      self.initial_forgetting_offset = 0.0

    self.reference = SpaceRepetitionReference(
          plot=False,
          epoch=self.epoch,
          plasticity=self.plasticity_rate,
          fdecay0=self.forgetting_decay_initial_value,
          fdecaytau=self.forgetting_decay_tau,
          ifo=self.initial_forgetting_offset,
    )

    self.feedback = SpaceRepetitionFeedback(
          [],
          [],
          epoch=self.start_time)

    self.control = SpaceRepetitionController(
          reference=self.reference,
          feedback=self.feedback,
          epoch=self.start_time)

    self.feedback_x = []
    self.feedback_y = []
    self.frames     = []
    self.control    = []

    if self.feedback_data:
      x, y = self.feedback_data
      for x_, y_, in zip(x, y):
        self.add_event(x_, y_)

  def add_event(self, event_x, event_y):
    self.feedback_x.append(event_x)
    self.feedback_y.append(event_y)
    self.frame = np.arange(1, len(self.feedback_x))

  def animate(self, name_of_mp4=None, student=None,
      time_per_event_in_seconds=None):
    self.base          = {}
    self.base["frame"] = {}
    range_    = self.feedback_x[-1] + self.feedback_x[-1] * 0.5
    self.base["range"] = range_
    #hr        = SpaceRepetitionReference(plot=False, range=range_, epoch=self.start_time)
    #hf        = SpaceRepetitionFeedback(self.feedback_x[0:2], self.feedback_y[0:2], range=range_, epoch=self.start_time)
    #hctrl     = SpaceRepetitionController(reference=hr, feedback=hf, range=range_, epoch=self.start_time)
    #data_dict, _ = hctrl.plot_graphs()

    #self.base["frame"]["0"] = dict(data_dict)
    #data_dict.clear()
    if name_of_mp4 is None:
      self.name_of_mp4 = "animate.mp4"
    else:
      self.name_of_mp4 = name_of_mp4

    #basename = os.path.basename(self.data_file)
    #self.data_file = os.path.splitext(basename)[0] + ".json"

    if student is None:
      self.student = "example"
    else:
      self.student = "student: {}".format(student)

    if time_per_event_in_seconds is not None:
      if time_per_event_in_seconds is 0:
        time_per_event_in_seconds = 1
      fps = 1/time_per_event_in_seconds
      interval = time_per_event_in_seconds * 1000
    else:
      fps = 1
      interval = 1000

    for item in range(0, len(self.feedback_x)):
      hr = SpaceRepetitionReference(
          epoch=self.epoch,
          plasticity=self.plasticity_rate,
          fdecay0=self.forgetting_decay_initial_value,
          fdecaytau=self.forgetting_decay_tau,
          ifo=self.initial_forgetting_offset,
          plot=False,
          range=range_)
      hf = SpaceRepetitionFeedback(
            self.feedback_x[0:item + 1],
            self.feedback_y[0:item + 1],
            range=range_, epoch=self.start_time)
      if item is 0:
        hctrl = SpaceRepetitionController(
            reference=hr,
            feedback=hf,
            range=range_,
            epoch=self.start_time)
      else:
        hctrl.initialize_feedback(feedback=hf)

      graph_handle, data_dict = hctrl.plot_graphs()
      self.base["frame"][str(item)] = dict(data_dict)
    # save some memory
    data_dict.clear()
    graph_handle.close()

    # number from frame
    self.base["frames"] = len(self.feedback_x)
    with open(self.data_file, 'w') as outfile:
      json.dump(self.base, outfile, ensure_ascii=False, sort_keys=True, indent=2)

    lt = LearningTrackerAnimation(json_file=self.data_file)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist=self.student), bitrate=600)
    ani = animation.FuncAnimation(lt.fig, lt.animate, np.arange(0, lt.frames),
        interval=interval, repeat=True)
    ani.save(self.name_of_mp4, writer=writer)

