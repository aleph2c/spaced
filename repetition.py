# spaced_repetition
import os
import json
import enum
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
from scipy.optimize import curve_fit
from graph import SpaceRepetitionPlot
import matplotlib.animation as animation
from animate import LearningTrackerAnimation

matplotlib.use("Agg")

ppp = pprint.PrettyPrinter(indent=2)
def pp(thing):
  ppp.pprint(thing)

class TimeFormat(enum.Enum):
  OFFSET = 1,
  DATE_TIME = 2

class SpacedKwargInterface():
  def __init__(self, *args, **kwargs):
    if("datetime" in kwargs):
      self.epoch = kwargs['epoch']
    if("range" in kwargs):
      self.range = kwargs['range']
    if("samples" in kwargs):
      self.samples = kwargs['samples']
    if("plasticity_root" in kwargs):
      self.plasticity_root = kwargs['plasticity_root']
    if("plasticity_denominator_offset" in kwargs):
      self.plasticity_denominator_offset = kwargs['plasticity_denominator_offset']
    if("long_term_clamp" in kwargs):
      self.long_term_clamp = kwargs['long_term_clamp']

    # forgetting decay tau
    if("fdecaytau" in kwargs):
      self.fdecaytau = kwargs['fdecaytau']

    # forgetting decay initital value
    if("fdecay0" in kwargs):
      self.fdecay0 = kwargs['fdecay0']

    # initial forgetting offset
    if("ifo" in kwargs):
      self.ifo = kwargs['ifo']

    if("feedback_data" in kwargs):
      self.feedback_data = kwargs['feedback_data']

    # PID control
    if("fdecaytau_kp" in kwargs):
      self.fdecaytau_kp = kwargs['fdecaytau_kp']
    if("fdecaytau_ki" in kwargs):
      self.fdecaytau_ki = kwargs['fdecaytau_ki']
    if("fdecaytau_kd" in kwargs):
      self.fdecaytau_kd = kwargs['fdecaytau_kd']

    if("fdecay0_kp  " in kwargs):
      self.fdecay0_kp = kwargs['fdecay0_kp']
    if("fdecay0_ki  " in kwargs):
      self.fdecay0_ki = kwargs['fdecay0_ki']
    if("fdecay0_kd  " in kwargs):
      self.fdecay0_kd = kwargs['fdecay0_kd']


class SpaceRepetitionDataBuilder():
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

class SpaceRepetition():

  Title           = "Spaced Memory Repetition Strategy\n"
  Horizontal_Axis = ""
  Vertical_Axis   = ""
  Default_Samples = 1000

  def __init__(self, *args, **kwargs):
    self.epoch                              = datetime.now()
    self.datetime                           = None
    self.plot                               = None
    self.range                              = None
    self.ref_events_x                       = []  # recommended events x
    self.ref_events_y                       = []  # recommended events y
    self.a_events_x                         = []  # actual events x
    self.a_events_y                         = []
    self.control_x                          = 0
    self.domain                             = 1
    self.vertical_bars                      = []
    self.x_and_y                            = [[], []]
    self.forgetting_properties              = {}
    self.forgetting_properties["fdecay"]    = {}
    self.forgetting_properties["fdecaytau"] = {}
    self.range                              = SpaceRepetitionReference.Default_Range
    self.plasticity_root                    = SpaceRepetitionReference.PlasticityRoot
    self.plasticity_denominator_offset      = SpaceRepetitionReference.PlasticityDenominatorOffset
    self.samples                            = SpaceRepetitionReference.Default_Samples
    self.fdecaytau                          = SpaceRepetitionReference.Forgetting_Decay_Tau
    self.fdecay0                            = SpaceRepetitionReference.Forgetting_Decay_Initial_Value
    self.ifo                                = SpaceRepetitionReference.Initial_Forgetting_Offset
    self.forgetting_functions               = []

    self.fdecaytau_kp = SpaceRepetitionController.DECAY_TAU_KP
    self.fdecaytau_ki = SpaceRepetitionController.DECAY_TAU_KI
    self.fdecaytau_kd = SpaceRepetitionController.DECAY_TAU_KD
    self.fdecay0_kp   = SpaceRepetitionController.DECAY_INITIAL_VALUE_KP
    self.fdecay0_ki   = SpaceRepetitionController.DECAY_INITIAL_VALUE_KI
    self.fdecay0_kd   = SpaceRepetitionController.DECAY_INITIAL_VALUE_KD

    self.long_term_clamp = SpaceRepetitionController.LONG_TERM_CLAMP

    SpacedKwargInterface.__init__(self, *args, **kwargs)

  def plot_graph(self, **kwargs):
    pass

  def decay_of_decay(self, initial, tau):
    """
    This function returns the ``scale`` values needed to generate the forgetting
    curves.  It returns a function with the values of ``initial`` and ``tau``
    enclosed within it:

    - fn(x): a curve which reaches an asymptote of 0, but is clamped to
      ``-1*self.long_term_clamp`` before it gets there.

    Where ``fn(x)`` is:

    .. code-block:: python

      -1 * initial * e^(-x/tau) clamped to -1*self.long_term_clamp

    ``initial`` determines where to start the curve.  It is negated by the
    function, its value determines at what negative y value to start.

    ``tau`` determines how fast the forgetting curves should transition from a
    fast state of forgetting to a slower state of forgetting.

    .. code-block:: python


              |
              | ---------------------------- self.long_term_clamp -
      -0.0003 |                     ...............................
              |                .....                               
              |             ...                                    
              |           ..                                       
              |          /                                         
              |         /                                          
              |        /                                           
              |       /                                            
      -0.7001 | -----/---------------------------------------------
              |     .                                              
              |                                                    
              |    .                                               
              |                                                    
              |   .                                                
              |                                                    
              |  .                                                 
              |                                                    
         -1.4 | .   
                0                      5                        10

    """

    def fn(x):
      exponential  = -1
      exponential *= x
      exponential /= tau
      result  = initial
      result *= np.exp(exponential)

      # force review roughly every 30 days
      if result < self.long_term_clamp:
        result = self.long_term_clamp
      result *= -1
      return result
    return fn

  def forgetting_curves(self, scale, offset):
    """
    This function returns two other functions which have the values of ``scale``
    and ``offset`` enclosed within them:

    - fn(x):  the falling-exponential forgetting curve profile
    - fn0(x, y): ``y - fn(x)``

    Where ``fn(x)`` is:

    .. code-block:: python

      e^(scale*(x-offset))

    ``scale`` determines how fast the forgetting curve falls

    ``offset`` determines how far to shift the curve to the right

    .. code-block:: python

        scale = -0.7
        offset = 10


        1096 |                                                        
             |                                                        
             |  .                                                     
             |                                                        
             |   .                                                    
             |    .                                                   
             |                                                        
             |     .                                                  
         548 | -----\-------------------------------------------------
             |       \                                                
             |        \                                               
             |         \                                              
             |          ..                                            
             |            \                                           
             |             ...                                        
             |                ...                                     
             |                   .....                                
        1.13 |                        ................................
                  0                      5                          10
    
    """

    def fn(x):
      return np.exp(scale * (x - offset))

    def fn0(x, y):
      """
      result = y - e^(scale*(x-offset))
      
      """
      return y - fn(x) 

    return [fn, fn0]

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
      time = self.epoch
      time += timedelta(seconds=(days * 86400))
    return time

  def plasticity_functions(self, denominator_offset, root):
    """
    This function returns three other functions which have the values of
    ``denominator_offset`` and ``root`` enclosed within them:

    - fn(x): the plasticity curve for the student
    - fn0(x, y): ``y - fn(x)``
    - invfn(y): returns the x value which provides y from ``fn(x)``


    Where ``fn(x)`` is:

    .. code-block:: python

                   x^(1/root)
      ------------------------------------ for x > shift
       (x + denominator_offset)^(1/root)

      0 for x <= shift

      example:
        root = 1.8
        denominator_offset = 1.0


        0.94754 |                          ...........................
                |                 .........                           
                |            .....                                    
                |         ...                                         
                |       ..                                            
                |      /                                              
                |     /                                               
                |    /                                                
        0.47377 | --.-------------------------------------------------
                |                                                     
                |                                                     
                |  .                                                  
                |                                                     
                |                                                     
                |                                                     
                |                                                     
                |                                                     
              0 | .                                                   
                  0                      5                          10

    Where ``fn0(x,y)`` is:

    .. code-block:: python
      
      y - fn(x)

    Where ``invfn(y)`` is:

    .. code-block:: python

                                   y^(root)
      -1 *denominator_offset * --------------- for 0 <= y <= 1
                                  y^(root) - 1
 
      example
        root = 1.8
        denominator_offset = 1.0

        6.89791 |                                                    
                |                                                    
                |                                                    
                |                                                    
                |                                                    
                |        .                                           
                |                                                    
                |         ...                                        
        0.77734 | -----------........................................
                |                                                    
                | ....                                               
                |                                                    
                |     .                                              
                |                                                    
                |                                                    
                |                                                    
                |                                                    
        -5.3432 |      .                                             
                  0                      5                          10
    """
    def is_array(var):
      return isinstance(var, (np.ndarray))

    def fn(x, shift=0):
      with np.errstate(all="ignore"):
        result = np.power(x, 1.0 / root)
        result /= np.power(x + denominator_offset, 1.0 / root)
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

    def fn0(x, y):
      return y - fn(x)

    def invfn(y):
      assert 0 <= y <= 1, "only 0 <= y <=1 supported by this function"
      result = -1 * denominator_offset
      result *= np.power(y, root)
      result /= (np.power(y, root) - 1)
      return result

    return [fn, fn0, invfn]

  def make_stickleback_data_for_plot(self, forgetting_functions, solutions):
    return_set = [[], []]

    solutions = np.array(solutions)
    solutions_left_shifted = np.array([solution - 0.0001 for solution in solutions])
    x_range   = np.array(np.linspace(0, self.range, self.samples))
    x_data = np.concatenate(
      (
        solutions.flatten(),
        x_range.flatten(),
        solutions_left_shifted.flatten()
      ))
    x_data.sort()
    y_data = np.zeros(len(x_data))

    return_set = [x_data, y_data]

    y = np.clip(np.array([forgetting_functions[0](x) for x in x_data]), 0, 1)
    return_set = [x_data, y[:]]
    for fn in forgetting_functions[1:]:
      y_ = [fn(x) for x in x_data]
      y = [0 if y > 1 else y for y in y_]
      overlap_set = [x_data, y[:]]

      for index in range(len(return_set[0])):
        if(return_set[1][index] < overlap_set[1][index]):
          return_set[1][index] = overlap_set[1][index]
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

  def days_from_start(self, datetime_):
    return self.datetime_to_days_offset_from_epoch(datetime_)

  def days_offset_from_epoch_to_datetime(self, offset_):
    result = self.epoch
    result += timedelta(days=offset_)
    return result

class SpaceRepetitionReference(SpaceRepetition):
  Title           = "Spaced Memory Reference\n"
  Horizontal_Axis = ""
  Vertical_Axis   = "recommendation"
  Default_Samples = 50
  Default_Range   = 40

  StickleBackColor          = 'xkcd:orangered'
  LongTermPotentiationColor = 'xkcd:blue'

  # fdecay0
  Forgetting_Decay_Initial_Value = 1.4

  # fdecaytau
  Forgetting_Decay_Tau = 1.2

  # plasticity
  PlasticityRoot = 1.8
  PlasticityDenominatorOffset = 1.0

  # initial forgetting offset
  Initial_Forgetting_Offset = 0.0

  def __init__(self, plot=None, *args, **kwargs):

    # Run our super initializations
    SpaceRepetition.__init__(self, *args, **kwargs)

    self.domain                             = 1
    self.vertical_bars                      = []
    self.x_and_y                            = [[], []]
    self.forgetting_properties              = {}
    self.forgetting_properties["fdecay"]    = {}
    self.forgetting_properties["fdecaytau"] = {}
    self.range                              = SpaceRepetitionReference.Default_Range
    self.plasticity_root                    = SpaceRepetitionReference.PlasticityRoot
    self.plasticity_denominator_offset      = SpaceRepetitionReference.PlasticityDenominatorOffset
    self.samples                            = SpaceRepetitionReference.Default_Samples
    self.fdecaytau                          = SpaceRepetitionReference.Forgetting_Decay_Tau
    self.fdecay0                            = SpaceRepetitionReference.Forgetting_Decay_Initial_Value
    self.forgetting_functions               = []

    SpacedKwargInterface.__init__(self, *args, **kwargs)

    # pc:  plasticity curve
    # pc0:  plasticity curve set to 0
    # ipc: inverse plasticity curve
    self.pc, self.pc0, self.ipc = self.plasticity_functions(
      self.plasticity_denominator_offset,
      self.plasticity_root
    )

    # create the forgetting curves that will ride on
    # top of the plasticity curve, it looks like a stickleback
    self.stickleback(
      fdecaytau=self.fdecaytau,
      fdecay0=self.fdecay0,
      ifo=self.ifo
    )

    if plot is not None and plot is True:
      self.plot_graph()

  def schedule(self):
    schedule = []
    for target_x in self.ref_events_x:
      schedule.append(self.epoch + timedelta(days=target_x))
    return schedule

  def schedule_as_offset_in_days_from_epoch(self):
    return self.ref_events_x

  def datetime_for(self, curve=None):
    '''get the datetime stamp of a curve in the stickleback.  Curves are
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
    end_date += (timedelta(days=self.range))
    result = list(np.arange(datetime_of_curve, end_date,
      timedelta(days=day_step_size)).astype(datetime))
    return result

  def next_lesson(self):
    return self.datetime_for(curve=1)

  def find_nearest(self, array, value):
      idx = (np.abs(array - value)).argmin()
      return array[idx], idx

  # generator needs to be put in here
  def stickleback(self, fdecaytau, fdecay0, ifo=None):

    # ifo: initial forgetting offset
    if ifo is None:
      ifo = 0.0

    # dod: decay of decay
    # describes how the student gets worse at forgetting after a refresh
    self.dod = self.decay_of_decay(fdecay0, fdecaytau)

    # construct our first forgetting curve, they haven't see this information before.
    ffn, ffn0 = self.forgetting_curves(self.dod(ifo), ifo)

    # Our first training schedule is set at time self.ifo
    self.ref_events_x.append(ifo)
    self.forgetting_functions.append(ffn)
    self.ref_events_y.append(1)

    # make an initial guess to help the solver
    self.solution_x, self.solution_y = 1.0, 0.5

    def generate_targets(fn, fn0):

      while True:
        # Solve fn and pc with their given tuning parameters
        def generate_equations(ffn0, fn0_2):
          def equations(xy):
            x, y = xy
            return(ffn0(x, y), fn0_2(x, y))
          return equations
        # link the zero'd form of the forgetting function and the zero'd form of
        # the plasticity curve, this is needed by fsolve to find where these
        # functions intersect 
        equations = generate_equations(fn0, self.pc0)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          # fsolve needs a guess at the answer, use the previous answers as our
          # first guess to find the solutions for our next problem
          solution_x, solution_y = fsolve(equations, (self.solution_x, self.solution_y))

        self.solution_x = solution_x
        self.solution_y = solution_y

        # generate our next forgetting curve
        fn_next, ffn0_next = self.forgetting_curves(self.dod(solution_x), solution_x)

        self.ref_events_x.append(solution_x)
        self.ref_events_y.append(solution_y)

        # save the forgetting function for the ref_events_x identified.  This
        # will be needed to make predictions about a student's ability to
        # recollect something at a given time in the future
        self.forgetting_functions.append(fn_next)
        return [solution_x, fn_next, ffn0_next]

    # get ready to generate the second forgetting curve
    ffn_next, ffn0_next = [ffn, ffn0]

    # generate as many forgetting curves that we need for the time over which we
    # are looking
    while(self.solution_x <= self.range):
      self.solution_x, ffn_next, ffn0_next = generate_targets(ffn_next, ffn0_next)


  def vertical_bar_information(self):
    self.ref_events_x = self.ref_events_x[:]
    self.ref_events_y = self.ref_events_y[:]

    for target_x, target_y in zip(self.ref_events_x, self.ref_events_y):
      self.vertical_bars.append([target_x, target_x])
      self.vertical_bars.append([0, target_y])

  def make_data_for_plot(self):
    self.recollection_x = np.linspace(0, self.range, self.samples)

    for solution_x, solution_y in zip(self.ref_events_x, self.ref_events_y):
      target_x, nearest_index = self.find_nearest(self.recollection_x, solution_x)
      self.recollection_x[nearest_index] = solution_x

    # Draw our first forgetting curve across our schedule
    self.recollection_y = [self.pc(x) for x in self.recollection_x]

    # create the stickleback x and y data that can be plotted
    self.x_and_y = self.make_stickleback_data_for_plot(
      forgetting_functions = self.forgetting_functions,
      solutions=self.ref_events_x
    )

    # parse our ref_events into vertical bar information for graphing
    self.vertical_bar_information()


  def plot_graph(self, epoch=None, plot_pane_data=None, panes=None):

    if epoch is None:
      if self.epoch is None:
        epoch = None
      else:
        epoch = self.epoch

    self.make_data_for_plot()

    x  = self.x_and_y[0]
    y  = self.x_and_y[1]
    rx = self.recollection_x
    ry = self.recollection_y

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

  def save_figure(self, filename="spaced.pdf"):
    plt.savefig(filename, dpi=300)

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
    index = curve - 1
    time_offset_in_days = self.schedule_as_offset_in_days_from_epoch()[index]
    y = recollection_function(moment_as_datetime - timedelta(days=time_offset_in_days))
    return y

  def recollect_function(self, curve=None):
    if curve is None:
      curve = 1

    index = curve - 1
    curves_start_since_epoch = self.schedule_as_offset_in_days_from_epoch()[index]
    forgetting_function = self.forgetting_functions[index]

    if index > 0:
      previous_offset = self.schedule_as_offset_in_days_from_epoch()[index - 1]
    else:
      previous_offset = 0

    def _tuned_after_feedback(moment, forgetting_function, offset, epoch, control_x):
      query_time = self.datetime_to_days_offset_from_epoch(moment)
      recollection_scalar = forgetting_function(query_time + offset)
      return recollection_scalar

    recollection_function = functools.partial(_tuned_after_feedback,
        forgetting_function=forgetting_function,
        offset=curves_start_since_epoch,
        epoch=self.epoch,
        control_x=self.control_x + previous_offset)

    return recollection_function


class SpaceRepetitionFeedback(SpaceRepetition):
  Title = "Spaced Memory Feedback\n"
  Vertical_Axis = "observed"
  LongTermPotentiationColor = 'xkcd:teal'
  StickleBackColor          = 'xkcd:red'

  def __init__(self, *args, **kwargs):
    SpaceRepetition.__init__(self, *args, **kwargs)
    self.range = 0

    self.discovered_plasticity_denominator_offset = \
      self.plasticity_denominator_offset

    self.discovered_plasticity_root = self.plasticity_root

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

    SpacedKwargInterface.__init__(self, *args, **kwargs)

    if("range" in kwargs):
      self.range = kwargs['range']
    else:
      self.range = 10

    self.domain = 1.01
    self.pc, dpo, dpr = self.plasticity_functions()
    self.discovered_plasticity_root = dpr
    self.discovered_plasticity_denominator_offset = dpo

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

    if event_x not in self.a_events_x:
      self.a_events_x.append(c_event_x)
      self.a_events_y.append(event_y)
      if c_event_x > self.range:
        self.range = c_event_x

    self.pc, dpo, dpr = self.plasticity_functions()
    self.discovered_plasticity_denominator_offset = dpo
    self.discovered_plasticity_root = dpr
    return [self.pc, dpo, dpr]

  def fitting_parameters(self, fn, xdata, ydata, weights):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      popt, pcov = curve_fit(fn, xdata, ydata, sigma=weights, method='dogbox',
          bounds=(0, [2.0, 2.0]))
    return [popt, pcov]

  def plasticity_functions(self):
    rx = self.a_events_x
    ry = self.a_events_y
    weights = np.flip(np.linspace(0.1, 1.0, len(rx)))
    if len(weights) >= 1:
      weights[0] = 0.2
      if len(weights) >= 2:
        pass
        weights[-2] = 0.1
        weights[-1] = 0.1

    rparams, rcov = self.fitting_parameters(self.recollection_curve_profile, rx, ry, weights)

    def fn(x):
      return self.recollection_curve_profile(x, *rparams)

    return fn, rparams[0], rparams[1]

  def plot_graph(self, **kwargs):
    observed_events = [[], []]

    pc   = self.pc
    x     = self.a_events_x
    fillx = np.linspace(0, self.range, 50)
    rx    = np.union1d(fillx, x)
    ry    = pc(rx)

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

class ControlData():
  def __init__(self, object_handle, *args, **kwargs):
    self._long_term_potentiation = None
    self._obj = object_handle

    if("pc" in kwargs):
      self._long_term_potentiation_graph = kwargs["pc"]

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

  DECAY_TAU_KP = 0.5
  DECAY_TAU_KI = 0.1
  DECAY_TAU_KD = 0.04

  DECAY_INITIAL_VALUE_KP = 0.5
  DECAY_INITIAL_VALUE_KI = 0.1
  DECAY_INITIAL_VALUE_KD = 0.03

  LONG_TERM_CLAMP = 0.00005

  def __init__(self, *args, **kwargs):

    # Run our super class
    SpaceRepetition.__init__(self, *args, **kwargs)
    self.vertical_bars = []

    self.reference        = ControlData(kwargs['reference'])
    self.reference.fn     = self.reference.o.pc
    self.reference.ipc  = self.reference.o.ipc
    self.reference.range  = self.reference.o.range
    self.reference.domain = self.reference.o.domain
    self.plasticity_root  = self.reference.o.plasticity_root
    self.plasticity_denominator_offset = self.reference.o.plasticity_denominator_offset

    self.feedback         = ControlData(kwargs['feedback'])
    self.feedback.fn      = self.feedback.o.pc
    self.feedback.range   = self.feedback.o.range
    self.feedback.domain  = self.feedback.o.domain
    self.range            = self.feedback.range
    self.input_x          = self.feedback.o.a_events_x[:]
    self.input_y          = self.feedback.o.a_events_y[:]
    self.control_x        = self.input_x[-1]

    SpacedKwargInterface.__init__(self, *args, **kwargs)

    self.discovered_plasticity_root = self.feedback.o.discovered_plasticity_root
    self.discovered_plasticity_denominator_offset = \
      self.feedback.o.discovered_plasticity_denominator_offset

    if("control_x" in kwargs):
      control_x = kwargs["control_x"]
    else:
      control_x = self.input_x[-1]

    # Our PID controller actually don't do a lot, there isn't time in a training
    # series to tune them properly.  They are mosting just proportial scalars
    # which effect two tuning parameters for our updated reference.

    # The majority of our system's response to a student training comes from
    # finding the intersection between our modelled graph of their plasticity
    # and the original reference as a training goal
    self.pid_forgetting_decay_tau = PID(
      kp=self.fdecaytau_kp,
      ki=self.fdecaytau_ki,
      kd=self.fdecaytau_kd)

    self.pid_forgetting_decay_initial_value = PID(
      kp=self.fdecay0_kp,
      ki=self.fdecay0_ki,
      kd=self.fdecay0_kd)
        
    self.control(control_x = control_x)

  def initialize_feedback(self, feedback, *args, **kwargs):
    self.feedback         = ControlData(feedback)
    self.feedback.fn      = self.feedback.o.pc
    self.feedback.range   = self.feedback.o.range
    self.feedback.domain  = self.feedback.o.domain
    self.range            = self.feedback.range
    self.input_x          = self.feedback.o.a_events_x[:]
    self.input_y          = self.feedback.o.a_events_y[:]
    self.control_x        = self.input_x[-1]
    self.control(control_x = self.control_x)

    self.discovered_plasticity_root = feedback.discovered_plasticity_root

    self.discovered_plasticity_denominator_offset = \
      feedback.discovered_plasticity_denominator_offset

    for day in self.updated_reference.ref_events_x:
      self._schedule.append(self.days_to_time(day))


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
    x = np.linspace(0, self.range, 50)
    rx = x[:]

    if(control_x is None):
      control_x = self.input_x[-1]

    self.control_x = control_x
    # If there is not a specified location to control, use the last viewed
    # location
    #
    # We take the x_input location, determine the corresponding y_input
    # then find the corresponding x location on the reference pc function.
    #
    # We shift the pc function on the x axis so that it will intersect where
    # the feedback pc function was at the x_input, then redraw our reference
    # graph at that location, to generate new recommendation times.
    x_reference_feedback_overlap = 0
    y_input                      = self.feedback.fn(control_x)
    x_reference_feedback_overlap = self.reference.ipc(y_input)
    self.x_reference_feedback_overlap = x_reference_feedback_overlap
    x_reference_shift            = control_x - x_reference_feedback_overlap
    self.x_reference_shift       = x_reference_shift

    # What we want is determined by our reference
    # What we are provided, is first used to build a model about the actual
    # plasticity curve, then we take the difference between what we want and
    # what we have been observing and use that as our error signal
    error_y = self.reference.fn(control_x) - self.feedback.fn(control_x)

    # fdecay0, seen it before?  Then pick a lower number (lower is better)
    fdecay0 = self.reference.o.fdecay0
    fdecay0 -= self.pid_forgetting_decay_initial_value.feedback(error_y)
    #try:
    #  # if our student is making mistakes we want to assume they can't learn
    #  # as fast as we want them to learn
    #  print("\noriginal fdecay0 {}".format(self.reference.o.fdecay0))
    #  print("new      fdecay0 {}".format(fdecay0))
    #  print("1: for positive error {}: this diff should be pos: {}".format(error_y, self.fdecay0 - fdecay0))
    #except:
    #  pass
    self.fdecay0 = fdecay0

    # fdecaytau, ability to improve after a lesson, (lower is better)
    fdecaytau = self.reference.o.fdecaytau
    fdecaytau -= self.pid_forgetting_decay_tau.feedback(error_y)
    #try:
    #  # if our student is making mistakes we want to assume they can't learn
    #  # as fast as we want them to learn
    #  print("\noriginal fdecaytau {}".format(self.reference.o.fdecaytau))
    #  print("new      fdecaytau {}".format(fdecaytau))
    #  print("2: for positive error {}: this diff should be pos: {}".format(error_y, self.fdecaytau - fdecaytau))
    #except:
    #  pass
    self.fdecaytau = fdecaytau

    self.updated_reference = SpaceRepetitionReference(
        range=self.range,
        ifo=x_reference_feedback_overlap,
        fdecay0=fdecay0,
        fdecaytau=fdecaytau,
        plasticity_denominator_offset=self.plasticity_denominator_offset,
        plasticity_root=self.plasticity_root,
        long_term_clamp=self.long_term_clamp,
    )
    x_reference_feedback_overlap = self.updated_reference.ipc(y_input)

    # the reference must be shift to the left for positive overlap
    self.x_reference_shift = self.control_x - x_reference_feedback_overlap

    self._schedule_as_offset = [
        ref_as_offset + self.x_reference_shift for
        ref_as_offset in
        self.updated_reference.schedule_as_offset_in_days_from_epoch()]

    self._schedule = [
        self.epoch + timedelta(days=ref_schedule_item) for
        ref_schedule_item in
        self._schedule_as_offset]

  def schedule(self):
    return self._schedule

  def schedule_as_offset_in_days_from_epoch(self):
    return self._schedule_as_offset

  def plot_control(self, **kwargs):
    control_x = self.control_x
    #updated_reference, x_reference_shift = self.control(control_x=control_x)
    x_ref = self.updated_reference.x_and_y[0]
    y_ref = self.updated_reference.x_and_y[1]

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
      x_new = self.x_reference_shift + x_data
      if(x_new <= self.range):
        if(x_new >= control_x):
          control_ref_x.append(x_new)
          control_ref_y.append(y_data)

    self.schedule_vertical_bars = []
    schedule_x = []
    schedule_y = []
    for target_x, target_y in zip(
        self.updated_reference.ref_events_x[1:],
        self.updated_reference.ref_events_y[1:]):
      new_x = target_x + self.x_reference_shift
      if(new_x <= self.range):
        schedule_x.append(new_x)
        schedule_y.append(target_y)
        self.schedule_vertical_bars.append([new_x, new_x])
        self.schedule_vertical_bars.append([0, target_y])

    x = np.linspace(0, self.range, 500)
    rx = x[:]
    ry = self.reference.fn(rx - self.x_reference_shift)
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

    self.updated_reference.make_data_for_plot()
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

  def predict_result(self, moment, curve=None):
    return self.recollect_scalar(moment, curve)

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

    index = curve - 1
    forgetting_function = self.updated_reference.forgetting_functions[index]
    query_time_in_days = self.days_from_start(moment_as_datetime) - self.x_reference_shift
    y = forgetting_function(query_time_in_days)
    return y

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

    if curve is 1:
      datetime_of_curve = self.epoch + timedelta(days=self.control_x)
    else:
      datetime_of_curve = self.datetime_for(curve=curve - 1)
    end_date  = self.epoch + (timedelta(days=self.range))
    result = list(np.arange(datetime_of_curve, end_date,
      timedelta(days=day_step_size)).astype(datetime))
    return result

  def next_lesson(self):
    return self.schedule()[0]

class LearningTracker():
  def __init__(self, *args, **kwargs):
    SpaceRepetition.__init__(self, *args, **kwargs)
    self.data_file     = "test_run.json"
    self.base          = {}
    self.base["frame"] = {}
    SpaceRepetition.__init__(self, *args, **kwargs)

    if not self.epoch:
      self.start_time = datetime.now()
    else:
      self.start_time = self.epoch

    # The d_<name> are used to hang onto the discoveries made through the
    # serialization process.  If you unpickle an object, you can still ask it
    # for it's previous model parameter discovered prior to it being pickled
    if 'd_fdecay0' in kwargs:
      self.d_fdecay0 = kwargs['d_fdecay0']
    else:
      self.d_fdecay0 = self.fdecay0

    if 'd_fdecaytau' in kwargs:
      self.d_fdecaytau = kwargs['d_fdecaytau']
    else:
      self.d_fdecaytau = self.fdecaytau

    if 'd_plasticity_root' in kwargs:
      self.d_plasticity_root = kwargs['d_plasticity_root']
    else:
      self.d_plasticity_root = self.plasticity_root

    if 'd_plasticity_denominator_offset' in kwargs:
      self.d_plasticity_denominator_offset = kwargs['d_plasticity_denominator_offset']
    else:
      self.d_plasticity_denominator_offset = self.plasticity_denominator_offset

    self.plasticity_root = SpaceRepetitionReference.PlasticityRoot
    self.plasticity_denominator_offset = SpaceRepetitionReference.PlasticityDenominatorOffset
    self.samples = SpaceRepetitionReference.Default_Samples
    self.fdecaytau = SpaceRepetitionReference.Forgetting_Decay_Tau
    self.fdecay0 = SpaceRepetitionReference.Forgetting_Decay_Initial_Value
    self.feedback_x = []
    self.feedback_y = []
    self.frames     = []

    SpacedKwargInterface.__init__(self, *args, **kwargs)

    self.reference = SpaceRepetitionReference(
      plot=False,
      epoch=self.epoch,
      range=self.range,
      plasticity_denominator_offset=self.plasticity_denominator_offset,
      plasticity_root=self.plasticity_root,
      fdecay0=self.fdecay0,
      fdecaytau=self.fdecaytau,
      ifo=self.ifo,
      long_term_clamp=self.long_term_clamp,
    )

    self._feedback = SpaceRepetitionFeedback(
      [],
      [],
      range=self.range,
      epoch=self.start_time,
      plasticity_denominator_offset=self.plasticity_denominator_offset,
      plasticity_root=self.plasticity_root,
      long_term_clamp=self.long_term_clamp,
    )

    self.control = SpaceRepetitionController(
      reference=self.reference,
      feedback=self._feedback,
      range=self.range,
      epoch=self.start_time,
      plasticity_root=self.plasticity_root,
      fdecay0=self.fdecay0,
      fdecaytau=self.fdecaytau,
      ifo=self.ifo,
      fdecaytau_kp=self.fdecaytau_kp,
      fdecaytau_ki=self.fdecaytau_ki,
      fdecaytau_kd=self.fdecaytau_kd,
      fdecay0_kp=self.fdecay0_kp,
      fdecay0_ki=self.fdecay0_ki,
      fdecay0_kd=self.fdecay0_kd,
      long_term_clamp=self.long_term_clamp,
    )

    if "feedback_data" in kwargs:
      self.feedback_data = kwargs['feedback_data']
      if self.feedback_data:
        x, y = self.feedback_data
        for x_, y_, in zip(x, y):
          self.add_event(x_, y_)

  def add_event(self, event_x, event_y):
    self.feedback_x.append(event_x)
    self.feedback_y.append(event_y)
    self.frame = np.arange(1, len(self.feedback_x))
    hf = SpaceRepetitionFeedback(self.feedback_x,
        self.feedback_y,
        range=self.range,
        epoch=self.epoch)
    self.control.initialize_feedback(feedback=hf)

  def plot_graphs(self):
    return self.control.plot_graphs()

  def save_figure(self, filename=None):
    self.control.save_figure(filename)

  def schedule(self):
    return self.control.schedule()

  def schedule_as_offset_in_days_from_epoch(self):
    return self.control.schedule_as_offset_in_days_from_epoch()

  def learned(self, when, result):
    moment = when
    if(isinstance(moment, datetime)):
      assert(moment + timedelta(days=0.01) > self.epoch)
      moment_as_datetime = moment
    elif(isinstance(moment, timedelta)):
      moment_as_datetime = self.epoch + timedelta
    elif(isinstance(moment, float) or isinstance(moment, int)):
      moment_as_datetime = self.epoch + timedelta(days=float(moment))
    else:
      raise

    moment_as_offset_in_days = self.control.days_from_epoch(moment_as_datetime)

    assert(0.0 <= result <= 1.0)
    self.add_event(
      moment_as_offset_in_days,
      result)

  def animate(self, name_of_mp4=None, student=None,
      time_per_event_in_seconds=None):
    self.base          = {}
    self.base["frame"] = {}
    range_    = self.feedback_x[-1] + self.feedback_x[-1] * 0.5
    self.base["range"] = range_

    if name_of_mp4 is None:
      self.name_of_mp4 = "animate.mp4"
    else:
      self.name_of_mp4 = name_of_mp4

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
          fdecay0=self.fdecay0,
          fdecaytau=self.fdecaytau,
          plasticity_root=self.plasticity_root,
          plasticity_denominator_offset=self.plasticity_denominator_offset,
          ifo=self.ifo,
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
            plasticity_root=self.plasticity_root,
            plasticity_denominator_offset=self.plasticity_denominator_offset,
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

  def range_for(self, curve=None, range=None, day_step_size=1):
    return self.control.range_for(curve, range, day_step_size)

  def predict_result(self, moment, curve=None):
    return self.recollect_scalar(moment, curve)

  def recollect_scalar(self, moment, curve=None):
    return self.control.recollect_scalar(moment, curve)

  def epoch_fdecaytau(self):
    return self.reference.fdecaytau

  def epoch_fdecay0(self):
    return self.reference.fdecaytau

  def epoch_plasticity_root(self):
    return self.reference.plasticity_root

  def epoch_plasticity_denominator_offset(self):
    return self.reference.plasticity_denominator_offset

  def discovered_fdecaytau(self):
    return self.control.fdecaytau

  def discovered_fdecay0(self):
    return self.control.fdecay0

  def days_from_epoch(self, time):
    return self.control.days_from_epoch(time)

  def discovered_plasticity_root(self):
    return self.control.discovered_plasticity_root

  def discovered_plasticity_denominator_offset(self):
    return self.control.discovered_plasticity_denominator_offset

  def feedback(self, time_format=None):
    results = None

    if time_format is None:
      time_format = TimeFormat.OFFSET

    if time_format is TimeFormat.OFFSET:
      results = self.feedback_x, self.feedback_y
    elif time_format is TimeFormat.DATE_TIME:
      results = [self.epoch + timedelta(days=f) for f in self.feedback_x], self.feedback_y

    return results

  def __getstate__(self):
    '''This magic method over-writes the load, or pickling process, by returning
    a custom dictionary that the pickle process can work with.  Without this
    method, the LearningTracker __dict__ object would be used by the pickler.

    If this method returns false, then the __setstate__ method will not be
    called

    keywords:
      custom pickling
      load loads
    '''
    p_dict = {}
    p_dict['epoch'] = self.start_time
    p_dict['plasticity_root'] = self.plasticity_root
    p_dict['plasticity_denominator_offset'] = self.plasticity_denominator_offset
    p_dict['samples'] = self.samples
    p_dict['fdecay0'] = self.fdecay0
    p_dict['fdecaytau'] = self.fdecaytau
    p_dict['d_fdecaytau'] = self.discovered_fdecaytau()
    p_dict['d_fdecay0'] = self.discovered_fdecay0()
    p_dict['d_plasticity_root'] = self.discovered_plasticity_root()
    p_dict['d_plasticity_denominator_offset'] = self.discovered_plasticity_denominator_offset()
    p_dict['feedback'] = self.feedback()
    return p_dict

  def __setstate__(self, state):
    '''
    Upon unpickling, this method will be called with the unpickled state.
    '''
    self.__init__(
        epoch=state['epoch'],
        plasticity_root=state['plasticity_root'],
        plasticity_denominator_offset=state['plasticity_denominator_offset'],
        fdecay0=state['fdecay0'],
        fdecaytau=state['fdecaytau'],
        feedback_data=state['feedback'],
        d_fdecaytau=state['d_fdecaytau'],
        d_fdecay0=state['d_fdecay0'],
        d_plasticity_root=state['d_plasticity_root'],
        d_plasticity_denominator_offset=state['d_plasticity_denominator_offset'],
    )

