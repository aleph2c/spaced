# -*- coding: utf-8 -*-
"""repetition

The repetition module is the primary interface into the ``spaced`` package.  The
repetition module provides access to a LearningTracker class which can be used
to build student training goals, to track their progress and output a schedule
which reacts to their behaviour.  From this schedule, predictions can be made
about their future performance.

**Example(s)**:
  Here is how you can build a learning tracker, input some student feedback and
  graph its predictions about a student's future performance:

  .. code-block:: python

    from datetime import datetime
    from repetition import LearningTracker

    lt = LearningTracker(epoch=datetime.now())
    for d, r in zip(
      [0,    0.8,  1.75, 3.02, 4.8,  7.33],
      [0.40, 0.44, 0.64, 0.76, 0.83, 0.89],
    ):
      # r: result
      # d: days since training epoch
      lt.learned(result=r, when=d)

    with lt.graphs(
      stop=43,
      show=True,
      control_handle=True,
      filename="module_docstring_example.svg") as ch:
       
        moments = lt.range_for(curve=1, stop=43, day_step_size=0.5)
        predictions = [
          lt.predict_result(moment, curve=1) for moment in moments
        ]
        ch.plot(moments, predictions, color='xkcd:azure')

The ``repetition`` module provides a lot of features which can be read about in `the
full spaced-package documentation <https://aleph2c.github.io/spaced/>`_

"""
import json
import enum
import pprint
import warnings
import matplotlib
import numpy as np
from pid import PID
from graph import ErrorPlot
from collections import deque
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from contextlib import contextmanager
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

  def _create_add_x_y_fn_for(self, data_dict, dictionary_name):
    data_dict[dictionary_name] = {}

    def add_x_y(name, list_x, list_y):
      data_dict[dictionary_name][name] = []
      for x_, y_ in zip(list_x, list_y):
        forgetting = {}
        forgetting['x'] = x_
        forgetting['y'] = y_
        data_dict[dictionary_name][name].append(forgetting)

    return add_x_y

  def _append_to_base(self, base, data_dict):
    key = list(data_dict.keys())[0]
    if type(data_dict[key]) is dict:
      base[key] = dict(data_dict[key])
      data_dict.clear()
    elif type(data_dict[key]) is list:
      base[key] = data_dict[key][:]
    return base

class SpaceRepetition():

  Horizontal_Axis = ""
  Vertical_Axis   = ""
  Default_Samples = 100

  def __init__(self, *args, **kwargs):
    self.epoch         = datetime.now()
    self.datetime      = None
    self.plot          = None
    self.ref_events_x  = []  # recommended events x
    self.ref_events_y  = []  # recommended events y
    self.a_events_x    = []  # actual events x
    self.a_events_y    = []
    self.control_x     = 0
    self.domain        = 1
    self.vertical_bars = []
    self.x_and_y       = [[], []]

    self.forgetting_properties              = {}
    self.forgetting_properties["fdecay"]    = {}
    self.forgetting_properties["fdecaytau"] = {}

    self.fdecaytau_kp = SpaceRepetitionController.DECAY_TAU_KP
    self.fdecaytau_ki = SpaceRepetitionController.DECAY_TAU_KI
    self.fdecaytau_kd = SpaceRepetitionController.DECAY_TAU_KD
    self.fdecay0_kp   = SpaceRepetitionController.DECAY_INITIAL_VALUE_KP
    self.fdecay0_ki   = SpaceRepetitionController.DECAY_INITIAL_VALUE_KI
    self.fdecay0_kd   = SpaceRepetitionController.DECAY_INITIAL_VALUE_KD

    self.plasticity_root               = SpaceRepetitionReference.PlasticityRoot
    self.plasticity_denominator_offset = SpaceRepetitionReference.PlasticityDenominatorOffset
    self.samples                       = SpaceRepetitionReference.Default_Samples
    self.fdecaytau                     = SpaceRepetitionReference.Forgetting_Decay_Tau
    self.fdecay0                       = SpaceRepetitionReference.Forgetting_Decay_Initial_Value
    self.ifo                           = SpaceRepetitionReference.Initial_Forgetting_Offset
    self.forgetting_functions          = []


    self.long_term_clamp = SpaceRepetitionController.LONG_TERM_CLAMP
    SpacedKwargInterface.__init__(self, *args, **kwargs)

  def tune_decay_of_decay(self, initial, tau):
    """
    Returns a tuned-decay_of_decay function.  The decay_of_decay,
    function can be called over and over again with greater values of t, and it
    will return a set of scalars which can be used to build, less and less
    agressive forgetting curves.
   
    To make this function tunable, you can enclose the ``initial`` and ``tau``
    parameters within it.

    **Args**:
      | ``initial`` (float): enclosed parameter
      | ``tau`` (float): enclosed parameter

    **Returns**:
      (function): fn(x), a curve which reaches an asymptote of 0, but is clamped to
      ``-1*self.long_term_clamp`` before it gets there.

    ``fn(x)`` is:

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
    Returns two other functions which have the values of ``scale``
    and ``offset`` enclosed within them.

    **Args**:
      | ``scale`` (int, float): enclosed parameter
      | ``offset`` (int, float): enclosed parameter

    **Returns**:
       (tuple): fn, fn0

    The returned functions are:

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
    '''Converts a datetime object into a number representing the days since the
    training epoch.

    **Args**:
      | ``time`` (int, float, datetime): days since the training epoch

    **Returns**:
       (float): days, as an offset, since the training epoch

    '''
    if isinstance(time, datetime):
      time_since_start  = time
      time_since_start -= self.epoch
      time_in_seconds   = time_since_start.total_seconds()
      c_time            = float(time_in_seconds / 86400.0)
    else:
      n_time = self.days_offset_from_epoch_to_datetime(time)
      c_time = self.days_from_epoch(time=n_time)

    c_time = 0 if c_time < 0 else c_time
    return c_time

  def days_from_start(self, time):
    '''(alias for days_from_epoch) Converts a datetime object into a number
    representing the days since the training epoch.

    **Args**:
      | ``time`` (int, float): days since the training epoch

    **Returns**:
       (float): days since training epoch

    '''
    return self.days_from_epoch(time)  

  def days_to_time(self, days):
    '''Converts a time [offset in days from epoch] into a datetime object.

    **Args**:
      | ``days`` (int, float): days since epoch

    **Returns**:
       (datetime): datetime of epoch plus days
    '''
    if isinstance(days, datetime):
      raise TypeError("datetime objects not supported")
    else:
      time = self.epoch
      time += timedelta(seconds=(days * 86400))
    return time

  def plasticity_functions(self, denominator_offset, root):
    """
    Returns three other functions which have the values of
    ``denominator_offset`` and ``root`` enclosed within them.

    **Args**:
      | ``denominator_offset`` (int, float): enclosed parameter
      | ``root`` (int, float): enclosed parameter

    **Returns**:
      (tuple): fn(x), fn0(x, y), invfn(y)

    These returned functions are:

    - fn(x): the plasticity curve for the student
    - fn0(x, y): ``y - fn(x)``, needed to find intersection with this curve and another
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

  def make_stickleback_data_for_plot(self, forgetting_functions, solutions, stop_day_as_offset):
    '''Creates the data set that can be used to graph the stickleback curve.

    **Args**:
      | ``forgetting_functions`` (list): forgetting_functions
      | ``solutions`` (list): x values where forgetting function intersects the plasticity curve
      | ``stop_day_as_offset`` (float, int): days after epoch to stop construction of stickleback

    **Returns**:
       (list of two lists):  x and y values of stickleback data
    '''
    return_set = [[], []]

    solutions = np.array(solutions)
    solutions_left_shifted = np.array([solution - 0.0001 for solution in solutions])
    x_range = np.array(np.linspace(0, stop_day_as_offset, self.samples))
    x_data = np.concatenate(
      (
        solutions.flatten(),
        x_range.flatten(),
        solutions_left_shifted.flatten()
      ))
    x_data.sort()
    y_data = np.zeros(len(x_data))

    return_set = [x_data, y_data]

    try:
      y = np.clip(np.array([forgetting_functions[0](x) for x in x_data]), 0, 1)
    except:
      import pdb; pdb.set_trace()

    return_set = [x_data, y[:]]
    for fn in forgetting_functions[1:]:
      y_ = [fn(x) for x in x_data]
      y = [0 if y > 1 else y for y in y_]
      overlap_set = [x_data, y[:]]

      for index in range(len(return_set[0])):
        if(return_set[1][index] < overlap_set[1][index]):
          return_set[1][index] = overlap_set[1][index]
    return return_set

  def datetime_to_days_offset_from_epoch(self, datetime_):
    '''Convert a datetime object into a float, representing the number of days
    difference between that datetime and when epoch.

    **Args**:
      | ``datetime_`` (datetime)

    **Returns**:
       (float): datetime_ - datetime of training epoch
    '''
    result = (datetime_ - self.epoch).total_seconds()
    result /= (60 * 60 * 24)
    return result

  def days_offset_from_epoch_to_datetime(self, offset_):
    '''Convert a days-offset-from-epoch into a datetime object.

    **Args**:
      | ``offset_`` (int, float): days since training epoch

    **Returns**:
       (datetime):  datetime of the moment which was ``offset_`` days since the
       training epoch

    '''
    result = self.epoch
    result += timedelta(days=offset_)
    return result

class SpaceRepetitionReference(SpaceRepetition):
  Title           = "Spaced Memory Reference\n"
  """(string): The title of the reference graph"""
  Horizontal_Axis = ""
  """(string): The title of the horizontal axis of the reference graph"""
  Vertical_Axis = "recommendation"
  """(string): The title of the vertical axis of the reference graph"""
  Default_Samples = 50
  """(int): The default number of samples to generate for the reference graph"""

  StickleBackColor = 'xkcd:orangered'
  """(string): The color of the skickleback forgetting curves"""
  LongTermPotentiationColor = 'xkcd:blue'
  """(string): The color of the reference graph's plasticity curve"""

  # fdecay0
  Forgetting_Decay_Initial_Value = 1.4

  # fdecaytau
  Forgetting_Decay_Tau = 1.2

  # plasticity
  PlasticityRoot = 1.8
  PlasticityDenominatorOffset = 1.0

  # initial forgetting offset
  Initial_Forgetting_Offset = 0.0

  Max_Length = 100  # keep a 100 scheduled spots

  def __init__(self, plot=None, *args, **kwargs):

    # Run our super initializations
    SpaceRepetition.__init__(self, *args, **kwargs)

    self.domain                             = 1
    self.vertical_bars                      = []
    self.x_and_y                            = [[], []]
    self.forgetting_properties              = {}
    self.forgetting_properties["fdecay"]    = {}
    self.forgetting_properties["fdecaytau"] = {}
    self.plasticity_root                    = SpaceRepetitionReference.PlasticityRoot
    self.plasticity_denominator_offset      = SpaceRepetitionReference.PlasticityDenominatorOffset
    self.samples                            = SpaceRepetitionReference.Default_Samples
    self.fdecaytau                          = SpaceRepetitionReference.Forgetting_Decay_Tau
    self.fdecay0                            = SpaceRepetitionReference.Forgetting_Decay_Initial_Value
    self.forgetting_functions               = []

    SpacedKwargInterface.__init__(self, *args, **kwargs)

    self.maxlen = SpaceRepetitionReference.Max_Length
    if 'maxlen' in kwargs:
      self.maxlen = kwargs['maxlen']

    # The functions used to hold the forgetting curves are enclosed
    # to keep their initial conditions, and held in a ring buffer (deque)
    # so that the scheduling of this object can happen indefinitely without 
    # causing a memory leak
    self.forgetting_enclosures = deque(maxlen=self.maxlen)
    self.dates_as_day_offsets = deque(maxlen=self.maxlen)
    self.results_at_training_moments = deque(maxlen=self.maxlen)

    # pc:  plasticity curve
    # pc0:  plasticity curve set to 0
    # ipc: inverse plasticity curve
    self.pc, self.pc0, self.ipc = self.plasticity_functions(
      self.plasticity_denominator_offset,
      self.plasticity_root
    )

    # return a generator which can be used to build a stickle back
    self.g_stickleback = self._G_stickleback(
      fdecaytau=self.fdecaytau,
      fdecay0=self.fdecay0,
      ifo=self.ifo
    )

    self.latest_date_offset = 0

    if plot is not None and plot is True:
      self.plot_graph()

  # reference
  def schedule_as_offset(self, stop):
    '''Returns a training schedule as a list of offsets measured in days from the training
    epoch.

    **Args**:
      | ``stop`` (float, int, datetime): at what point do you want the schedule
      information to stop.  If ``stop`` is an int or a float it will be
      interpreted as the number of days-from-epoch at which you would like this
      function to stop providing information. If ``stop`` is a datetime, the
      output schedule will include all of the schedule suggestions up until
      that time.

    **Returns**:
       (list of floats): The schedule as offset from the training epoch

    '''
    if type(stop) is datetime:
      stop_as_offset_in_days = self.days_from_epoch(stop)
    else:
      stop_as_offset_in_days = stop

    while stop_as_offset_in_days > self.latest_date_offset:
      self.latest_date_offset, self.latest_result = next(self.g_stickleback)

    results = [
      offset for
      offset in
      self.dates_as_day_offsets if
      offset < stop_as_offset_in_days]

    return results

  def schedule(self, stop):
    '''Returns a training schedule as of datetimes up until and possibily including
    datetime or offset represented by ``stop``.

    **Args**:
      | ``stop`` (float, int, datetime): at what point do you want the schedule
      information to stop.  If ``stop`` is an int or a float it will be
      interpreted as the number of days-from-epoch at which you would like this
      function to stop providing information. If ``stop`` is a datetime, the
      output schedule will include all of the schedule suggestions up until
      that time.

    **Returns**:
       (list of datetime objects): The schedule

    '''
    if type(stop) is datetime:
      stop_as_offset_in_days = self.days_from_epoch(stop)
    else:
      stop_as_offset_in_days = float(stop)

    schedule_as_offsets_in_days = \
      self.schedule_as_offset(
        stop=stop_as_offset_in_days
      )

    schedule = [
      self.epoch + timedelta(offset) 
        for offset 
          in schedule_as_offsets_in_days
    ]

    return schedule

  # repetition
  def range_for(self, stop, curve=None, day_step_size=1):
    '''Returns a list of useful datetime values for a given curve, up to, but not
    including the stop value, in increments of day_step_size.

    **Args**:
       | ``stop`` (datetime, float, int):  datetime or offset from epoch in days
       | ``curve=None`` (type1): default: which forgetting curve, starting at 1
       | ``day_step_size=1`` (type1): step size of the result (unit of days)

    **Returns**:
       (list): list of datetimes that can be used in another query

    '''
    curve = curve if curve or curve >= 1 else 1

    if curve >= self.maxlen:
      raise Exception('You can not query beyond the bounds of the current ring buffer')

    if type(stop) is datetime:
      stop_day_as_offset = self.days_from_epoch(stop)
    else:
      stop_day_as_offset = float(stop)

    if stop_day_as_offset > self.latest_date_offset:
      self.schedule_as_offset(stop=stop_day_as_offset)

    days_offset_of_curve = self.dates_as_day_offsets[curve-1]

    start_date = \
      self.days_offset_from_epoch_to_datetime(days_offset_of_curve)

    end_date = \
      self.days_offset_from_epoch_to_datetime(stop_day_as_offset)

    result = list(
      np.arange(
        start_date,
        end_date, 
        timedelta(days=day_step_size)
      ).astype(datetime)
    )

    return result

  def _find_nearest(self, array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

  def _G_stickleback(self, fdecaytau, fdecay0, ifo=None):
    '''Returns stickleback generator (co-routine).
    
    It will return a generator function that can be called an infinite number of
    times to build forgetting curves and schedule time offsets and results.  The
    forgetting curves will be stored within the self.forgetting_enclosures deque
    and similarily the schedule time offset and results will be stored within
    the self.dates_as_day_offsets and self.results deques respectively.

    **Args**:
       | ``fdecaytau`` (type1): ``initial`` parameter for ``tune_decay_of_decay``
       | ``fdecay0`` (type1): ``tau`` parameter for ``tune_decay_of_decay``
       | ``ifo=None`` (type1): initial forgetting offset (days from epoch)

    **Returns**:
       (generator): a function which can be called an infinite number of times

    '''
    # ifo: initial forgetting offset
    if ifo is None:
      ifo = 0.0

    # dod: decay of decay
    # describes how the student gets worse at forgetting after a refresh
    self.dod = self.tune_decay_of_decay(fdecay0, fdecaytau)

    # construct our first forgetting curve, they haven't see this information before.
    ffn, ffn0 = self.forgetting_curves(self.dod(ifo), ifo)

    self.forgetting_enclosures.append(ffn)
    self.dates_as_day_offsets.append(ifo)
    self.results_at_training_moments.append(1)

    # make an initial guess to help the solver
    self.solution_x, self.solution_y = 1.0, 0.5

    yield (ifo, 1)
     
    # Solve fn and pc with their given tuning parameters
    def generate_equations(ffn0, fn0_2):
      def equations(xy):
        x, y = xy
        return(ffn0(x, y), fn0_2(x, y))
      return equations

    while True:

      # Place a contract.  If this contract is broken, this code would become
      # extremely difficult to debug for long running learning trackers
      assert len(self.forgetting_enclosures) == len(self.dates_as_day_offsets)
      assert len(self.forgetting_enclosures) == len(self.results_at_training_moments)

      if len(self.forgetting_enclosures) >= self.maxlen:
        self.forgetting_enclosures.popleft()
        self.dates_as_day_offsets.popleft()
        self.results_at_training_moments.popleft()

      # link the zero'd form of the forgetting function and the zero'd form of
      # the plasticity curve, this is needed by fsolve to find where these
      # functions intersect 
      equations = generate_equations(ffn0, self.pc0)

      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # fsolve needs a guess at the answer, use the previous answers as our
        # first guess to find the solutions for our next problem
        solution_x, solution_y = fsolve(equations, (self.solution_x, self.solution_y))

      self.dates_as_day_offsets.append(solution_x)
      self.results_at_training_moments.append(solution_y)

      #self.solution_x = solution_x
      #self.solution_y = solution_y
     
      # Without this function, the ffn function will always just reference the last
      # constructured forgetting curve.  Every item in the forgetting_enclosures
      # will just be a reference the same last constructured forgetting curve.
      # We need them to be unique, so we create local variable, then pass them
      # into the forgetting_enclosures (copy and deepcopy do not work)
      def next_forgetting_curve(solution_x):
        _ffn, _ffn0 = self.forgetting_curves(self.dod(solution_x), solution_x)
        self.forgetting_enclosures.append(_ffn)
        return _ffn0 

      # generate our next forgetting curve
      ffn0 = next_forgetting_curve(solution_x)

      yield (solution_x, solution_y)

  # reference
  def _vertical_bar_information(self):
    self.ref_events_x = list(self.dates_as_day_offsets)[:]
    self.ref_events_y = list(self.results_at_training_moments)[:]

    for target_x, target_y in zip(self.ref_events_x, self.ref_events_y):
      self.vertical_bars.append([target_x, target_x])
      self.vertical_bars.append([0, target_y])

  # reference
  def _make_data_for_plot(self, stop_day_as_offset):

    self.recollection_x = np.linspace(0, stop_day_as_offset, self.samples)

    for solution_x, solution_y in zip(self.dates_as_day_offsets, self.results_at_training_moments):
      target_x, nearest_index = self._find_nearest(self.recollection_x, solution_x)
      self.recollection_x[nearest_index] = solution_x

    # Draw our first forgetting curve across our schedule
    self.recollection_y = [self.pc(x) for x in self.recollection_x]

    # create the stickleback x and y data that can be plotted
    self.x_and_y = self.make_stickleback_data_for_plot(
      forgetting_functions = list(self.forgetting_enclosures),
      solutions=list(self.dates_as_day_offsets),
      stop_day_as_offset=stop_day_as_offset
    )

    # parse our ref_events into vertical bar information for graphing
    self._vertical_bar_information()


  # reference
  def plot_graph(self, stop, plot_pane_data=None, panes=None):
    '''Plots the reference data.

    **Args**:
       | ``stop`` (datetime, float, int): when to stop graphing
       | ``plot_pane_data=None`` (int, float):  plot handle
       | ``panes=None`` (type1): number of sub-plot panes to make

    **Returns**:
       (tuple):  (PlotPaneData, dict)

    '''
    if type(stop) is datetime:
      self.schedule(stop=stop)
      stop_day_as_offset = self.days_from_epoch(stop)
    else:
      self.schedule_as_offset(stop=stop)
      stop_day_as_offset = stop

    self._make_data_for_plot(stop_day_as_offset)

    x  = self.x_and_y[0]
    y  = self.x_and_y[1]
    rx = self.recollection_x
    ry = self.recollection_y

    data_dict = {}
    add_x_y = SpaceRepetitionDataBuilder()._create_add_x_y_fn_for(data_dict, "recommendation")
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
      x_range            = stop_day_as_offset,
      y_domain           = self.domain + 0.01,
      epoch              = self.epoch,
      panes              = panes,
      plot_pane_data     = plot_pane_data
    )
    return self.plot.ppd, data_dict

  # reference
  def save_figure(self, filename="spaced.pdf"):
    '''Saves the file produced by the plot_graph method.

    **Args**:
       | ``filename="spaced.pdf"`` (string): filename
    '''
    plt.savefig(filename, dpi=300)

  # reference
  def show(self):
    '''show the plot after a plot_graph method call'''
    plt.show()

  # reference
  def recollect_scalar(self, moment, curve=None):
    '''
    Provides a recollection prediction at a provided moment after a given
    schedule point.

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    The ``moment`` must occur after the ``curve`` number has been activated

    '''
    if(isinstance(moment, datetime)):
      assert moment + timedelta(days=0.01) > self.epoch
      moment_as_offset_in_days = self.datetime_to_days_offset_from_epoch(moment)
    else:
      assert moment >= 0
      moment_as_offset_in_days = moment

    self.schedule_as_offset(stop=moment_as_offset_in_days)

    if curve is None:
      curve = 1

    forgetting_function = self.forgetting_enclosures[curve-1]

    return forgetting_function(moment_as_offset_in_days)

  # reference
  def predict_result(self, moment, curve=None):
    '''
    (alias for recollect_scalar) Provides a recollection prediction at a
    provided moment after a given schedule point.

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    The ``moment`` must occur after the ``curve`` number has been activated

    '''
    return self.recollect_scalar(moment, curve)

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

    # assume the have looked at it before they try to remember the
    # idea through a self examination
    self.add_event(0, 1)

    for feedback_moment, result in zip(args[0], args[1]):
      conditioned_x = None
      if isinstance(feedback_moment, datetime) is False:
        conditioned_x = self.days_offset_from_epoch_to_datetime(feedback_moment)
      else:
        conditioned_x = feedback_moment
      self.add_event(conditioned_x, result)

    SpacedKwargInterface.__init__(self, *args, **kwargs)

    if("range" in kwargs):
      self.range = kwargs['range']
    else:
      self.range = 10

    self.domain = 1.01
    self.pc, dpo, dpr = self.plasticity_functions()
    self.discovered_plasticity_root = dpr
    self.discovered_plasticity_denominator_offset = dpo

  def _recollection_curve_profile(self, x, adder_, pdiv_):
    with np.errstate(all="ignore"):
      result  = np.power(x, 1.0 / pdiv_)
      result /= np.power(x + adder_, 1.0 / pdiv_)
    return result

  def add_event(self, feedback_moment, result):
    '''Adds a feedback moment to the feedback dataset.

    **Args**:
       | ``feedback_moment`` (int, float, datetime): the self examination moment
       | ``result`` (type1): how well they did 0.0 to  1,0

    **Returns**:
       (list)  [plasticity curve, ``discovered_plasticity_root``,
       ``discovered_plasticity_denominator_offset``]

       * fn: the plasticity curve that matches what the student is actually doing
       * ``discovered_plasticity_root``:  can be fed into
       ``plasticity_functions`` of the reference object to make a reference
       plasticty curve that matches this one.
       * ``discovered_plasticity_denominator_offset``: can be fed into the
       ``plasticity_functions`` of the reference object to make a reference
       plasticity curve that matches this one.
    '''
    if isinstance(feedback_moment, datetime):
      c_feedback_moment = self.days_from_epoch(feedback_moment)
    else:
      c_feedback_moment = feedback_moment

    if feedback_moment not in self.a_events_x:
      self.a_events_x.append(c_feedback_moment)
      self.a_events_y.append(result)
      if c_feedback_moment > self.range:
        self.range = c_feedback_moment

    # sort the times and results of a_events_x and a_events_y together
    s = sorted(zip(self.a_events_x, self.a_events_y))
    self.a_events_x, self.a_events_y = map(list, zip(*s))

    self.pc, dpo, dpr = self.plasticity_functions()
    self.discovered_plasticity_denominator_offset = dpo
    self.discovered_plasticity_root = dpr
    return [self.pc, dpo, dpr]

  def _fitting_parameters(self, fn, xdata, ydata, weights):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      popt, pcov = curve_fit(fn, xdata, ydata, sigma=weights, method='dogbox',
          bounds=(0, [2.0, 2.0]))
    return [popt, pcov]

  # feedback
  def plasticity_functions(self):
    '''Returns the observed plasticity function and two discovered plasticity parameters. 

    **Returns**:
       (list): fn, discovered_plasticity_root, discovered_plasticity_denominator_offset

       * fn: the plasticity curve that matches what the student is actually doing
       * ``discovered_plasticity_root``:  can be fed into
       ``plasticity_functions`` of the reference object to make a reference
       plasticty curve that matches this one.
       * ``discovered_plasticity_denominator_offset``: can be fed into the
       ``plasticity_functions`` of the reference object to make a reference
       plasticity curve that matches this one.

    '''
    rx = self.a_events_x
    ry = self.a_events_y
    weights = np.flip(np.linspace(0.1, 1.0, len(rx)))
    if len(weights) >= 1:
      weights[0] = 0.2
      if len(weights) >= 2:
        pass
        weights[-2] = 0.1
        weights[-1] = 0.1

    rparams, rcov = self._fitting_parameters(self._recollection_curve_profile, rx, ry, weights)

    def fn(x):
      return self._recollection_curve_profile(x, *rparams)

    return fn, rparams[0], rparams[1]

  # feedback
  def plot_graph(self, plot_pane_data=None, panes=None, stop=None):
    '''Plots the feedback data.

    **Args**:
       | ``stop`` (datetime, float, int): when to stop graphing
       | ``plot_pane_data=None`` (int, float):  plot handle
       | ``panes=None`` (type1): number of sub-plot panes to make

    **Returns**:
       (type):  SpaceRepetitionPlot, dict

    '''
    observed_events = [[], []]

    if panes is None:
      panes = 1

    if stop is None:
      stop = self.range

    pc = self.pc
    x = [item for item in self.a_events_x if item <= stop]
    fillx = np.linspace(0, stop, 50)
    rx = np.union1d(fillx, x)
    ry = pc(rx)
    
    vertical_bars = []
    vertical_bars.append([0, 0])
    vertical_bars.append([0, 1])
    observed_events[0].append(self.a_events_x[0])
    observed_events[1].append(self.a_events_y[0])

    for target_x, target_y in zip(self.a_events_x[1:], self.a_events_y[1:]):
      if target_x <= stop:
        vertical_bars.append([target_x, target_x])
        vertical_bars.append([0, target_y])
        observed_events[0].append(target_x)
        observed_events[1].append(target_y)
        observed_events[0].append(target_x)
        observed_events[1].append(1)

    data_dict = {}
    add_x_y = SpaceRepetitionDataBuilder()._create_add_x_y_fn_for(data_dict, "feedback")
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
                     x_range=stop,
                     y_domain=self.domain + 0.01,
                     epoch=self.epoch,
                     plot_pane_data=plot_pane_data
                     )
    return self.plot, data_dict

  # feedback
  def show(self):
    '''Shows the feedback plot after the ``plot_graph`` method is call.'''
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
    self.reference.ipc    = self.reference.o.ipc
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

    self.next_scheduled_day_as_offset = 0
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

    self.latest_date_offset = 0
    self.latest_result = 0

  def initialize_feedback(self, feedback, stop=None):
    '''Initializes the controller's feedback information.

    **Args**:
       | ``feedback`` (SpaceRepetitionFeedback): feedback object
       | ``stop=None`` (int, float):  days offset since epoch

    '''
    self.feedback         = ControlData(feedback)
    self.feedback.fn      = self.feedback.o.pc
    self.feedback.range   = self.feedback.o.range
    self.feedback.domain  = self.feedback.o.domain
    self.input_x          = self.feedback.o.a_events_x[:]
    self.input_y          = self.feedback.o.a_events_y[:]
    self.control_x        = self.input_x[-1]
    self.control(control_x = self.control_x)

    if stop is None:
      self.range = self.feedback.range
    else:
      self.range = stop

    self.discovered_plasticity_root = feedback.discovered_plasticity_root

    self.discovered_plasticity_denominator_offset = \
      feedback.discovered_plasticity_denominator_offset

    for day in self.updated_reference.ref_events_x:
      self._schedule.append(self.days_to_time(day))

  def error(self, x):
    '''Build an error signal event given x.

    The error signal is the result of the reference-plasticity-function(x), minus the
    feedback-plasticity-function(x).

    **Args**:
       | ``x`` (int, float): offset in days from epoch

    **Returns**:
       | (float): reference plasticity result - feedback plasticity result
       |          a positive error means we need to push harder
       |          a negative error means we are pushing too hard

    '''
    error = self.reference.fn(x) - self.feedback.fn(x)
    return error

  def plot_error(self, stop, plot_pane_data=None, panes=None):
    '''Plots the error signal.

    **Args**:
       | ``stop`` (datetime, float, int): when to stop graphing
       | ``plot_pane_data=None`` (int, float):  plot handle
       | ``panes=None`` (type1): number of sub-plot panes to make

    **Returns**:
       (tuple): (PlotPaneData, dict)

    '''
    if type(stop) is datetime:
      self.schedule(stop=stop)
      stop_day_as_offset = self.days_from_epoch(stop)
    else:
      self.schedule_as_offset(stop=stop)
      stop_day_as_offset = stop

    x1     = np.linspace(0, stop_day_as_offset, SpaceRepetition.Default_Samples)
    y1     = self.error(x1)
    bars   = self._error_vertical_bars()

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
                    x_range=stop_day_as_offset,
                    y_domain=0.4,
                    y_label="error signal",
                    x_lable="time",
                    epoch=self.epoch,
                    panes=panes,
                    plot_pane_data=plot_pane_data
                    )

    return plt.ppd, data_dict

  def _error_vertical_bars(self):
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
    '''Runs the space repetition control system.

    **Args**:
       | ``control_x=None`` (int, float): offset in days where to start control
       system.  This value defaults to the last time the student provided
       feedback.

    '''

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

  # controller
  def schedule_as_offset(self, stop):
    '''Returns the learning schedule as a list offsets measure in days since
       they training epoch.

    This list will contain the recommended training moments up to, and
    possibly including the date described by the ``stop`` input argument.

    **Args**:
       | ``stop`` (int, float, datetime): At what point to stop the schedule
       |          if stop is an int or float it represents the days since the training epoch

    **Returns**:
       (list): list of floats containing a the recommended schedule in days
               since the training epoch

    '''

    if type(stop) is datetime:
      stop_day_as_offset = self.updated_reference.days_from_epoch(stop)
    else:
      stop_day_as_offset = stop

    while stop_day_as_offset >= self.latest_date_offset:
      latest_ref_offset, self.latest_result = next(self.updated_reference.g_stickleback)
      self.latest_date_offset = latest_ref_offset + self.x_reference_shift

    if len(self.feedback.o.a_events_x) > 0:
      last_feedback_event = self.feedback.o.a_events_x[-1]
    else:
      last_feedback_event = 0

    self.dates_as_day_offsets = []

    next_found = False
    for ref_item in self.updated_reference.dates_as_day_offsets:
      schedule_item = ref_item + self.x_reference_shift

      if last_feedback_event < schedule_item:

        if not next_found:
          next_found = True
          self.next_scheduled_day_as_offset = schedule_item

        if schedule_item < stop_day_as_offset:
          self.dates_as_day_offsets.append(schedule_item)

    # return up to the last item in the list
    return list(self.dates_as_day_offsets)

  # controller
  def schedule(self, stop):
    '''Returns the learning schedule.

    This list will contain the recommended training moments up to, and possibly
    including the date described by the ``stop`` input argument.

    **Args**:
       | ``stop`` (int, float, datetime): At what point to stop the schedule
       |          if stop is an int or float it represents the days since the training epoch

    **Returns**:
       (list): list of datetime objects containing the recommended training schedule.

    '''
    stop_as_offset_in_days = self.days_from_epoch(stop)

    schedule_as_offsets_in_days = \
      self.schedule_as_offset(
        stop=stop_as_offset_in_days
      )

    schedule = [
      self.epoch + timedelta(offset) 
        for offset 
          in schedule_as_offsets_in_days
    ]

    return schedule

  def plot_control(self, stop, plot_pane_data=None, panes=None):
    '''Plots the control data.

    **Args**:
       | ``stop`` (datetime, float, int): when to stop graphing
       | ``plot_pane_data=None`` (int, float):  plot handle
       | ``panes=None`` (type1): number of sub-plot panes to make

    **Returns**:
       (tuple): (PlotPaneData, dict)

    '''
    control_x = self.control_x

    if type(stop) is datetime:
      self.updated_reference.schedule(stop=stop)
      stop_day_as_offset = self.updated_reference.days_from_epoch(stop)
    else:
      self.updated_reference.schedule_as_offset(stop)
      stop_day_as_offset = stop

    if self.reference.o.latest_date_offset < stop_day_as_offset:
      self.reference.o.schedule_as_offset(stop=stop_day_as_offset)

    if self.updated_reference.latest_date_offset < stop_day_as_offset:
      self.updated_reference.schedule_as_offset(stop=stop_day_as_offset)

    #updated_reference, x_reference_shift = self.control(control_x=control_x)
    x_ref = self.updated_reference.x_and_y[0]
    y_ref = self.updated_reference.x_and_y[1]

    control_ref_x = []
    control_ref_y = []
    for x_data, y_data in zip(x_ref, y_ref):
      x_new = self.x_reference_shift + x_data
      if(x_new <= stop_day_as_offset):
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
      if(new_x <= stop_day_as_offset):
        schedule_x.append(new_x)
        schedule_y.append(target_y)
        self.schedule_vertical_bars.append([new_x, new_x])
        self.schedule_vertical_bars.append([0, target_y])

    x = np.linspace(0, stop_day_as_offset, 500)
    rx = x[:]
    ry = self.reference.fn(rx - self.x_reference_shift)
    feedback_x = x[:]
    feedback_y = self.feedback.fn(x)
    data_args = [feedback_x, feedback_y, rx, ry, control_ref_x, control_ref_y]
    data_args += self.feedback.o.vertical_bars
    data_args += self.schedule_vertical_bars
    vertical_bars = {'vertical_bars': self.schedule_vertical_bars, 'colour': "orange"}

    data_dict = {}
    add_x_y = SpaceRepetitionDataBuilder()._create_add_x_y_fn_for(data_dict, "control")
    add_x_y("feedback_potentiation", feedback_x, feedback_y)
    add_x_y("reference_potentiation_with_offset", rx, ry)
    add_x_y("reference_forgetting_with_offset", control_ref_x, control_ref_y)
    add_x_y("schedule", schedule_x, schedule_y)
    add_x_y("moments", self.feedback.o.a_events_x, self.feedback.o.a_events_y)

    # control
    self.plot = SpaceRepetitionPlot(*data_args,
      x_range=stop_day_as_offset,
      y_domain=1 + 0.01,
      y_label="control",
      x_label="",
      scheduled=vertical_bars,
      first_graph_color=SpaceRepetitionFeedback.LongTermPotentiationColor,
      second_graph_color=SpaceRepetitionReference.LongTermPotentiationColor,
      third_graph_color=SpaceRepetitionReference.StickleBackColor,
      epoch=self.epoch,
      plot_pane_data=plot_pane_data,
      range=stop_day_as_offset,
      panes=panes)
    return self.plot, data_dict

  # control
  def plot_graphs(self, stop=None):
    '''Plots the reference, feedback, error and control data points on one
    graph.

    **Args**:
       | ``stop`` (datetime, float, int): when to stop graphing

    **Returns**:
       (tuple): (PlotPaneData, dict)

    '''
    if type(stop) is datetime:
      self.updated_reference.schedule(stop=stop)

      stop_day_as_offset = self.updated_reference.days_from_epoch(stop)
    else:
      self.updated_reference.schedule_as_offset(stop)
      stop_day_as_offset = stop

    if self.reference.o.latest_date_offset < stop:
      self.reference.o.schedule_as_offset(stop=stop)

    if self.updated_reference.latest_date_offset < stop:
      self.updated_reference.schedule_as_offset(stop=stop)

    base = {}
    db = SpaceRepetitionDataBuilder()

    hdl, data_dict = self.reference.o.plot_graph(
      panes=4,
      stop=stop_day_as_offset
    )
    db._append_to_base(base, data_dict)

    h, data_dict = self.feedback.o.plot_graph(
      plot_pane_data=hdl,
      stop=stop_day_as_offset
    )

    if stop is None:
      stop = self.feedback.o.range

    db._append_to_base(base, data_dict)

    h, data_dict = self.plot_error(
      plot_pane_data=hdl,
      stop=stop_day_as_offset
    )

    db._append_to_base(base, data_dict)

    self.updated_reference._make_data_for_plot(stop)
    h, data_dict = self.plot_control(
      plot_pane_data=hdl,
      stop=stop_day_as_offset
    )

    db._append_to_base(base, data_dict)
    graph_handle = h.ppd
    #graph_handle = hdl.ppd
    return graph_handle, base

  # control
  def show(self):
    '''Show the graph you have plotted using the ``plot_graphs`` method.'''
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

  # control
  def save_figure(self, filename="spaced.pdf"):
    '''Saves a graph made with the ``plot_graphs`` method, to file.

    **Args**:
       | ``filename="spaced.pdf"`` (string): The file name and extension (defining how it will be saved)

    **Notes**:
      This method uses the matplotlib library, which will automatically save the
      plot in the format indicated by the file name's extension.

    '''
    plt.savefig(filename, dpi=300)

  def predict_result(self, moment, curve=None):
    '''(alias for recollect_scalar) Predicts a student's recollection ability at
    a specified time on a specified forgetting curve.

    **Args**:
       | ``moment`` (float, int, datetime, timedelta):  see note
       | ``curve=None`` (int): Which forgetting curve to use in the prediction

    **Note**:

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    **Returns**:
       (float): a number between 0 and 1.  0 means the idea is completely
       forgotten and 1 means the idea is perfectly remembered.

    '''
    return self.recollect_scalar(moment, curve)

  def prediction(self, moment, curve=None):
    '''(alias for recollect_scalar) Predicts a student's recollection ability at
    a specified time on a specified forgetting curve.

    **Args**:
       | ``moment`` (float, int, datetime, timedelta):  see note
       | ``curve=None`` (int): Which forgetting curve to use in the prediction

    **Note**:

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    **Returns**:
       (float): a number between 0 and 1.  0 means the idea is completely
       forgotten and 1 means the idea is perfectly remembered.

    '''
    return self.recollect_scalar(moment, curve)

  # controller
  def recollect_scalar(self, moment, curve=None):
    '''Predicts a student's recollection ability at a specified time on a
    specified forgetting curve.

    **Args**:
       | ``moment`` (float, int, datetime, timedelta):  see note
       | ``curve=None`` (int): Which forgetting curve to use in the prediction

    **Note**:

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    **Returns**:
       (float): a number between 0 and 1.  0 means the idea is completely
       forgotten and 1 means the idea is perfectly remembered.

    '''
    if(isinstance(moment, datetime)):
      assert moment + timedelta(days=0.01) > self.epoch
      moment_as_offset_in_days = self.datetime_to_days_offset_from_epoch(moment)
    else:
      assert moment >= 0
      moment_as_offset_in_days = moment

    self.schedule_as_offset(stop=moment_as_offset_in_days)

    if curve is None:
      curve = 1

    forgetting_function = self.updated_reference.forgetting_enclosures[curve-1]
    # shift the query into the updated_reference time reference
    query_time_in_days = moment_as_offset_in_days - self.x_reference_shift
    y = forgetting_function(query_time_in_days)
    return y

  # controller
  def datetime_for(self, curve=None):
    '''Returns the datetime of a specified training suggestion.
    
    The moment of this suggestion is represented by the ``curve`` input
    argument.  The curve numbers are indexed from 1, not 0.  The curve is the
    forgetting curve for which you want the start-date of.

    **Args**:
       | ``curve=None`` (int): The number of the forgetting curve.  The curves
       are indexed from 1, not 0.  If no argument is provided, the curve number
       will be set to 1.

    **Returns**:
       (datetime): The suggested training moment.

    '''
    if curve is None or curve <= 0:
      index = 0
    else:
      index = curve - 1
    return self.days_to_time(self.dates_as_day_offsets[index])

  # control
  def range_for(self, stop, curve=None, day_step_size=1):
    '''Returns a list of useful datetime values for a given forgetting-curve, up
    to, but not including the stop value, in increments of day_step_size.

    **Args**:
       | ``stop`` (datetime, float, int):  datetime or offset from epoch in days
       | ``curve=None`` (type1): default: which forgetting curve, starting at 1
       | ``day_step_size=1`` (type1): step size of the result (unit of days)

    **Returns**:
       (list): list of datetimes that can be used in another query

    '''
    curve = curve if curve or curve >= 1 else 1

    if curve >= self.updated_reference.maxlen:
      raise Exception('You can not query beyond the bounds of the current ring buffer')

    if type(stop) is datetime:
      stop_day_as_offset = self.days_from_epoch(stop)
    else:
      stop_day_as_offset = stop

    if stop_day_as_offset > self.latest_date_offset:
      self.schedule_as_offset(stop=stop_day_as_offset)

    days_offset_of_curve = self.dates_as_day_offsets[curve-1]

    start_date = \
      self.days_offset_from_epoch_to_datetime(days_offset_of_curve)

    end_date = \
      self.days_offset_from_epoch_to_datetime(stop_day_as_offset)

    result = list(
      np.arange(
        start_date,
        end_date, 
        timedelta(days=day_step_size)
      ).astype(datetime)
    )

    return result

  # control
  def next_offset(self):
    '''Returns the next suggested time to study as a float, representing the
       number of days since the training epoch.

    **Returns**:
       (float): next suggested lesson [days since training epoch]

    '''
    if len(self.feedback.o.a_events_x) > 0:
      last_feedback_event = self.feedback.o.a_events_x[-1]
    else:
      last_feedback_event = 0

    if not hasattr(self, 'next_scheduled_day_as_offset') or \
      self.next_scheduled_day_as_offset < last_feedback_event:

      self.schedule_as_offset(stop=last_feedback_event)

    return self.next_scheduled_day_as_offset

  # control
  def next(self):
    '''Returns the next suggested time to study as a datetime object

    **Returns**:
       (datetime): next suggested lesson

    '''
    return self.days_to_time(self.next_offset())

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
      epoch=self.start_time,
      plasticity_denominator_offset=self.plasticity_denominator_offset,
      plasticity_root=self.plasticity_root,
      long_term_clamp=self.long_term_clamp,
    )

    self.control = SpaceRepetitionController(
      reference=self.reference,
      feedback=self._feedback,
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

  # learning tracker
  def add_event(self, feedback_moment, result):
    '''Adds a training event to the learning tracker.

    **Args**:
       | ``feedback_moment`` (int, float): moment as time offset in days from epoch
       | ``result`` (float): how well the student performed (0.0-1.0)

    '''
    self.feedback_x.append(feedback_moment)
    self.feedback_y.append(result)

    # sort the times and results of a_events_x and a_events_y together
    s = sorted(zip(self.feedback_x, self.feedback_y))
    self.feedback_x, self.feedback_y, map(list, zip(*s))

    self.frame = np.arange(1, len(self.feedback_x))
    hf = SpaceRepetitionFeedback(self.feedback_x,
        self.feedback_y,
        epoch=self.epoch)
    self.control.initialize_feedback(feedback=hf)

  # learning tracker
  def plot_graphs(self, stop):
    '''Plots the reference, feedback, error and control data points on one graph.

    **Args**:
       | ``stop`` (datetime, float, int): when to stop graphing

    **Returns**:
       (tuple): (PlotPaneData, dict)

    '''
    return self.control.plot_graphs(stop)

  @contextmanager
  def graphs(self, stop,
      filename=None,
      reference_handle=False,
      feedback_handle=False,
      error_handle=False,
      control_handle=False,
      show=False):
    '''Writes graphs, yields the requested graph handles, writes to file
       and automatically closes the graph handlers all within a context manager.

    **Args**:
       | ``stop`` (datetime, float, int): when to stop graphing
       | ``filename="None"`` (string): The file name and extension (defining how it will be saved)
       | ``reference_handle="False"`` (bool): Reference graph handle needed?
       | ``feedback_handle="False"`` (bool): Feedback graph handle needed?
       | ``error_handle="False"`` (bool): Error graph handle needed?
       | ``control_handle="False"`` (bool): Control graph handle needed?

    **Yields**:
       (tuple): graphs handles as described by input args

    **Example(s)**:
      
    .. code-block:: python
        
        from datetime import datetime
        from spaced import LearningTracker
 
        lt = LearningTracker(epoch=datetime.now())
 
        moments = lt.range_for(curve=1, stop=43, day_step_size=0.5)
        predictions = [
          lt.recollect_scalar(moment, curve=1) for 
          moment in momemts]

        with lt.graphs(stop=43, filename='contenxt_demo.pdf') as c_hdl:
          c_hdl.plot(moments, predictions, color='xkcd:azure')

        # the ordering of the handles provided by the context manager
        # matches the graph ordering.
        # 1 - reference
        # 2 - feedback
        # 3 - error
        # 4 - control
        with lt.graphs(stop=43, filename='contenxt_demo.pdf',
          control_handle=True, reference_handle=True) as (r_dl, c_hdl):

          c_hdl.plot(moments, predictions, color='xkcd:azure')

    '''
    try:
      gh, _ = self.plot_graphs(stop) 

      # 0000
      if reference_handle is False and \
         feedback_handle is False and \
         error_handle is False and \
         control_handle is False:
        yield gh

      # 0001
      elif reference_handle is False and \
         feedback_handle is False and \
         error_handle is False and \
         control_handle is True:
        # default behavior
        yield gh.axarr[3]

      # 0010
      elif reference_handle is False and \
         feedback_handle is False and \
         error_handle is True and \
         control_handle is False:
        yield gh.axarr[2]

      # 0011
      elif reference_handle is False and \
         feedback_handle is False and \
         error_handle is True and \
         control_handle is True:
        yield gh.axarr[2], gh.ararr[3]

      # 0100
      elif reference_handle is False and \
         feedback_handle is True and \
         error_handle is False and \
         control_handle is False:
        yield gh.axarr[1]

      # 0101
      elif reference_handle is False and \
         feedback_handle is True and \
         error_handle is False and \
         control_handle is True:
        yield gh.axarr[1], gh.axarr[3]

      # 0110
      elif reference_handle is False and \
         feedback_handle is True and \
         error_handle is True and \
         control_handle is False:
        yield gh.axarr[1], gh.axarr[2]

      # 0111
      elif reference_handle is False and \
         feedback_handle is True and \
         error_handle is True and \
         control_handle is True:
        yield gh.axarr[1], gh.axarr[2], gh.axarr[3]

      # 1000
      elif reference_handle is True and \
         feedback_handle is False and \
         error_handle is False and \
         control_handle is False:
        yield gh.axarr[0]

      # 1001
      elif reference_handle is True and \
         feedback_handle is False and \
         error_handle is False and \
         control_handle is True:
        yield gh.axarr[0], gh.axarr[3]

      # 1010
      elif reference_handle is True and \
         feedback_handle is False and \
         error_handle is True and \
         control_handle is False:
        yield gh.axarr[0], gh.axarr[2]

      # 1011
      elif reference_handle is True and \
         feedback_handle is False and \
         error_handle is True and \
         control_handle is True:
        yield gh.axarr[0], gh.axarr[2], gh.axarr[3]

      # 1100
      elif reference_handle is True and \
         feedback_handle is True and \
         error_handle is False and \
         control_handle is False:
        yield gh.axarr[0], gh.axarr[1]

      # 1101
      elif reference_handle is True and \
         feedback_handle is True and \
         error_handle is False and \
         control_handle is True:
        yield gh.axarr[0], gh.axarr[1], gh.axarr[3]

      # 1110
      elif reference_handle is True and \
         feedback_handle is True and \
         error_handle is True and \
         control_handle is False:
        yield gh.axarr[0], gh.axarr[1], gh.axarr[2]

      # 1111
      else:
        yield gh.axarr[0], gh.axarr[1], gh.axarr[2], gh.axarr[3]

    finally:
      if filename is not None:
        self.save_figure(filename)

      # show has to happen after a save or jupyter will save a blank file
      if show:
        plt.show()
      gh.close() 

  # learning tracker
  def save_figure(self, filename=None):
    '''Save the graph plotted by the ``plot_graphs`` method to file.

    **Args**:
       | ``filename="spaced.pdf"`` (string): The file name and extension (defining how it will be saved)

    **Notes**:
      This method uses the matplotlib library, which will automatically save the
      plot in the format indicated by the file name's extension.

    '''
    self.control.save_figure(filename)

  # learning tracker
  def schedule(self, stop):
    '''Returns a schedule as of datetimes, up-until and possibily-including the
    datetime or offset represented by the ``stop`` argument.

    **Args**:
      | ``stop`` (float, int, datetime): at what point do you want the schedule
      information to stop.  If ``stop`` is an int or a float it will be
      interpreted as the number of days-from-epoch at which you would like this
      function to stop providing information. If ``stop`` is a datetime, the
      output schedule will include all of the schedule suggestions up until
      that time.

    **Returns**:
       (list of datetime objects): The schedule

    '''
    return self.control.schedule(stop)

  # learning tracker
  def schedule_as_offset(self, stop):
    '''Return a schedule as a list of offsets measured in days from the training
    epoch, up-until and possibily-including the datetime or offset reprented by
    the ``stop`` argument.

    **Args**:
      | ``stop`` (float, int, datetime): at what point do you want the schedule
      information to stop.  If ``stop`` is an int or a float it will be
      interpreted as the number of days-from-epoch at which you would like this
      function to stop providing information. If ``stop`` is a datetime, the
      output schedule will include all of the schedule suggestions up until
      that time.

    **Returns**:
       (list of floats): The schedule as offset from the training epoch

    '''
    return self.control.schedule_as_offset(stop)

  def learned(self, when, result):
    '''Tells the learning tracker about something the student has learned.

    **Args**:
       | ``when`` (int, float, datetime): moment as either time offset in days from epoch, or the moment of training represented by a datetime object
       | ``result`` (float): how well the student performed (0.0-1.0)
    '''
    moment = when
    if(isinstance(moment, datetime)):
      assert(moment + timedelta(days=0.01) > self.epoch)
      moment_as_datetime = moment
    elif(isinstance(moment, float) or isinstance(moment, int)):
      moment_as_datetime = self.epoch + timedelta(days=float(moment))
    else:
      raise("object not supported")

    moment_as_offset_in_days = self.control.days_from_epoch(moment_as_datetime)

    assert(0.0 <= result <= 1.0)
    self.add_event(
      moment_as_offset_in_days,
      result)

  # learning tracker
  def animate(self, name_of_mp4=None,
      student=None,
      time_per_event_in_seconds=None,
      stop=None):
    '''Creates an mp4 video of a student's training up to the moment indicated by the
    ``stop`` input argument.

    **Note**:
       This function uses the animation feature of the matplotlib library.

    **Args**:
       | ``name_of_mp4=None`` (string): name and path of the resulting output video
       | ``student=None`` (string): name of the student to be placed on the video
       | ``time_per_event_in_seconds=None`` (int): time per frame event
       | ``stop=None`` (int, float, datetime): when to stop the video.  If an
       int or float provided, this indicates the number of days since the
       training epoch.
    '''
    self.base          = {}
    self.base["frame"] = {}

    if stop is None:
      stop_day_as_offset = self.feedback_x[-1] + self.feedback_x[-1] * 0.5
    elif type(stop) is datetime:
      self.control.schedule(stop=stop)
      stop_day_as_offset = self.control.days_from_epoch(stop)
    else:
      stop_day_as_offset = stop
    self._feedback.range = stop_day_as_offset

    self.control.schedule_as_offset(stop_day_as_offset)

    self.base["range"] = stop_day_as_offset

    if name_of_mp4 is None:
      self.name_of_mp4 = "animate.mp4"
    else:
      self.name_of_mp4 = name_of_mp4

    if student is None:
      self.student = "example"
    else:
      self.student = "student: {}".format(student)

    if time_per_event_in_seconds is not None:
      if time_per_event_in_seconds == 0:
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
      )
      hf = SpaceRepetitionFeedback(
          self.feedback_x[0:item + 1],
          self.feedback_y[0:item + 1],
          epoch=self.start_time,
          range=stop_day_as_offset,
      )
      if item == 0:
        hctrl = SpaceRepetitionController(
            reference=hr,
            feedback=hf,
            plasticity_root=self.plasticity_root,
            plasticity_denominator_offset=self.plasticity_denominator_offset,
            epoch=self.start_time
      )
      else:
        hctrl.initialize_feedback(feedback=hf, stop=stop_day_as_offset)

      graph_handle, data_dict = hctrl.plot_graphs(stop=stop_day_as_offset)
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

  # learning tracker
  def range_for(self, stop, curve=None, day_step_size=1):
    '''Returns a list of useful datetime values for a given curve, up to, but not
    including the stop value, in increments of day_step_size.

    **Args**:
       | ``stop`` (datetime, float, int):  datetime or offset from epoch in days
       | ``curve=None`` (type1): default: which forgetting curve, starting at 1
       | ``day_step_size=1`` (type1): step size of the result (unit of days)

    **Returns**:
       (list): list of datetimes that can be used in another query

    '''
    return self.control.range_for(stop, curve, day_step_size)

  # learning tracker
  def predict_result(self, moment, curve=None):
    '''(alias for recollect_scalar) Predicts a student's recollection ability, at
    a specified time, on a specifiy forgetting curve.

    **Args**:
       | ``moment`` (float, int, datetime, timedelta):  see note
       | ``curve=None`` (int): Which forgetting curve to use in the prediction

    **Note**:

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    **Returns**:
       (float): a number between 0 and 1.  0 means the idea is completely
       forgotten and 1 means the idea is perfectly remembered.

    '''
    return self.recollect_scalar(moment, curve)

  # learning tracker
  def recollect_scalar(self, moment, curve=None):
    '''Predicts a student's recollection ability, at a specified time, on a
    specifiy forgetting curve.

    **Args**:
       | ``moment`` (float, int, datetime, timedelta):  see note
       | ``curve=None`` (int): Which forgetting curve to use in the prediction

    **Note**:

    The ``moment`` input can be one of these types:

      * float     - representing the number of days since epoch (when the
                    spaced_repetition algorithm started to track)
      * datetime  - a datetime object, which must be after epoch
      * timedelta - the difference in time from when the client wants to know
                    something and epoch

    **Returns**:
       (float): a number between 0 and 1.  0 means the idea is completely
       forgotten and 1 means the idea is perfectly remembered.

    '''
    return self.control.recollect_scalar(moment, curve)

  # learning tracker
  def epoch_fdecaytau(self):
    '''Returns the reference's fdcecaytau parameter.'''
    return self.reference.fdecaytau

  # learning tracker
  def epoch_fdecay0(self):
    '''Returns the reference's fdecay0 parameter.'''
    return self.reference.fdecaytau

  # learning tracker
  def epoch_plasticity_root(self):
    '''Returns the reference's plasticity_root parameter.'''
    return self.reference.plasticity_root

  # learning tracker
  def epoch_plasticity_denominator_offset(self):
    '''Returns the reference's plasticity_denominator_offset parameter.'''
    return self.reference.plasticity_denominator_offset

  # learning tracker
  def discovered_fdecaytau(self):
    '''Returns the control's discovered fdecaytau parameter.'''
    return self.control.fdecaytau

  # learning tracker
  def discovered_fdecay0(self):
    '''Returns the control's discovered fdecay0 parameter.'''
    return self.control.fdecay0

  # learning tracker
  def days_from_epoch(self, time):
    '''Converts a datetime object into a number representing the days since the
    training epoch.

    **Args**:
      | ``time`` (int, float): days since the training epoch

    **Returns**:
       (float): days since training epoch

    '''
    return self.control.days_from_epoch(time)

  # learning tracker
  def discovered_plasticity_root(self):
    '''Returns the control's discovered plasticity_root parameter.'''
    return self.control.discovered_plasticity_root

  # learning tracker
  def discovered_plasticity_denominator_offset(self):
    '''Returns the control's discovered denominator_offset parameter.'''
    return self.control.discovered_plasticity_denominator_offset

  # learning tracker
  def feedback(self, time_format=None):
    '''Returns a list of the feedback events seen by this learning tracker.

    **Args**:
       | ``time_format(None)`` (TimeFormat): OFFSET or DATE_TIME

    **Returns**:
       (list): moment of feedback in the specified format.

    '''
    results = None

    if time_format is None:
      time_format = TimeFormat.OFFSET

    if time_format is TimeFormat.OFFSET:
      results = self.feedback_x, self.feedback_y
    elif time_format is TimeFormat.DATE_TIME:
      results = [self.epoch + timedelta(days=f) for f in self.feedback_x], self.feedback_y

    return results

  # learning tracker
  def next_offset(self):
    '''Returns the next suggested time to study as a float, representing the
       number of days since the training epoch.

    **Returns**:
       (float): next suggested lesson [days since training epoch]

    '''
    return self.control.next_offset()

  # learning tracker
  def next(self):
    '''Returns the next suggested time to study as a datetime object

    **Returns**:
       (datetime): next suggested lesson

    '''
    return self.control.next()

  # learning tracker
  def show(self):
    self.control.show()

  def __getstate__(self):
    '''Over-writes the load, or pickling process, by returning
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
    
if __name__ == "__main__":
  import doctest
  doctest.testmod()
