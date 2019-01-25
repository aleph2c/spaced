"""graph

The graph module provides the graphing features needed by the ``spaced``
package.  It mainly acts as a wrapper for `matplotlib <https://matplotlib.org/>`_.

The ``graph`` module provides a lot of features which can be read about in `the
full spaced-package documentation <https://aleph2c.github.io/spaced/>`_.

"""
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class PlotPaneData(object):

  def __init__(self, *args, **kwargs):
    self.graph_location = -1
    self.figure         = None
    self.axarr          = None
    self.epoch          = None

    if("graph_location" in kwargs):
      self.graph_location = kwargs["graph_location"]

    if("figure" in kwargs):
      self.figure = kwargs["figure"]

    if("axarr" in kwargs):
      self.axarr = kwargs["axarr"]

    if("epoch" in kwargs):
      self.epoch = kwargs["epoch"]

    if("plot" in kwargs):
      self.plot = kwargs["plot"]

  def close(self):
    self.plot.close()

class SpaceRepetitionBasePlotClass:
  def __init__(self, *data, **kwargs):
    self.figure         = None
    self.data_args      = 0

    if("epoch" in kwargs):
      epoch = kwargs["epoch"]
    else:
      epoch = None

    if 'plot_pane_data' in kwargs and kwargs['plot_pane_data'] is not None:
      self.ppd = kwargs['plot_pane_data']
    else:
      if not 'panes' in kwargs:
        kwargs['panes'] = 1
      if kwargs['panes'] is None:
        kwargs['panes'] = 1

      #figure, axarr = plt.subplots(kwargs['panes'], 1, sharex=True, figsize=(11, 8.5), facecolor='#07000d')
      figure, axarr = plt.subplots(kwargs['panes'], 1, sharex=True, figsize=(11, 8.5), facecolor='white')
      self.figure = figure
      self.axarr = axarr
      self.ppd = PlotPaneData(graph_location=-1,
        figure=figure,
        axarr=axarr,
        epoch=epoch,
        plot=plt)

    self.ppd.graph_location += 1

    # alias ppd to make it easier to understand from the outside
    self.plot_pane_data = self.ppd

    if 'x_label' in kwargs:
      self.x_label = kwargs['x_label']
    else:
      self.x_label = ''

    if 'scheduled' in kwargs:
      self.scheduled = kwargs['scheduled']
    else:
      self.scheduled = None

    if 'y_label' in kwargs:
      self.y_label = kwargs['y_label']
    else:
      self.y_label = ''

    if 'first_graph_color' in kwargs:
      self.first_graph_color = kwargs['first_graph_color']
      self.data_args = 4
    else:
      self.first_graph_color = None

    if 'second_graph_color' in kwargs:
      self.second_graph_color = kwargs['second_graph_color']
      self.data_args = 4
    else:
      self.second_graph_color = None

    if 'third_graph_color' in kwargs:
      self.third_graph_color = kwargs['third_graph_color']
      self.data_args = 6
    else:
      self.third_graph_color = None

    if 'forth_graph_color' in kwargs:
      self.forth_graph_color = kwargs['forth_graph_color']
      self.data_args = 6
    else:
      self.forth_graph_color = None

  def get_time_series_function(self):
    def ftime_series(x, y):
      new_data    = [[], []]
      time_series = []
      time_series = np.asarray([self.days_to_time(self.ppd.epoch, days) for days in x])
      new_data[0] = time_series[:]
      new_data[1] = y[:]
      return [new_data[0][:], new_data[1][:]]
    return ftime_series

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

  def days_to_time(self, epoch, days):
    if isinstance(days, datetime):
      raise TypeError("datetime objects not supported")
    else:
      time              = epoch
      time             += timedelta(seconds=(days * 86400))
    return time

  def show(self):
    plt.show(block=True)

  def close(self):
    plt.close()

class SpaceRepetitionPlot(SpaceRepetitionBasePlotClass):
  def __init__(self, *data, **kwargs):
    if 'epoch' in kwargs:
      from_epoch     = SpaceRepetitionPlotDaysFromEpoch(*data, **kwargs)
      self.ppd       = from_epoch.ppd
      self.ppd.epoch = kwargs['epoch']
    else:
      from_zero      = SpaceRepetitionPlotDaysFromZero(*data, **kwargs)
      self.ppd       = from_zero.ppd
      self.ppd.epoch = None

class SpaceRepetitionPlotDaysFromEpoch(SpaceRepetitionBasePlotClass):
  def __init__(self, *data, **kwargs):
    SpaceRepetitionBasePlotClass.__init__(self, *data, **kwargs)

    i              = self.ppd.graph_location
    f              = self.ppd.figure
    axarr          = self.ppd.axarr
    x_range        = kwargs['x_range']
    y_domain       = kwargs['y_domain']

    f.fmt_xdata = mdates.DateFormatter('%y-%m-%d')
    try:
      plot = axarr[i]
    except:
      plot = axarr
    axes = f.gca()

    date_min = self.ppd.epoch
    date_max = self.days_to_time(self.ppd.epoch, x_range)
    axes.set_xlim(date_min, date_max)
    axes.set_ylim(0, y_domain)

    ftime_series = self.get_time_series_function()
    if len(data) > self.data_args:
      new_data = ftime_series(data[0], data[1])
      if self.first_graph_color is None:
        plot.plot(*new_data[0:2])
      else:
        plot.plot(*new_data[0:2], color=self.first_graph_color)


      new_data = ftime_series(data[2], data[3])
      if self.second_graph_color is None:
        if len(data) > 2:
          plot.plot(*new_data[0:2])
      else:
        plot.plot(*new_data[0:2], color=self.second_graph_color)

      new_data = ftime_series(data[4], data[5])
      if self.third_graph_color is None:
        if len(data) > 4:
          plot.plot(*new_data[0:2])
      else:
        plot.plot(*new_data[0:2], color=self.third_graph_color)

      for x in range(self.data_args, len(data)):
        if x % 2 == 0:
          new_data = ftime_series(data[x], data[x + 1])
          plot.plot(*new_data, color='xkcd:cement', linestyle="-.")

      if self.scheduled is not None:
        vb = self.scheduled['vertical_bars']
        colour = self.scheduled['colour']
        for x in range(0, len(vb)):
          if x % 2 == 0:
            new_data = ftime_series(vb[x], vb[x + 1])
            plot.plot(*new_data, color=colour, linestyle="-.")
            #bars = vb[x:x+2]
            #plot.plot(*bars, color=colour, linestyle="-.")

      plot.axes.set_ylabel(self.y_label)
      plot.axes.set_xlabel(self.x_label)
      #plot.xaxis.set_major_locator(ticker.FixedLocator(([date_min, date_max])))
      for tick in plot.xaxis.get_ticklabels():
        tick.set_rotation(25)
        tick.set_rotation_mode('anchor')
        tick.set_va('top')
        tick.set_ha('right')

class SpaceRepetitionPlotDaysFromZero(SpaceRepetitionBasePlotClass):
  def __init__(self, *data, **kwargs):
    SpaceRepetitionBasePlotClass.__init__(self, *data, **kwargs)

    i              = self.ppd.graph_location
    f              = self.ppd.figure
    axarr          = self.ppd.axarr
    x_range        = kwargs['x_range']
    y_domain       = kwargs['y_domain']

    f.fmt_xdata = mdates.DateFormatter('%y-%m-%d')
    plot        = axarr[i]

    if len(data) > self.data_args:
      if self.first_graph_color is None:
        plot.plot(*data[0:2])
      else:
        plot.plot(*data[0:2], color=self.first_graph_color)

      if self.second_graph_color is None:
        if len(data) > 2:
          plot.plot(*data[2:4])
      else:
        plot.plot(*data[2:4], color=self.second_graph_color)

      if self.third_graph_color is None:
        if len(data) > 4:
          plot.plot(*data[4:6])
      else:
        plot.plot(*data[4:6], color=self.third_graph_color)

      #plot.plot(data[0],data[1],data[2],data[3])
      #plot.plot(*data[2:4])
      axes = f.gca()
      axes.set_xlim(0, x_range)
      axes.set_ylim(0, y_domain)
      for x in range(self.data_args, len(data)):
        if x % 2 == 0:
          bars = data[x:x + 2]
          plot.plot(*bars, color='xkcd:cement', linestyle="-.")
      if self.scheduled is not None:
        vb = self.scheduled['vertical_bars']
        colour = self.scheduled['colour']
        for x in range(0, len(vb)):
          if x % 2 == 0:
            bars = vb[x:x + 2]
            plot.plot(*bars, color=colour, linestyle="-.")

      plot.axes.set_ylabel(self.y_label)
      plot.axes.set_xlabel(self.x_label)

class ErrorPlot(SpaceRepetitionBasePlotClass):

  def __init__(self, *data, **kwargs):
    if 'epoch' in kwargs:
      from_epoch     = ErrorPlotFromEpoch(*data, **kwargs)
      self.ppd       = from_epoch.ppd
      self.ppd.epoch = kwargs['epoch']
    else:
      from_zero      = ErrorPlotFromZero(*data, **kwargs)
      self.ppd       = from_zero.ppd
      self.ppd.epoch = None

class ErrorPlotFromZero(SpaceRepetitionBasePlotClass):

  def __init__(self, *data, **kwargs):
    SpaceRepetitionBasePlotClass.__init__(self, *data, **kwargs)
    axarr          = self.ppd.axarr
    i              = self.ppd.graph_location
    f              = self.ppd.figure

    f.fmt_xdata = mdates.DateFormatter('%y-%m-%d')
    plot        = axarr[i]

    x_range  = kwargs['x_range']

    if 'title' in kwargs:
      plot.title(kwargs['title'])

    if 'x_label' in kwargs:
      x_label = kwargs['x_label']
    else:
      x_label = ''

    if 'y_label' in kwargs:
      y_label = kwargs['y_label']
    else:
      y_label = ''

    if len(data) >= 3:
      plot.plot(*data[0:2])
      axes = f.gca()
      axes.set_xlim(0, x_range)
      axes.set_ylim(-0.4, 0.4)
      plot.plot([0, x_range], [0, 0], color='xkcd:cement')
      for x in range(2, len(data)):
        bars = data[x]
        plot.plot(*bars, color='xkcd:cement', linestyle="-.")

      plot.axes.set_ylabel(y_label)
      plot.axes.set_xlabel(x_label)

class ErrorPlotFromEpoch(SpaceRepetitionBasePlotClass):

  def __init__(self, *data, **kwargs):
    SpaceRepetitionBasePlotClass.__init__(self, *data, **kwargs)
    axarr = self.ppd.axarr
    i     = self.ppd.graph_location
    f     = self.ppd.figure

    f.fmt_xdata = mdates.DateFormatter('%y-%m-%d')
    plot        = axarr[i]
    axes        = f.gca()

    x_range  = kwargs['x_range']

    date_min = self.ppd.epoch
    date_max = self.days_to_time(self.ppd.epoch, x_range)
    axes.set_xlim(date_min, date_max)
    axes.set_ylim(-0.4, 0.4)

    ftime_series = self.get_time_series_function()

    if 'x_label' in kwargs:
      x_label = kwargs['x_label']
    else:
      x_label = ''

    if 'y_label' in kwargs:
      y_label = kwargs['y_label']

    else:
      y_label = ''
    if len(data) >= 3:
      new_data = ftime_series(data[0], data[1])
      plot.plot(*new_data[0:2])

      new_data = ftime_series([0, x_range], [0, 0])
      plot.plot(*new_data, color='xkcd:cement')
      for x in range(2, len(data)):
        new_data = ftime_series(data[x][0], data[x][1])
        plot.plot(*new_data, color='xkcd:cement', linestyle="-.")

    plot.axes.set_ylabel(y_label)
    plot.axes.set_xlabel(x_label)


