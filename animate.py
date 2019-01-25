"""animate

The animate module provides the animation features needed by the ``spaced``
package.  It mainly acts as a wrapper for `matplotlib <https://matplotlib.org/>`_.

The ``graph`` module provides a lot of features which can be read about in `the
full spaced-package documentation <https://aleph2c.github.io/spaced/>`_.

"""
import matplotlib.pyplot as plt
import pprint
import json

ppp = pprint.PrettyPrinter(indent=2)
def pp(thing):
  ppp.pprint(thing)

class LearningTrackerAnimation():

  def __init__(self, json_file, *args, **kwargs):
    json_data = open(json_file, 'r').read()
    self.data = json.loads(json_data)
    self.fig, self.axarr = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, figsize=(11, 8.5))
    self.fig.subplots_adjust(hspace=0)
    self.axarr[0].axes.set_ylabel("ref")
    self.axarr[1].axes.set_ylabel("feedback")
    self.axarr[2].axes.set_ylabel("error")
    self.axarr[3].axes.set_ylabel("control")
    self.axarr[3].set_xlim(0, self.data['range'])
    self.axarr[3].set_ylim(0, 1.1)
    plt.setp([a.get_xticklabels() for a in self.fig.axes[:-1]], visible=False)
    self.json_file = json_file

    self.frames = self.data['frames']

  def animate(self, i):
    fig   = self.fig
    axarr = self.axarr
    data  = self.data

    json_data = open(self.json_file, 'r').read()
    data = json.loads(json_data)
    for pane in range(0, len(axarr)):
      axarr[pane].clear()
    axarr[3].xaxis.label.set_color('white')
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]],  visible=False)
    axarr[3].set_xlim(0, data['range'])
    axarr[3].set_ylim(0, 1.1)

    axarr[0].axes.set_ylabel("ref")
    axarr[1].axes.set_ylabel("feedback")
    axarr[2].axes.set_ylabel("error")
    axarr[3].axes.set_ylabel("control")

    frame_number = str(i)

    def graph(data, pane, graph_name, item, color):
      rp = data['frame'][frame_number][graph_name][item]
      x = []
      y = []
      for item in rp:
        x.append(item['x'])
        y.append(item['y'])
      axarr[pane].plot(x, y, color=color)

    def vertical_bars(data, pane, graph_name, item, color):
      rp = data['frame'][frame_number][graph_name][item]
      for item in rp:
        x = item['x']
        y = item['y']
        bars_x = [x, x]
        bars_y = [0, y]
        axarr[pane].plot(bars_x, bars_y, color=color, linestyle="-.")

    if frame_number in data['frame']:

      graph(data, 0, 'recommendation', 'forgetting', 'xkcd:orangered')
      graph(data, 0, 'recommendation', 'long_term_potentiation', 'xkcd:blue')
      vertical_bars(data, 0, 'recommendation', 'moments', 'orange')

      graph(data, 1, 'feedback', 'long_term_potentiation', 'xkcd:teal')
      graph(data, 1, 'feedback', 'forgetting', 'xkcd:red')
      vertical_bars(data, 1, 'feedback', 'moments', 'xkcd:cement')

      rp = data['frame'][frame_number]['error']
      x = []
      y = []
      for item in rp:
        x.append(item['x'])
        y.append(item['y'])
      axarr[2].plot(x, y)
      vertical_bars(data, 2, 'feedback', 'moments', 'xkcd:cement')

      graph(data, 3, 'control', 'reference_potentiation_with_offset', 'xkcd:blue')
      graph(data, 3, 'control', 'feedback_potentiation', 'xkcd:teal')
      graph(data, 3, 'control', 'reference_forgetting_with_offset',  'xkcd:orangered')
      vertical_bars(data, 3, 'control', 'schedule', 'orange')
      vertical_bars(data, 3, 'control', 'moments', 'xkcd:cement')


