from spaced_repetition import *
from graph import *
import numpy as np
from datetime import datetime
from datetime import timedelta

def get_feedback1():
  '''
  This is the results coming back from the student all at once.  x_v is the day
  from the start that they actually did the work.  y_v is a number between
  0-1.0, where 0 means they got everything wrong and 1.0 means they remembered
  everything perfectly.

  There are 10 days matched to 10 results.
  '''
  x_v = [
          0,
          0.80640320160080037,
          1.7523761880940469,
          3.0240120060030016,
          4.8074037018509257,
          8.3351675837918959,
          10.932966483241621,
          16.004002001000501,
          23.029014507253628,
          29.029014507253628
        ]

  y_v = [
          0.40,
          0.44646101201172317,
          0.64510969552772512-0.1,
          0.76106659335521121-0.2,
          0.83741905134093808-0.3,
          0.89000147547559727,
          1.00000000000000000,
          0.99994393927347419,
          0.99929261332375605,
          1.00
          ]
  return [x_v,y_v]

def get_feedback2():
  x_v = [
          0,
          0.80640320160080037,
          1.7523761880940469,
          3.0240120060030016,
          4.8074037018509257,
          7.3351675837918959,
          10.932966483241621,
          16.004002001000501,
          23.029014507253628
        ]
  y_v = [1,
          0.44646101201172317,
          0.64510969552772512,
          0.76106659335521121,
          0.93741905134093808,
          0.99000147547559727,
          0.99902389355455017,
          1.00000000000000000,
          0.90000000000000000 ]
  return [x_v,y_v]

def time_round_trip(start_time, time_from_start):
  time =  start_time
  time += timedelta(seconds=(time_from_start*86400.0))
  round_trip = 0
  if isinstance(time, datetime):
    round_trip =  time
    round_trip -= start_time
  result_in_seconds = round_trip.total_seconds()
  result = float(result_in_seconds/(86400.0))
  return result

def test_reference():
  start_time = datetime.now()
  x, y       = get_feedback1()
  range_     = x[-1] * 1.5
  hr = SpaceRepetitionReference(
         plot=True,
         range=range_,
         epoch=start_time,
         )
  hr.plot_graph()
  result = hr.datetime_for(curve=1)
  print(result)
  print(hr.range)
  print(len(hr.range_for(curve=1)))
  #plt.close('all')
#
#def test_feedback():
#  x_v,y_v = get_feedback1()
#  hdl = SpaceRepetitionFeedback(x_v,y_v)
#  hdl.plot_graph()
#  hdl.show()
#  plt.close('all')
def build_time_and_results(start_time, test_set_number=1):
  '''
  This function uses the set of ten days given as the first returned argument
  from the get_feedback1 call, and turns it into an actual datetime number
  offset from the start_time.

  These new datetimes numbers placed into a list and returned along with the y
  values gotten from the get_feedback1 call.
  '''
  if test_set_number is 1:
    x, y = get_feedback1()
  elif test_set_number is 2:
    x, y = get_feedback1()
  else:
    raise ValuesError()

  dtevents = []
  for day in x:
    new_time = start_time
    # 86400 seconds in a day
    new_time += timedelta(seconds=(day * 86400.0))
    dtevents.append(new_time)
  return [dtevents, y]

def test_controller():
  start_time = datetime.now()
  t, y       = build_time_and_results(start_time, 1)
  x, y       = get_feedback1()
  # he uses the last number from his test set, plus 50 percent of this number as
  # his range ... his last testing day is about 29, so his range is about 43
  # days.
  range_     = x[-1] + x[-1] * 0.5
  # here is appears that he was testing to see if his timedelta code was working
  # as expected.
  assert abs(time_round_trip(start_time, 0) - 0) < 0.001
  result = time_round_trip(start_time, 1.1)
  assert abs(result - 1.1) < 0.001
  hr = SpaceRepetitionReference(
         plot=False,
         range=range_,
         epoch=start_time)
  hf  = SpaceRepetitionFeedback(
         t,y,
         range=range_,
         epoch=start_time)
  hctrl = SpaceRepetitionController(
         reference=hr,
         feedback=hf,
         range=range_,
         epoch=start_time)
  hctrl.plot_graphs()

  #pp(hctrl.schedule)

  #hctrl.show()
  #hctrl.save_figure()
  #plt.close('all')

def test_series():
#  """
#    base['frame']['0']['recommendation']['long_term_potentiation']
#      -> list of {'x': value_x, 'y': value_y }
#    base['frame']['0']['recommendation']['moments']
#      -> list of {'x': value_x, 'y': value_y }
#    base['frame']['0']['recommendation']['forgetting']
#      -> list of {'x': value_x, 'y': value_y }
#
#    base['frame']['0']['error']
#      -> list of {'x': value_x, 'y': value_y }
#
#    base['frame']['0']['feedback']['long_term_potentiation']
#      -> list of {'x': value_x, 'y': value_y }
#    base['frame']['0']['feedback']['moments']
#      -> list of {'x': value_x, 'y': value_y }
#    base['frame']['0']['feedback']['forgetting']
#      -> list of {'x': value_x, 'y': value_y }
#
#    base['frame']['0']['control']['reference_potentiation_with_offset']
#      -> list of {'x': value_x, 'y': value_y }
#    base['frame']['0']['control']['feedback_potentiation_with_offset']
#      -> list of {'x': value_x, 'y': value_y }
#    base['frame']['0']['control']['reference_forgetting_with_offset']
#      -> list of {'x': value_x, 'y': value_y }
#    base['frame']['0']['control']['moments']
#      -> list of {'x': value_x, 'y': value_y }
#
#  """
  base = {}
  base["frame"] = {}

#  start_time = datetime.now()
#  data_file  = "test_run.json"
#  t, y       = build_time_and_results(start_time, 1)
#  x, y       = get_feedback1()
#  range_     = x[-1] + x[-1] * 0.5
#  hr         = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
#  hf         = SpaceRepetitionFeedback(t[0:2], y, range=range_, epoch=start_time)
#  hctrl      = SpaceRepetitionController(reference=hr, feedback=hf, range=range_, epoch=start_time)
#  # print(hctrl.schedule)  # this is what you want
#  data_dict, _  = hctrl.plot_graphs()
#
#  base["frame"]["0"] = dict(data_dict)
#  data_dict.clear()
#  hctrl.save_figure("spaced_0.pdf")
#
#  hr = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
#  hf = SpaceRepetitionFeedback(t[0:3], y ,range=range_, epoch=start_time)
#  hctrl.initialize_feedback(feedback=hf)
#  data_dict, _  = hctrl.plot_graphs()
#
#  base["frame"]["1"] = dict(data_dict)
#  data_dict.clear()
#  hctrl.save_figure("spaced_1.pdf")
#
#  hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
#  hf = SpaceRepetitionFeedback(t[0:4],y,range=range_,epoch=start_time)
#  hctrl.initialize_feedback(feedback=hf)
#  data_dict, _ = hctrl.plot_graphs()
#
#  base["frame"]["2"] = dict(data_dict)
#  data_dict.clear()
#  hctrl.save_figure("spaced_2.pdf")
#
#  hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
#  hf = SpaceRepetitionFeedback(t[0:5],y,range=range_,epoch=start_time)
#  hctrl.initialize_feedback(feedback=hf)
#  data_dict, _ = hctrl.plot_graphs()
#
#  base["frame"]["3"] = dict(data_dict)
#  data_dict.clear()
#  hctrl.save_figure("spaced_3.pdf")
#
#  hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
#  hf = SpaceRepetitionFeedback(t[0:6],y,range=range_,epoch=start_time)
#  hctrl.initialize_feedback(feedback=hf)
#  data_dict, _ = hctrl.plot_graphs()
#
#  base["frame"]["4"] = dict(data_dict)
#  data_dict.clear()
#  hctrl.save_figure("spaced_4.pdf")
#
#  hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
#  hf = SpaceRepetitionFeedback(t[0:7],y,range=range_,epoch=start_time)
#  hctrl.initialize_feedback(feedback=hf)
#  data_dict, _ = hctrl.plot_graphs()
#  base["frame"]["5"] = dict(data_dict)
#  data_dict.clear()
#  hctrl.save_figure("spaced_5.pdf")
#  #hctrl.open_figure("spaced_5.pdf")
#
#  hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
#  hf = SpaceRepetitionFeedback(t[0:8], y, range=range_,epoch=start_time)
#  hctrl.initialize_feedback(feedback=hf)
#  data_dict, graph_handle = hctrl.plot_graphs()
#
#  graph_handle.epoch      # the epoch
#  graph_handle.axarr      # array of matplotlib axes mapping the subplots
#  graph_handle.figure     # the matplotlib figure
#  graph_handle.axarr[-1]  # the control plot
#
#  base["frame"]["6"] = dict(data_dict)
#  data_dict.clear()
#  hctrl.save_figure("spaced_6.pdf")
#
#  base['range'] = range_
#  with open(data_file, 'w') as outfile:
#    json.dump(base, outfile, ensure_ascii=False, sort_keys=True, indent=2)

  #pp(hctrl.schedule)

#def test_learning_tracker():
#  lt  = LearningTracker()
#  x, y = get_feedback1()
#  for x_, y_ in zip(x, y):
#    lt.add_event(x_, y_)
#  lt.animate(name_of_mp4="example.mp4", artist="7 learning events")

def test_predictions():
  base = {}
  base["frame"] = {}

  start_time = datetime.now()
  x, y       = get_feedback1()
  t          = [start_time + timedelta(days=offset) for offset in x]
  range_     = x[-1] + x[-1] * 0.5
  hr         = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
  hf         = SpaceRepetitionFeedback(t[0:5], y, range=range_, epoch=start_time)
  hctrl      = SpaceRepetitionController(reference=hr, feedback=hf, range=range_, ephoch=start_time)
  # plot the reference suggestion, the feedback, error and the updated training
  # suggestions (control)
  data_dict, graph_handle  = hctrl.plot_graphs()

  # Now make some predictions about a students ability to recollect something at
  # future points of time... ask your question using datetimes and plot your
  # answer on the same graph made above
  last_feeback_moment = hctrl.control_x
  training_moments = [start_time + 
                      timedelta(days=days) +
                      timedelta(days=last_feeback_moment) 
                        for days
                          in hctrl.updated_reference.ref_events_x[2:-1]]
                          #in [0] + hctrl.updated_reference.ref_events_x]

  results = [hctrl.recollect_scalar(training_moment, curve=3) for training_moment in training_moments]
  for result in results:
    print(result)
  times = [moment for moment in training_moments]
  #control_plot = graph_handle.axarr[-1]
  #control_plot.plot(times, results, color='xkcd:azure')

  # Ask the same question using day offsets from epoch
  curve = 3
  training_moments = hctrl.range_for(curve=curve, day_step_size=0.5)
  results = [hctrl.recollect_scalar(training_moment, curve=curve) for training_moment in training_moments]
  control_plot = graph_handle.axarr[-1]
  control_plot.plot(training_moments, results, color='xkcd:azure')

  # use the original reference to make a prediction
  ref_training_moments = [
    start_time + timedelta(days=days) 
      for days 
      #in [0] + hr.ref_events_x
      in hr.ref_events_x[1:-1]
  ]
  ref_results = [hr.recollect_scalar(ref_training_moment, curve=1) for ref_training_moment in ref_training_moments]
  reference_plot = graph_handle.axarr[0]
  reference_plot.plot(ref_training_moments, ref_results, color='xkcd:teal')

  hctrl.save_figure("spaced_predict.pdf")

def test_datetime_for_curve():
  start_time = datetime.now()
  x, y       = get_feedback1()
  t          = [start_time + timedelta(days=offset) for offset in x]
  range_     = x[-1] + x[-1] * 0.5
  hr         = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
  hf         = SpaceRepetitionFeedback(t[0:5], y, range=range_, epoch=start_time)
  hctrl      = SpaceRepetitionController(reference=hr, feedback=hf, range=range_, ephoch=start_time)

  result = hctrl.datetime_for(curve=1)
  moments = hctrl.range_for(curve=1)
#  graph_handle.epoch      # the epoch
#  graph_handle.axarr      # array of matplotlib axes mapping the subplots
#  graph_handle.figure     # the matplotlib figure
#  graph_handle.axarr[-1]  # the control plot
