from repetition import *
from graph import *
import numpy as np
from datetime import datetime
from datetime import timedelta
import pickle
import pytest

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
          23.029014507253628,
          30.0,
          60,
        ]
  y_v = [1,
          0.44646101201172317,
          0.64510969552772512,
          0.76106659335521121,
          0.93741905134093808,
          0.99000147547559727,
          0.99902389355455017,
          1.00000000000000000,
          0.90000000000000000,
          0.7,
          0.90000000000000000,
          ]
  return [x_v,y_v]

def get_feedback3():
  x_v = [
          0,
          0.80640320160080037,
          1.20640320160080037,
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
          0.0,
          0.40,
          1.0,
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

  start_time = datetime.now()
  x, y       = get_feedback1()
  range_     = x[-1] + x[-1] * 0.5
  hr         = SpaceRepetitionReference(
                plot=False,
                range=range_,
                epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:2], y, range=range_, epoch=start_time)
  hctrl = SpaceRepetitionController(reference=hr, feedback=hf, range=range_, epoch=start_time)
  graph_handle, data_dict  = hctrl.plot_graphs()
  hctrl.save_figure("results/spaced_0.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:3], y ,range=range_, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs()
  hctrl.save_figure("results/spaced_1.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:4],y,range=range_,epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _  = hctrl.plot_graphs()
  hctrl.save_figure("results/spaced_2.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:5],y,range=range_,epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs()

  data_dict.clear()
  hctrl.save_figure("results/spaced_3.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:6],y,range=range_,epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs()

  data_dict.clear()
  hctrl.save_figure("results/spaced_4.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:7],y,range=range_,epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs()
  data_dict.clear()
  hctrl.save_figure("results/spaced_5.pdf")
  graph_handle.close()
  #hctrl.open_figure("spaced_5.pdf")

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:8], y, range=range_,epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs()

  graph_handle.epoch      # the epoch
  graph_handle.axarr      # array of matplotlib axes mapping the subplots
  graph_handle.figure     # the matplotlib figure
  graph_handle.axarr[-1]  # the control plot

  data_dict.clear()
  hctrl.save_figure("results/spaced_6.pdf")
  graph_handle.close()

def test_learning_tracker():
  epoch = datetime.now()

  lt = LearningTracker(
    epoch=epoch,
    feedback_data=get_feedback1(),
    plasticity_root=1.4, 
    fdecaytau=1.87,
    fdecay0 = 0.9,
  ).animate(
    name_of_mp4="results/example.mp4",
    student="Marnie MacMillan",
    time_per_event_in_seconds=1.0
  )

def test_predictions():
  base = {}
  base["frame"] = {}

  start_time = datetime.now()
  x, y       = get_feedback1()
  t          = [start_time + timedelta(days=offset) for offset in x]
  range_     = x[0] + x[-1] * 0.5
  hr = SpaceRepetitionReference(
      plot=False,
      range=range_,
      epoch=start_time,
      plasticity_root=0.03699,
      plasticity_denominator_offset=0.0054,
      )
  hf = SpaceRepetitionFeedback(x[0:5], y, range=range_, epoch=start_time)
  hctrl = SpaceRepetitionController(
      reference=hr,
      feedback=hf,
      range=range_,
      epoch=start_time,
  )
  # plot the reference suggestion, the feedback, error and the updated training
  # suggestions (control)
  graph_handle, data_dict  = hctrl.plot_graphs()

  # Ask a question using day offsets from epoch
  curve = 1
  #import pdb; pdb.set_trace()
  training_moments = hctrl.range_for(curve=curve, day_step_size=0.01)
  #training_moments = [training_moment - timedelta(days=0.666) for training_moment in training_moments]
  print(training_moments[0])
  print(hctrl.schedule())
  #import pdb; pdb.set_trace()
  results = [hctrl.recollect_scalar(training_moment, curve=curve) for training_moment in training_moments]
  control_plot = graph_handle.axarr[-1]
  control_plot.plot(training_moments, results, color='xkcd:azure')
  print(hctrl.schedule())

  curve = 2
  ref_training_moments = hr.range_for(curve=curve, day_step_size=0.5)
  ref_results = [hr.recollect_scalar(ref_training_moment, curve=curve) for ref_training_moment in ref_training_moments]
  reference_plot = graph_handle.axarr[0]
  reference_plot.plot(ref_training_moments, ref_results, color='xkcd:teal')

  hctrl.save_figure("results/spaced_predict.pdf")
  graph_handle.close()

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

def test_closing_features():
  start_time = datetime.now()
  x, y       = get_feedback1()
  t          = [start_time + timedelta(days=offset) for offset in x]
  range_     = x[-1] + x[-1] * 0.5
  hr         = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
  hdl, _ = hr.plot_graph()
  hdl.close()
  
def test_predictions_2():
  epoch = datetime.now()
  hr = SpaceRepetitionReference(
    epoch=epoch,
    plasticity_root=1.4, 
    fdecaytau=1.87,
    fdecay0 = 0.9,
    )
  x, y = get_feedback3()
  range_ = x[-1] + x[-1] * 0.5
  for index in range(1, len(x)):

    hf = SpaceRepetitionFeedback(
      x[0:index],
      y,
      range=range_,
      epoch=epoch)

    hctrl = SpaceRepetitionController(
      reference=hr, 
      feedback=hf, 
      range=range_, 
      epoch=epoch)

    graph_handle, data_dict = hctrl.plot_graphs()
    hctrl.save_figure("results/spaced_b_{}.pdf".format(index))
    graph_handle.close()
    data_dict.clear()


def test_simplified_interface():

  lt = LearningTracker(
    epoch = datetime.now(),
    plasticity_root=1.8, 
  )

  moments, results = get_feedback1()
  for index, (moment, result) in enumerate(zip(moments, results)):
    #print(lt.feedback.discovered_plasticity_root, lt.feedback.discovered_plasticity_denominator_offset)
    #if index == 4:
    # break
    lt.learned(when=moment, result=result)

  hdl, _ = lt.plot_graphs()
  lt.save_figure("results/learning_tracker_guessed_parameters.pdf")
  hdl.close()

  lt_next = LearningTracker(
    epoch = datetime.now(),
    plasticity_root=lt.discovered_plasticity_root(), 
    plasticity_denominator_offset=lt.discovered_plasticity_denominator_offset(), 
    fdecay0=lt.discovered_fdecay0(),
    fdecaytau=lt.discovered_fdecaytau()
  )

  moments, results = get_feedback1()
  for index, (moment, result) in enumerate(zip(moments, results)):
    lt_next.learned(when=moment, result=result)
    if index == 3:
      break

  hdl, _ = lt_next.plot_graphs()
  lt_next.save_figure("results/learning_tracker_learned_parameters.pdf")
  hdl.close()

@pytest.mark.pickle
def test_serialization_epoch():
  lt = LearningTracker(
    epoch = datetime.now(),
  )
  byte_stream = pickle.dumps(lt)
  unpickled_learning_tracker = pickle.loads(byte_stream)
  hdl, _ = unpickled_learning_tracker.plot_graphs()
  unpickled_learning_tracker.save_figure("results/post_pickle.pdf")
  hdl.close()


