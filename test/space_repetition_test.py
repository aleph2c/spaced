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

def longterm_feedback():
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
  return [x_v, y_v]

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

@pytest.mark.reference
def test_reference_deque_maxlen():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  assert hr.maxlen == SpaceRepetitionReference.Max_Length

@pytest.mark.reference
def test_reference_sizeof_growing_deques():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  # the deques should grow at the same rate, otherwise
  # we will have ring buffer slippage
  assert len(hr.forgetting_enclosures) == len(hr.new_dates)
  assert len(hr.forgetting_enclosures) == len(hr.new_results)

@pytest.mark.reference
def test_reference_scheduled_offsets_from_generator():
  hr = SpaceRepetitionReference(epoch=datetime.now())

  targets = [
    0.0,
    0.4589442709995035,
    0.8952268306891566,
    1.3571621263040674,
    1.881320415336438,
    2.517823066268917,
    3.3604547989757783,
    4.635636774985627,
    7.118908457377074,
    16.123000823500735
  ]

  for r, t in zip(hr.schedule_as_offset(stop=40), targets):
    assert abs(r-t) <= 0.001    

@pytest.mark.reference
def test_reference_scheduled_datetimes_from_generator():
  start_time = datetime.now()
  hr = SpaceRepetitionReference(epoch=datetime.now())

  targets = [
    0.0,
    0.4589442709995035,
    0.8952268306891566,
    1.3571621263040674,
    1.881320415336438,
    2.517823066268917,
    3.3604547989757783,
    4.635636774985627,
    7.118908457377074,
    16.123000823500735
  ]

  datetime_targets = [
    start_time + timedelta(days=t) for
    t in targets]

  stop_date = start_time + timedelta(days=40)
  for r, t in zip(hr.schedule(stop_date), datetime_targets):
    assert abs(r-t) <= 0.001

@pytest.mark.reference
def test_reference_graph_too_short():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  hdl, _ = hr.plot_graph(stop_date=4)
  hr.save_figure("results/space_reference_too_short_plot.pdf")
  hdl.close()

@pytest.mark.reference
def test_reference_graph_default():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  hdl, _ = hr.plot_graph(stop_date=43)
  hr.save_figure("results/space_reference_plot.pdf")
  hdl.close()

@pytest.mark.reference
def test_reference_graph_too_long():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  hdl, _ = hr.plot_graph(stop_date=60)
  hr.save_figure("results/space_reference_too_long_plot.pdf")
  hdl.close()

@pytest.mark.reference
def test_reference_plot_closing_features():
  start_time = datetime.now()
  hr         = SpaceRepetitionReference(epoch=start_time)
  hdl, _ = hr.plot_graph(stop_date=42)
  hdl.close()

@pytest.mark.reference
def test_reference_prediction_range_for_feature():
  start_time = datetime.now()
  hr = SpaceRepetitionReference(
    epoch=start_time,
    plasticity_root=0.03699,
    plasticity_denominator_offset=0.0054,
  )
  training_moments = hr.range_for(stop_at=30, curve=1, day_step_size=0.1)
  assert training_moments[0] == hr.days_offset_from_epoch_to_datetime(hr.new_dates[0])
  assert training_moments[-1] == \
    hr.days_offset_from_epoch_to_datetime(30) - timedelta(days=0.1)

  training_moments = hr.range_for(stop_at=30, curve=2, day_step_size=0.1)
  assert training_moments[0] == hr.days_offset_from_epoch_to_datetime(hr.new_dates[1])

@pytest.mark.reference
def test_reference_prediction_feature():
  start_time = datetime.now()
  hr = SpaceRepetitionReference(
    epoch=start_time,
  )
  
  graph_handle, data_dict  = hr.plot_graph(stop_date=30)
  training_moments_1 = hr.range_for(stop_at=30, curve=1, day_step_size=0.1)
  results_1 = [hr.recollect_scalar(training_moment, curve=1)
    for training_moment in training_moments_1]
  training_moments_2 = hr.range_for(stop_at=30, curve=2, day_step_size=0.1)
  results_2 = [hr.recollect_scalar(training_moment, curve=2)
    for training_moment in training_moments_2]
  training_moments_3 = hr.range_for(stop_at=30, curve=3, day_step_size=0.1)
  results_3 = [hr.recollect_scalar(training_moment, curve=3)
    for training_moment in training_moments_3]
  reference_plot = graph_handle.axarr
  reference_plot.plot(training_moments_1, results_1, color='xkcd:azure')
  reference_plot.plot(training_moments_2, results_2, color='xkcd:darkgreen')
  reference_plot.plot(training_moments_3, results_3, color='xkcd:maroon')
  hr.save_figure("results/space_reference_predictions.pdf")
  graph_handle.close()

@pytest.mark.feedback
def test_feedback_graph_too_short():
  x, y = get_feedback1()
  start_time = datetime.now()
  hr = SpaceRepetitionReference(epoch=start_time)
  hf = SpaceRepetitionFeedback(x, y, epoch=start_time)
  hdl, _ = hf.plot_graph(stop_date=4)
  hr.save_figure("results/space_feedback_too_short.pdf")
  hdl.close()

@pytest.mark.feedback
def test_feedback_graph_fit_feedback():
  x, y = get_feedback1()
  start_time = datetime.now()
  hr = SpaceRepetitionReference(epoch=start_time)
  hf = SpaceRepetitionFeedback(x, y, epoch=start_time)
  hdl, _ = hf.plot_graph()  # it should automatically fit the data
  hr.save_figure("results/space_feedback_plot.pdf")
  hdl.close()

@pytest.mark.feedback
def test_feedback_graph_too_long():
  x, y = get_feedback1()
  start_time = datetime.now()
  hr = SpaceRepetitionReference(epoch=start_time)
  hf = SpaceRepetitionFeedback(x, y, epoch=start_time)
  hdl, _ = hf.plot_graph(stop_date=50)  # it should automatically fit the data
  hr.save_figure("results/space_feedback_too_long.pdf")
  hdl.close()

@pytest.mark.control
def test_generator_controller():
  start_time = datetime.now()
  x, y = get_feedback1()

  hr = SpaceRepetitionReference(epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:4], y, epoch=start_time)
  hctrl = SpaceRepetitionController(
    reference=hr,
    feedback=hf,
    epoch=start_time
  )

  hdl, _ = hctrl.plot_graphs(stop_date=43)
  hctrl.save_figure("results/space_control.pdf")
  hdl.close()

@pytest.mark.control
def test_a_control_series():
  start_time = datetime.now()
  x, y       = get_feedback1()
  stop_date  = 2 * x[-1] * 0.5
  hr         = SpaceRepetitionReference(
                plot=False,
                epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:2], y, epoch=start_time)
  hctrl = SpaceRepetitionController(reference=hr, feedback=hf, epoch=start_time)
  graph_handle, data_dict  = hctrl.plot_graphs(stop_date=stop_date)
  hctrl.save_figure("results/spaced_0.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:3], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop_date=stop_date)
  hctrl.save_figure("results/spaced_1.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:4], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _  = hctrl.plot_graphs(stop_date=stop_date)
  hctrl.save_figure("results/spaced_2.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:5], y,epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop_date=stop_date)
  data_dict.clear()
  hctrl.save_figure("results/spaced_3.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:6], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop_date=stop_date)
  data_dict.clear()
  hctrl.save_figure("results/spaced_4.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:7], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop_date=stop_date)
  data_dict.clear()
  hctrl.save_figure("results/spaced_5.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:8], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop_date=stop_date)
  graph_handle.epoch      # the epoch
  graph_handle.axarr      # array of matplotlib axes mapping the subplots
  graph_handle.figure     # the matplotlib figure
  graph_handle.axarr[-1]  # the control plot

  data_dict.clear()
  hctrl.save_figure("results/spaced_6.pdf")
  graph_handle.close()

@pytest.mark.control
def test_control_predition_range_for_feature():
  start_time = datetime.now()
  x, y = get_feedback1()
  hr = SpaceRepetitionReference(epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:4], y, epoch=start_time)
  hctrl = SpaceRepetitionController(epoch=start_time, reference=hr, feedback=hf)
  training_moments = hctrl.range_for(stop_at=30, curve=1, day_step_size=0.1)
  assert training_moments[0] == hctrl.days_offset_from_epoch_to_datetime(hctrl.new_dates[0])

  training_moments = hctrl.range_for(stop_at=30, curve=2, day_step_size=0.1)
  assert training_moments[0] == hctrl.days_offset_from_epoch_to_datetime(hctrl.new_dates[1])

@pytest.mark.control
def test_control_predition_for_feature():
  start_time = datetime.now()
  x, y = get_feedback1()
  hr = SpaceRepetitionReference(epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:4], y, epoch=start_time)
  hctrl = SpaceRepetitionController(epoch=start_time, reference=hr, feedback=hf)

  graph_handle, data_dict = hctrl.plot_graphs(stop_date=30)
  training_moments_1 = hctrl.range_for(stop_at=30, curve=1, day_step_size=0.1)
  results_1 = [hctrl.recollect_scalar(training_moment, curve=1)
    for training_moment in training_moments_1]
  training_moments_2 = hctrl.range_for(stop_at=30, curve=2, day_step_size=0.1)
  results_2 = [hctrl.recollect_scalar(training_moment, curve=2)
    for training_moment in training_moments_2]
  training_moments_3 = hr.range_for(stop_at=30, curve=3, day_step_size=0.1)
  results_3 = [hctrl.recollect_scalar(training_moment, curve=3)
    for training_moment in training_moments_3]
  ctrl_plot = graph_handle.axarr[-1]
  ctrl_plot.plot(training_moments_1, results_1, color='xkcd:azure')
  ctrl_plot.plot(training_moments_2, results_2, color='xkcd:darkgreen')
  ctrl_plot.plot(training_moments_3, results_3, color='xkcd:maroon')
  hr.save_figure("results/space_ctrl_predictions.pdf")
  graph_handle.close()

@pytest.mark.learning_tracker
def test_learning_tracker_graph_to_short():
  pass

@pytest.mark.learning_tracker
def test_learning_tracker_graph_default():
  pass

@pytest.mark.learning_tracker
def test_learning_tracker_graph_too_long():
  pass

@pytest.mark.learning_tracker
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
    time_per_event_in_seconds=1.0,
    stop_date=30,
  )

#@pytest.mark.learning_tracker
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

@pytest.mark.longterm
def test_longterm_response():
  start_time = datetime.now()

  lt = LearningTracker(
    epoch=start_time,
    range=500,
    long_term_clamp=0.00005
  )

  offsets = [lt.days_from_epoch(dt) for dt in lt.schedule()]

  moments, results = longterm_feedback()
  for moment, result in zip(moments, results):
    lt.learned(when=moment, result=result)

  hdl, _ = lt.plot_graphs()
  lt.save_figure("results/longterm.pdf")
  hdl.close()
  #print(offsets)

  

