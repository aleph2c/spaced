# 
from repetition import LearningTracker
from repetition import SpaceRepetitionReference
from repetition import SpaceRepetitionFeedback
from repetition import SpaceRepetitionController
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
          0.02,
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
          0.30,
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
  y_v = [ 1,
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

@pytest.mark.reference
def test_reference_deque_maxlen():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  assert hr.maxlen == SpaceRepetitionReference.Max_Length

@pytest.mark.reference
def test_reference_sizeof_growing_deques():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  # the deques should grow at the same rate, otherwise
  # we will have ring buffer slippage
  assert len(hr.forgetting_enclosures) == len(hr.dates_as_day_offsets)
  assert len(hr.forgetting_enclosures) == len(hr.results_at_training_moments)

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
  for r, t in zip(hr.schedule(stop=stop_date), datetime_targets):
    assert abs((r-t).total_seconds()) <= 0.001

@pytest.mark.reference
def test_reference_graph_too_short():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  hdl, _ = hr.plot_graph(stop=4)
  hr.save_figure("results/space_reference_too_short_plot.pdf")
  hdl.close()

@pytest.mark.reference
def test_reference_graph_default():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  hdl, _ = hr.plot_graph(stop=43)
  hr.save_figure("results/space_reference_plot.pdf")
  hdl.close()

@pytest.mark.reference
def test_reference_graph_too_long():
  hr = SpaceRepetitionReference(epoch=datetime.now())
  hdl, _ = hr.plot_graph(stop=60)
  hr.save_figure("results/space_reference_too_long_plot.pdf")
  hdl.close()

@pytest.mark.reference
def test_reference_plot_closing_features():
  start_time = datetime.now()
  hr         = SpaceRepetitionReference(epoch=start_time)
  hdl, _ = hr.plot_graph(stop=42)
  hdl.close()

@pytest.mark.reference
def test_reference_prediction_range_for_feature():
  start_time = datetime.now()
  hr = SpaceRepetitionReference(
    epoch=start_time,
    plasticity_root=0.03699,
    plasticity_denominator_offset=0.0054,
  )
  training_moments = hr.range_for(stop=30, curve=1, day_step_size=0.1)
  assert training_moments[0] == hr.days_offset_from_epoch_to_datetime(hr.dates_as_day_offsets[0])
  assert training_moments[-1] == \
    hr.days_offset_from_epoch_to_datetime(30) - timedelta(days=0.1)

  training_moments = hr.range_for(stop=30, curve=2, day_step_size=0.1)
  assert training_moments[0] == hr.days_offset_from_epoch_to_datetime(hr.dates_as_day_offsets[1])

@pytest.mark.reference
def test_reference_prediction_feature():
  start_time = datetime.now()
  hr = SpaceRepetitionReference(
    epoch=start_time,
  )
  
  graph_handle, data_dict  = hr.plot_graph(stop=30)
  training_moments_1 = hr.range_for(stop=30, curve=1, day_step_size=0.1)
  results_1 = [hr.recollect_scalar(training_moment, curve=1)
    for training_moment in training_moments_1]
  training_moments_2 = hr.range_for(stop=30, curve=2, day_step_size=0.1)
  results_2 = [hr.recollect_scalar(training_moment, curve=2)
    for training_moment in training_moments_2]
  training_moments_3 = hr.range_for(stop=30, curve=3, day_step_size=0.1)
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
  hdl, _ = hf.plot_graph(stop=4)
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
  hdl, _ = hf.plot_graph(stop=50)  # it should automatically fit the data
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

  hdl, _ = hctrl.plot_graphs(stop=43)
  hctrl.save_figure("results/space_control.pdf")
  hdl.close()

@pytest.mark.control
def test_control_graph_too_short():
  start_time = datetime.now()
  x, y       = get_feedback1()
  stop_date  = 5
  hr         = SpaceRepetitionReference(
                epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:2], y, epoch=start_time)
  hctrl = SpaceRepetitionController(reference=hr, feedback=hf, epoch=start_time)
  graph_handle, data_dict  = hctrl.plot_graphs(stop=stop_date)
  hctrl.save_figure("results/spaced_control_too_short.pdf")
  graph_handle.close()

@pytest.mark.control
def test_control_graph_too_long():
  start_time = datetime.now()
  x, y       = get_feedback1()
  stop_date  = 1000
  hr         = SpaceRepetitionReference(
                epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:2], y, epoch=start_time)
  hctrl = SpaceRepetitionController(reference=hr, feedback=hf, epoch=start_time)
  graph_handle, data_dict  = hctrl.plot_graphs(stop=stop_date)
  hctrl.save_figure("results/spaced_control_too_long.svg")
  hctrl.save_figure("results/spaced_control_too_long.pdf")
  graph_handle.close()

@pytest.mark.control
def test_a_control_series():
  start_time = datetime.now()
  x, y       = get_feedback1()
  stop_date  = 2 * x[-1] * 0.5
  hr         = SpaceRepetitionReference(
                epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:2], y, epoch=start_time)
  hctrl = SpaceRepetitionController(reference=hr, feedback=hf, epoch=start_time)
  graph_handle, data_dict  = hctrl.plot_graphs(stop=stop_date)
  hctrl.save_figure("results/spaced_0.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False, range=range_, epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:3], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop=stop_date)
  hctrl.save_figure("results/spaced_1.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:4], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _  = hctrl.plot_graphs(stop=stop_date)
  hctrl.save_figure("results/spaced_2.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:5], y,epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop=stop_date)
  data_dict.clear()
  hctrl.save_figure("results/spaced_3.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:6], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop=stop_date)
  data_dict.clear()
  hctrl.save_figure("results/spaced_4.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:7], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop=stop_date)
  data_dict.clear()
  hctrl.save_figure("results/spaced_5.pdf")
  graph_handle.close()

  #hr = SpaceRepetitionReference(plot=False,range=range_,epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:8], y, epoch=start_time)
  hctrl.initialize_feedback(feedback=hf)
  graph_handle, _ = hctrl.plot_graphs(stop=stop_date)
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
  training_moments = hctrl.range_for(stop=30, curve=1, day_step_size=0.1)
  assert training_moments[0] == hctrl.days_offset_from_epoch_to_datetime(hctrl.dates_as_day_offsets[0])

  training_moments = hctrl.range_for(stop=30, curve=2, day_step_size=0.1)
  assert training_moments[0] == hctrl.days_offset_from_epoch_to_datetime(hctrl.dates_as_day_offsets[1])

@pytest.mark.control
def test_control_predition_for_feature():
  start_time = datetime.now()
  x, y = get_feedback1()
  hr = SpaceRepetitionReference(epoch=start_time)
  hf = SpaceRepetitionFeedback(x[0:4], y, epoch=start_time)
  hctrl = SpaceRepetitionController(epoch=start_time, reference=hr, feedback=hf)

  graph_handle, data_dict = hctrl.plot_graphs(stop=30)
  training_moments_1 = hctrl.range_for(stop=30, curve=1, day_step_size=0.1)
  results_1 = [hctrl.recollect_scalar(training_moment, curve=1)
    for training_moment in training_moments_1]
  training_moments_2 = hctrl.range_for(stop=30, curve=2, day_step_size=0.1)
  results_2 = [hctrl.recollect_scalar(training_moment, curve=2)
    for training_moment in training_moments_2]
  training_moments_3 = hr.range_for(stop=30, curve=3, day_step_size=0.1)
  results_3 = [hctrl.recollect_scalar(training_moment, curve=3)
    for training_moment in training_moments_3]
  ctrl_plot = graph_handle.axarr[-1]
  ctrl_plot.plot(training_moments_1, results_1, color='xkcd:azure')
  ctrl_plot.plot(training_moments_2, results_2, color='xkcd:darkgreen')
  ctrl_plot.plot(training_moments_3, results_3, color='xkcd:maroon')
  hr.save_figure("results/space_ctrl_predictions.pdf")
  graph_handle.close()

@pytest.mark.learning_tracker
def test_learning_tracker_graph_too_short():
  start_time = datetime.now()
  moments, results = get_feedback1()

  lt = LearningTracker(epoch=start_time)
  for index, (moment, result) in enumerate(zip(moments[0:2], results)):
    lt.learned(when=moment, result=result)

  hd, _ = lt.plot_graphs(stop=5)
  lt.save_figure("results/space_learning_tracker_too_short.pdf")
  hd.close()

@pytest.mark.longterm
@pytest.mark.learning_tracker
def test_learning_tracker_longterm_response():
  start_time = datetime.now()
  moments, results = get_feedback1()

  lt = LearningTracker(epoch=start_time)
  for index, (moment, result) in enumerate(zip(moments[0:2], results)):
    lt.learned(when=moment, result=result)

  hd, _ = lt.plot_graphs(stop=300)
  lt.save_figure("results/space_learning_tracker_too_long.pdf")
  hd.close()

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
    stop=30,
  )

  # adding this to remove pyflakes warning
  assert lt

@pytest.mark.pickle
@pytest.mark.learning_tracker
def test_serialization_epoch():
  lt = LearningTracker(
    epoch = datetime.now(),
  )
  moments, results = get_feedback1()
  for index, (moment, result) in enumerate(zip(moments[0:4], results)):
    lt.learned(when=moment, result=result)

  byte_stream = pickle.dumps(lt)
  unpickled_learning_tracker = pickle.loads(byte_stream)
  hdl, _ = unpickled_learning_tracker.plot_graphs(stop=40)
  unpickled_learning_tracker.save_figure("results/post_pickle.pdf")
  hdl.close()


@pytest.mark.learning_tracker
def test_learning_tracker_scheduled_offsets_from_generator():
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

@pytest.mark.learning_tracker
def test_learning_tracker_scheduled_datetimes_from_generator_1():
  lt = LearningTracker(epoch=datetime.now())

  targets = [
    0.4589442709995035,
    0.8952268306895583,
    1.357162126293895,
    1.8813204153239627,
    2.5178230662543553,
    3.360454798955533,
    4.635636774952296,
    7.118908457356618,
    16.123000823378668
  ]

  for (index, result) in enumerate(lt.schedule_as_offset(stop=40)):
    assert( abs(targets[index] - result) < 0.001)

@pytest.mark.learning_tracker
def test_learning_tracker_scheduled_should_output_dates_after_feedback():
  lt = LearningTracker(epoch=datetime.now())

  lt.learned(result=0.40, when=0.01)
  lt.learned(result=0.44, when=0.8)
  for offset in lt.schedule_as_offset(stop=10):
    assert offset > 0.8

@pytest.mark.learning_tracker
def test_learning_tracker_graph_context_manager_xxxx_handle():
 
  lt = LearningTracker(epoch=datetime.now())
  with lt.graphs(stop=43, filename='results/lt_content_xxxx_handle.pdf'):
    pass

@pytest.mark.learning_tracker
def test_learning_tracker_graph_context_manager_rxxc_handle():
 
  lt = LearningTracker(epoch=datetime.now())
  moments = lt.range_for(curve=2, stop=43, day_step_size=0.5)
  predictions = [lt.recollect_scalar(moment, curve=2) for moment in moments]

  with lt.graphs(stop=43, filename='results/lt_content_rxxc_handle.pdf',
    control_handle=True, reference_handle=True, show=True) as (rh, ch):

    rh.plot(moments, predictions, color='xkcd:azure')
    ch.plot(moments, predictions, color='xkcd:azure')

@pytest.mark.learning_tracker
def test_learning_tracker_graph_context_manager_rfxc_handle():
 
  lt = LearningTracker(epoch=datetime.now())
  moments = lt.range_for(curve=1, stop=43, day_step_size=0.5)
  predictions = [lt.recollect_scalar(moment, curve=1) for moment in moments]

  with lt.graphs(stop=43, filename='results/lt_content_rfxc_handle.pdf',
    control_handle=True, reference_handle=True, feedback_handle=True) as (rh, fh, ch):
    rh.plot(moments, predictions, color='xkcd:azure')
    fh.plot(moments, predictions, color='xkcd:azure')
    ch.plot(moments, predictions, color='xkcd:azure')

@pytest.mark.learning_tracker
def test_learning_tracker_graph_context_manager_rfec_handle():
 
  lt = LearningTracker(epoch=datetime.now())
  moments = lt.range_for(curve=1, stop=43, day_step_size=0.5)
  predictions = [lt.recollect_scalar(moment, curve=1) for moment in moments]

  with lt.graphs(stop=43, 
    filename='results/lt_content_rfec_handle.pdf',
    control_handle=True,
    reference_handle=True,
    feedback_handle=True,
    error_handle=True) as (rh, fh, eh, ch):
    rh.plot(moments, predictions, color='xkcd:azure')
    fh.plot(moments, predictions, color='xkcd:azure')
    eh.plot(moments, predictions, color='xkcd:red')
    ch.plot(moments, predictions, color='xkcd:azure')

@pytest.mark.learning_tracker
def test_learning_tracker_graph_context_manager_xxxc_handle():
 
  lt = LearningTracker(epoch=datetime.now())
  moments = lt.range_for(curve=1, stop=43, day_step_size=0.5)
  predictions = [lt.recollect_scalar(moment, curve=1) for moment in moments]

  with lt.graphs(stop=43, filename='results/lt_content_xxxc_handle.pdf', control_handle=True) as ch:
    ch.plot(moments, predictions, color='xkcd:azure')

@pytest.mark.learning_tracker
def test_learning_tracker_broken_feedback_curve_reproduction():
  lt = LearningTracker(epoch=datetime.now())
  lt.learned(result=0.44, when=0.8)
  lt.learned(result=0.64, when=1.75)
  lt.learned(result=0.55, when=1.3)
  lt.learned(result=0.40, when=0.001)
  lt.learned(result=0.64, when=3.02)
  lt.plot_graphs(stop=30)
  lt.save_figure("results/lt_broken_feedback_curve.pdf")

@pytest.mark.learning_tracker
def test_learning_tracker_next_offset_feature():
  lt = LearningTracker(epoch=datetime.now())
  lt.learned(result=0.40, when=0.001)
  lt.learned(result=0.44, when=0.8)
  lt.learned(result=0.55, when=1.3)
  lt.learned(result=0.64, when=1.75)
  lt.learned(result=0.76, when=3.02)
  results = [
    [4.8,  7.33, 10.93, 16.00, 23.00, 29.00],
    [0.83, 0.89,  1.00,  0.99, 0.99,   1.00],
  ]

  for d, r in zip(*results):
    days_since_training_epoch, result = d, r
    lt.learned(result=result, when=days_since_training_epoch)

  sc = lt.next_offset()
  assert  abs(sc-122.4043) < 0.01 

@pytest.mark.learning_tracker
def test_learning_tracker_next_feature():
  lt = LearningTracker(epoch=datetime.now())
  lt.learned(result=0.40, when=0.001)
  lt.learned(result=0.44, when=0.8)
  lt.learned(result=0.55, when=1.3)
  lt.learned(result=0.64, when=1.75)
  lt.learned(result=0.76, when=3.02)
  results = [
    [4.8,  7.33, 10.93, 16.00, 23.00, 29.00],
    [0.83, 0.89,  1.00,  0.99, 0.99,   1.00],
  ]

  for d, r in zip(*results):
    days_since_training_epoch, result = d, r
    lt.learned(result=result, when=days_since_training_epoch)

  sc = lt.next()
  assert type(sc) is datetime
  assert abs(lt.days_from_epoch(sc)-122.4043) < 0.01 

@pytest.mark.learning_tracker
def test_learning_tracker_broken_two_handle_graph():

	lt = LearningTracker(epoch=datetime.now())

	# give our learning tracker some feedback
	for d, r in zip(
			[0,    0.8,  1.75, 3.02, 4.8,  7.33],
			[0.40, 0.44, 0.64, 0.76, 0.83, 0.89],
		):
		lt.learned(result=r, when=d)

	with lt.graphs(
		stop=43, 
		show=True, 
		control_handle=True, 
		reference_handle=True,
		filename='results/context_manager_two_handles.svg') as (rh, ch):
			
		r_m = lt.reference.range_for(curve=3, stop=43, day_step_size=0.5)
		r_p = [lt.reference.predict_result(moment, curve=3) for moment in r_m]
		rh.plot(r_m, r_p, color='xkcd:ruby')
		c_m = lt.range_for(curve=1, stop=43, day_step_size=0.5)
		c_p = [lt.predict_result(moment, curve=1) for moment in c_m]
		ch.plot(c_m, c_p, color='xkcd:azure')
