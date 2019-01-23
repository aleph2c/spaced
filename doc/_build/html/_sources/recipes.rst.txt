.. _recipes:

  *Simple things should be simple, complex things should be possible.* 
  
  -- Alan Kay

Recipes
=======
Here are a set of recipes that you can reference to learn by example.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. _recipes-diagrams:

Diagrams
--------

.. _recipes-generate-learning-tracker-curves:

Generate Learning Tracker Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To generate the learning tracker curves:

.. code-block:: python

  from datetime import datetime
  from space.repitition import LearningTracker

  days_to_track = 43
  lesson_to_graph = 3

  lt = LearningTracker(epoch=datetime.new())

  day_offset_from_epoch_and_results = [ 
      [0,    0.81, 1.75, 3.02, 4.8,  8.33, 10.93, 16.00, 23.00, 29.00],
      [0.40, 0.44, 0.64, 0.84, 0.83, 0.89,  1.00,  0.99, 0.99,   1.00],
  ]

  # mimic a full training session
  for index, (d, r) in \
    enumerate(zip(*day_offset_from_epoch_and_results)):
    # r: result
    # d: days since training epoch
    lt.learned(result=r, when=d)

    # plot the lesson we want to graph
    if index is lesson_to_graph - 1:
     
      # PLOTTING THE GRAPHS
      hdl, _ = lt.plot_graphs(stop=days_to_track)

The third set of learning tracker plots would look like this:

.. image:: _static/quickstart_arbitrary.svg
    :target: _static/quickstart_arbitrary.pdf
    :align: center

.. _recipes-generate-a-reference-curve:

Generate a Reference Curve
^^^^^^^^^^^^^^^^^^^^^^^^^^
To generate a reference curve:

.. code-block:: python

  from datetime import datetime
  from repetition import LearningTracker

  # create a learning tracker
  learning_tracker = LearningTracker(epoch=datetime.now())
  hdl, _ = learning_tracker.reference.plot_graph(stop=43)

  # to save this plot (usual api, since I can't imagine why a 
  # regular user would save the reference, but I have included it
  # as a recipe since I needed to do this to document this package
  hdl.ppd.plot.savefig("quickstart_reference.svg")
  hdl.ppd.plot.savefig("quickstart_reference.pdf")

  hdl.close()


.. image:: _static/quickstart_reference.svg
    :target: _static/quickstart_reference.pdf
    :align: center

.. _recipes-to-plot-a-set-of-queries:

To Plot a set of Queries
^^^^^^^^^^^^^^^^^^^^^^^^
To see how to plot a set of predictions about future performance look :ref:`here.<quickstart-predicting-future-results>` 

.. _recipes-save-a-diagram:

Save a Diagram
^^^^^^^^^^^^^^
To save a diagram:

.. code-block:: python

  from datetime import datetime
  from space.repitition import LearningTracker
  lt = LearningTracker(epoch=datetime.new())

  hdl, _ = lt.plot_graphs(stop=43)

  # SAVE SVG
  lt.save_figure("replace_with_the_filename_you_want.svg")

  # SAVE PNG
  lt.save_figure("replace_with_the_filename_you_want.png")

  # SAVE PDF
  lt.save_figure("replace_with_the_filename_you_want.pdf")

  hdl.close()

.. _recipes-closing-the-graph-handle:

Closing the Graph Handle
^^^^^^^^^^^^^^^
Under the hook ``spaced`` uses ``matplotlib``.  The ``matplotlib`` library
recommends that graph handles are closed when you are finished looking at the
graph so as to save system memory:

.. code-block:: python

  from datetime import datetime
  from space.repitition import LearningTracker
  lt = LearningTracker(epoch=datetime.new())

  # plot something
  hdl, _ = lt.plot_graphs(stop=43)

  # CLOSE THE GRAPH HANDLE
  hdl.close()

.. _recipes-videos:

Videos
------

.. _recipes-ensure-ffmpeg-is-installed:

Installing ffmpeg
^^^^^^^^^^^^^^^^^
In a linux system or WLS running a Debian distribution:

.. code-block:: bash

  sudo apt update
  sudo apt-get install ffmpeg

.. _recipes-animating-a-learning-tracker:

Animating a Learning Tracker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  from datetime import datetime
  from repetition import LearningTracker

  day_offset_from_epoch_and_results = [ 
      [0,    0.81, 1.75, 3.02, 4.8,  8.33, 10.93, 16.00, 23.00, 29.00],
      [0.40, 0.44, 0.64, 0.84, 0.83, 0.89,  1.00,  0.99, 0.99,   1.00],
  ]

  # create a learning tracker with arbitrary default parameters
  lt_arbitrary = LearningTracker(epoch=datetime.now())

  for d, r in zip(*day_offset_from_epoch_and_results):
    # r: result
    # d: days since training epoch
    lt_arbitrary.learned(result=r, when=d)

  lt.animate(
    student="Name of Student",
    name_of_mp4="results/report_card.mp4",
    time_per_event_in_seconds=2.2,
    stop=43)

This would generate the following ``mp4``:

.. raw:: html

  <center>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/H8llYuwH5L0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </center>

.. _recipes-forgetting-curves:

Forgetting Curves
-----------------

.. _recipes-tuning-your-forgetting-curves:

Tuning your forgetting curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``fdecay0`` and ``fdecaytau`` parameters control the forgetting curves:

.. code-block:: python

  from repetition import LearningTracker
  from datetime import datetime

  start_time = datetime.now()

  # setting up our reference (goals)
  lt = LearningTracker(
    fdecaytau=1.00,  # ability to improve after a lesson, lower is better
    fdecay0=0.9,     # seen it before? then pick lower numbers
    epoch=start_time,
  )

.. _recipes-forgetting-curve-model-discovery:

Forgetting curve model discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use information from one learning tracker session to build a better set of goals
for another:

.. code-block:: python

  from datetime import datetime
  from repetition import LearningTracker

  day_offset_from_epoch_and_results = [ 
      [0,    0.81, 1.75, 3.02, 4.8,  8.33, 10.93, 16.00, 23.00, 29.00],
      [0.40, 0.44, 0.64, 0.84, 0.83, 0.89,  1.00,  0.99, 0.99,   1.00],
  ]

  # create a learning tracker with arbitrary default parameters
  lt_arbitrary = LearningTracker(epoch=datetime.now())

  # mimic a full training session
  for d, r in \
    zip(*day_offset_from_epoch_and_results):
    # r: result
    # d: days since training epoch
    lt_arbitrary.learned(result=r, when=d)

  # get better initial model parameters based on previous
  # experience with the student
    
  # to get the discovered parameter from a previous training session,
  # pre-pend 'discovered' in front of the parameter name,
  # and call this word like a function
  bf0 = lt_arbitrary.discovered_fdecay0()
  bft = lt_arbitrary.discovered_fdecaytau()

  # a better learning tracker
  lt_better_fit = LearningTracker(
    epoch=datetime.now(),
    fdecay0=bf0,
    fdecaytau=bft
  )


You can read more about model learning :ref:`here.<quickstart-building-a-better-initial-student-model>`

.. _recipes-plasticity-curves:

Plasticity Curves
-----------------

The ``plasticity_root`` and ``plasticity_denominator_offset`` parameters control the reference plasticity curve:

.. code-block:: python

  from repetition import LearningTracker
  from datetime import datetime

  start_time = datetime.now()

  #                    x**(1.0/plasticity_root)
  # r = ---------------------------------------------------------
  #     (x+plasticity_denominator_offset)**(1.0/plasticity_root)

  # setting up our reference-plasticity 
  # (long term memory formation)
  hr = LearningTracker(
    plasticity_root=1.8, # for slower learning pick lower numbers
    plasticity_denominator_offset=1.0,
  )

.. _recipes-plasticity-curve-model-discovery:

Plasticity Curve Model Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use information from one learning tracker session to build a better set of goals
for another:

.. code-block:: python

  from datetime import datetime
  from repetition import LearningTracker

  day_offset_from_epoch_and_results = [ 
      [0,    0.81, 1.75, 3.02, 4.8,  8.33, 10.93, 16.00, 23.00, 29.00],
      [0.40, 0.44, 0.64, 0.84, 0.83, 0.89,  1.00,  0.99, 0.99,   1.00],
  ]

  # create a learning tracker with arbitrary default parameters
  lt_arbitrary = LearningTracker(
    epoch=datetime.now(),
  )

  # mimic a full training session
  for d, r in \
    zip(*day_offset_from_epoch_and_results):
    # r: result
    # d: days since training epoch
    lt_arbitrary.learned(result=r, when=d)

  # get better initial model parameters based on previous
  # experience with the student
    
  # to get the discovered parameter from a previous training session,
  # pre-pend 'discovered' in front of the parameter name,
  # and call this word like a function
  bpr = lt_arbitrary.discovered_plasticity_root()
  bpdo = lt_arbitrary.discovered_plasticity_denominator_offset()

  # a better learning tracker
  lt_better_fit = LearningTracker(
    epoch=datetime.now(),
    plasticity_root=bpr,
    plasticity_denominator_offset=bpdo,
  )

You can read more about model learning :ref:`here.<quickstart-building-a-better-initial-student-model>`

.. _recipes-feedback:

Feedback
-------

To give feedback to a learning tracker:

.. code-block:: python

  from datetime import datetime
  from datetime import timedelta
  from repetition import LearningTracker
  epoch = datetime.now()
  lt = LearningTracker(epoch=epoch)

  result = 0.33 # thirty three percent recollection
  day_offset_from_epoch = 0.80
  lt.learned(result=result, when=day_offset_from_epoch)

  # the learning tracker can also accept datetime inputs
  result = 0.45 # forty five percent recollection
  day_offset_from_epoch = 1.20
  lt.learned(
    result=result, 
    when=epoch + timedelta(days=day_offset_from_epoch)
  )

.. _recipes-control:

Control
-------

.. _recipes-what-is-being-controlled-anyway?:

What is Being Controlled Anyway?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a control graph taken from the :ref:`quickstart <quick-start>`.  We will
use it to talk about the different parts of the control system:

.. image:: _static/quickstart_better_fit.svg
    :target: _static/quickstart_better_fit.pdf
    :align: center

For a control system to work, we need to know a few things. We need to know what
we want, this is called the reference, and we need to know what actually
happened, this is called feedback.

As previously mentioned, the reference is the dark blue line representing the
plasticity curve in the first graph.


With the spaced algorithm, the feedback data is used to build a second
plasticity curve, which can be seen in the second and forth boxes as a light
blue line. This second plasticity curve is the observed plasticity curve. It is
a model of how the student is actually learning the idea being tracked. 

The plasticity curves are built using this equation:

.. code-block:: python

  #                     x**(1.0/plasticity_root)
  # pl = ---------------------------------------------------------
  #      (x+plasticity_denominator_offset)**(1.0/plasticity_root)

While spaced is making observed plasticity curve, it puts extra emphasis on the
most recent data provided by the student. This is because a student's recent
understanding should outweigh their previous ignorance. The observed plasticity
curve is constructed using the curve_fit method of the ``scipy.optimize``
library. The curve_fit method finds values of ``plasticity_root`` and
``plasticity_denominator_offset`` that minimize the difference between the
observed plasticity curve, and the feedback provided by the student.

After a training session, the spaced algorithm shifts the blue reference line
such that it intersects with the light blue observed plasticity curve, then
re-draws the forgetting curves, which causes projections onto the timeline which
represent the updated schedule (This is a kind of `feedforward loop
<https://en.wikipedia.org/wiki/Feed_forward_(control)>`_). Well, this is very
close to what happens, we are missing one little step.

When we begin to track a student's learning, we make assumptions about how fast
they forget. But we don't actually know how fast they will forget something, so
the spaced algorithm tries to match the forgetting curves used by the control
graph, to what they look like in the student's observed graph.

Specifically, the ``fdecaytau`` and the ``fdecay0`` are adjusted, so that the forgetting
curves in the control graph fall at similar rates as they do in the observed
graph. It will be very difficult to tune these control parameters, since we have
no way of actually modeling the system. (the interaction between the student's
mind, their environment, the training material, the context in which they
train...) For now, we will be happy with making mistakes in the right direction.

.. _recipes-changing-the-control-parameters:

Changing the Control Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To change the control system parameters:

.. code-block:: python

  from repetition import LearningTracker
  from datetime import datetime
  from datetime import timedelta
  
  start_time = datetime.now()
  x, y = \
      [0,    0.81, 1.75, 3.02, 4.8,  8.33],
      [0.40, 0.44, 0.64, 0.84, 0.83, 0.89]

  # you can feed custom control parameters to the learning tracker
  lt = LearningTracker(
    epoch=start_time,
    fdecaytau_kp=1.0,  # kp for fdecay control (PID)
    fdecaytau_ki=0.1,  # ki " "
    fdecaytau_kd=0.04, # kd " "
    fdecay0_kp=1.0,    # kp for fdecaytau_kp (PID)
    fdecay0_ki=0.1,    # ki " "
    fdecay0_kd=0.03,   # kd " "
  )

.. _recipes-queries:

Queries
-------

To make a query about the performance of a student based on their learning
tracker:

.. code-block:: python

  from datetime import datetime
  from repetition import pp
  from repetition import LearningTracker

  training_epoch = datetime.now()

  # create a learning tracker
  lt = LearningTracker(epoch=training_epoch)

  # give our learning tracker some feedback
  for d, r in zip( 
      [0,    0.8,  1.75, 3.02, 4.8,  7.33],
      [0.40, 0.44, 0.64, 0.76, 0.83, 0.89],
    ):
    # r: result
    # d: days since training epoch
    lt.learned(result=r, when=d)

  predicted_result_at_day_eight = lt.predict_result(
    moment=training_epoch+timedelta(days=8.00),
    curve=1)

To read more about making predictions look :ref:`here.<quickstart-predicting-future-results>`

.. _recipes-get-a-useful-set-of-datetimes:

Get a useful set of Datetimes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Get a useful range of datetimes from a learning tracker (used for graphing with
queries).

.. code-block:: python

  from datetime import datetime
  from repetition import LearningTracker

  # create a learning tracker
  lt = LearningTracker(
    epoch=datetime.now(),
  )

  # give our learning tracker some feedback
  for d, r in zip( 
      [0,    0.8,  1.75, 3.02, 4.8,  7.33],
      [0.40, 0.44, 0.64, 0.76, 0.83, 0.89],
    ):
    # r: result
    # d: days since training epoch
    lt.learned(result=r, when=d)

  # get a set of datetimes
  useful_range_of_datetimes = \
    lt.range_for(curve=1, range=10, day_step_size=0.5)

.. _recipes-serialization:

Serialization
-------------
The learning tracker has a custom pickler so you can serialize it.

.. _recipes-pickle-a-learningtracker:

Pickle a LearningTracker into a Bytestream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To pickle a learning tracker into a bytestream:

.. code-block:: python

  import pickle
  from datetime import datetime
  from repetition import LearningTracker

  # create a learning tracker
  lt = LearningTracker(epoch=datetime.now())

  # create a byte stream
  byte_stream = pickle.dumps(lt)

  # re-create the object from the bytestream (should be small)
  # (plural for byte stream)
  unpickled_learning_tracker = pickle.loads(byte_stream)

  hdl, _ = unpickled_learning_tracker.plot_graphs() 
  hdl.plot_graphs(stop=43) # plots the graph

  # close the plot to save memory
  hdl.close()

.. _recipes-pickle-a-learningtracker-into-a-file:

Pickle a LearningTracker into a File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To pickle a learning tracker to a file:

.. code-block:: python

  import pickle
  from datetime import datetime
  from repetition import LearningTracker

  # create a learning tracker
  lt = LearningTracker(epoch=datetime.now())

  # pickle the learning tracker
  with open('leaning_tracker.pickle', 'wb') as f:
    pickle.dump(lt, f, pickle.HIGHEST_PROTOCOL)

  # to turn the 'learning_tracker.pickle' file back into
  # an object
  with open('learning_tracker.pickle', 'rb') as f:
    # (no-plural for byte stream)
    unpickled_learning_tracker = pickle.load(f)

.. _recipes-model-learning:

Model Learning
--------------
:ref:`read about model learning here<quickstart-building-a-better-initial-student-model>`

.. raw:: html

  <a class="reference internal" href="quickstart.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="how_it_works.html"><span class="std std-ref">next</span></a>

.. _umlet: http://www.umlet.com/
.. _umletino: http://www.umlet.com/umletino/umletino.html
.. _OMG: https://en.wikipedia.org/wiki/Object_Management_Group
.. _mandala: https://en.wikipedia.org/wiki/Sand_mandala
.. _drawit: https://github.com/vim-scripts/DrawIt
