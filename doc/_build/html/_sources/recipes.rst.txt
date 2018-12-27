.. _recipes:

  *Simple things should be simple, complex things should be possible.* 
  
  -- Alan Kay

Recipes
=======

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

  lt = LearningTracker(
    epoch=datetime.new(),
    range=days_to_track
  )

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
      hdl, _ = lt.plot_graphs()

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
  hdl, _ = learning_tracker.reference.plot_graph()

  # to save this plot (usual api, since I can't imagine why a 
  # regular user would save the reference, but I have included it
  # as a recipe since I needed to do this to document this package
  hdl.ppd.plot.savefig("quickstart_reference.svg")
  hdl.ppd.plot.savefig("quickstart_reference.pdf")

  hdl.close()


.. image:: _static/quickstart_reference.svg
    :target: _static/quickstart_reference.pdf
    :align: center

.. _recipes-save-a-diagram:

Save a Diagram
^^^^^^^^^^^^^^
To save a diagram:

.. code-block:: python

  from datetime import datetime
  from space.repitition import LearningTracker
  lt = LearningTracker(
    epoch=datetime.new(),
  )

  hdl, _ = lt.plot_graphs()

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
  hdl, _ = lt.plot_graphs()

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
  lt_arbitrary = LearningTracker(
    epoch=datetime.now(),
  )

  for d, r in zip(*day_offset_from_epoch_and_results):
    # r: result
    # d: days since training epoch
    lt_arbitrary.learned(result=r, when=d)

  lt.animate(
    student="Name of Student",
    name_of_mp4="results/report_card.mp4",
    time_per_event_in_seconds=2.2)

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
  hr = LearningTracker(fdecaytau=1.00,  # ability to improve after a lesson, lower is better
                       fdecay0=0.9,      # seen it before? then pick lower numbers
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
  bf0 = lt_arbitrary.discovered_fdecay0()
  bft = lt_arbitrary.discovered_fdecaytau()

  # a better learning tracker
  lt_better_fit = LearningTracker(
    epoch=datetime.now(),
    fdecay0=bf0,
    fdecaytau=bft
  )

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

.. _recipes-feedback:

Feedback
-------

.. _recipes-control:

Control
-------

.. _recipes-queries:

Queries
-------

.. _recipes-reflection:

Reflection
----------

.. _recipes-serialization:

Serialization
-------------

.. _recipes-model-learning:

Model Learning
--------------

.. raw:: html

  <a class="reference internal" href="zero_to_one.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="how_it_works.html"><span class="std std-ref">next</span></a>

.. _umlet: http://www.umlet.com/
.. _umletino: http://www.umlet.com/umletino/umletino.html
.. _OMG: https://en.wikipedia.org/wiki/Object_Management_Group
.. _mandala: https://en.wikipedia.org/wiki/Sand_mandala
.. _drawit: https://github.com/vim-scripts/DrawIt
