.. _quick-start:

Quick Start
===========
To build a schedule, import a learning tracker and provide it with its initial
conditions:

.. code-block:: python

  from datetime import datetime
  from space.repitition import pp
  from space.repitition import LearningTracker

  days_to_track = 43
  lt = LearningTracker(
    epoch=datetime.new(),
    range=days_to_track
  )

Then ask your learning tracker for its schedule:

.. code-block:: python

  print("Scheduled as dates")
  pp(lt.schedule())

Something like this would appear in your terminal window:

.. code-block:: text 

  Scheduled as dates:
  [ datetime.datetime(2018, 12, 18, 6, 9, 22, 858534),
    datetime.datetime(2018, 12, 18, 17, 10, 15, 643548),
    datetime.datetime(2018, 12, 19, 3, 38, 30, 456706),
    datetime.datetime(2018, 12, 19, 14, 43, 41, 666247),
    datetime.datetime(2018, 12, 20, 3, 18, 28, 942419),
    datetime.datetime(2018, 12, 20, 18, 35, 2, 771460),
    datetime.datetime(2018, 12, 21, 14, 48, 26, 153166),
    datetime.datetime(2018, 12, 22, 21, 24, 41, 875893),
    datetime.datetime(2018, 12, 25, 9, 0, 36, 549251),
    datetime.datetime(2019, 1, 3, 9, 6, 30, 129684)]

This isn't that useful, since it has nothing to do with the student and their
ability, or the quality of their course material, or their environment.  It's
just an initial guess about what might work.

To get the learning tracker to perform a bit better, we need to give it some
data about your student's performance.

Let's build a feedback function that we can use to demonstrate how to make your
learning tracker respond to a student's memory:

.. code-block:: python

  def feedback():
  '''
  Here are 10 actual training moments with corresponding results.
  '''
  offset_in_days_from_introduction_to_idea = [
    0,
    0.80640320160080037,
    1.7523761880940469,
    3.0240120060030016,
    4.8074037018509257,
    7.3351675837918959,
    10.932966483241621,
    16.004002001000501,
    23.029014507253628,
    29.029014507253628
  ]
  # how well they could remember the thing prior to training on it
  # so as to get perfect recollection.
  results = [
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
  return [offset_in_days_from_introduction_to_idea, results]

This feedback function returns two lists: timing-information and results.
Each of these lists contains 10 items.

The timing information is a set of offset-in-days from when the training
started.  The start time for the training was set as your ``epoch`` parameter
of the LearningTracker initialization.

The numbers representing the results are in the range of 0 to 1.  A result of 0
means that the student has perfectly forgotten the tracked idea, while a 1 means they have
a perfect recollection of the idea.

Now that we have some feedback, let's express how this student learns to the
learning tracker.  The following code will simulate what would happen over the
span of the 43 days of training we are tracking:

.. code-block:: python

  # feed the learning tracker
  for index, (moment, result) in enumerate(zip(feedback())):

    # you would use this api to give your learning tracker its feedback
    lt.learned(result=result, when=moment)

After each training event, the learning tracker will respond with a new
schedule.  Since there is so much going on, ``spaced`` lets you create a kind
of report card, which is actually just a video.  To make a video, using the
``animate`` api:

.. code-block:: python

  lt.animate(
    student="Name of Student",
    name_of_mp4="results/report_card.mp4",
    time_per_event_in_seconds=2.2)

If you were to write this code, the results of this session would be used to
make a video in ``results/report_card.mp4``.  That video would look like this:

.. raw:: html

  <center>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/H8llYuwH5L0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </center>

Let's take a look at 

Now that you have completed a learning cycle, you can ask spaced for the model
parameters it has discovered so that it can make a better model.

.. raw:: html

  <a class="reference internal" href="introduction.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="zero_to_one.html"><span class="std std-ref">next</span></a>

.. toctree::
   :maxdepth: 2
   :caption: Contents:


