  *If you tell the truth, you don't have to remember anything.* 
  
  -- Mark Twain

.. _introduction-introduction:

Introduction
============

``Spaced`` is a Python library that recommends when a student should study.  It
provides cognitive training schedules that over time will adapt to the
behaviours of the student.

``Spaced`` suggests training dates which gradually become more dispersed from
each other over time.  If you train in such a way, you end up remembering more
than you would if you put the same amount of attention into your studies over a
short duration (say, through intense cramming sessions).

This `spacing effect <https://en.wikipedia.org/wiki/Spacing_effect>`_ was
discovered by the German psychologist `Hermann Ebbinghaus
<https://en.wikipedia.org/wiki/Hermann_Ebbinghaus>`_ in the 1880s.

The algorithm used by ``spaced``, is a mash-up of a spaced-learning-algorithm
with a `simple control system <https://en.wikipedia.org/wiki/PID_controller>`_
and a lightweight machine learning routine.  It provides training date
suggestions, which become less frequent over time while being responsive to what
a student actually does.  

You can use the results of one training session to feed in better expectations
about how a student remembers and how they forget for their next session.  In
this way, the ``spaced`` schedules are adaptive and will become more useful for
the student as they interact with the system; ``spaced`` helps
the student learn, and it learns from the student as they engage with their
education.

The ``spaced`` package :ref:`can provide graphs <recipes-diagrams>` and
:ref:`video feedback to give insights <recipes-animating-a-learning-tracker>` on
how a student is responding to their training over time.  This is useful if you
want to get an intuitive feel about the relationship between a student's
attention and how they are responding to their training.  These graphs may
provide insights into the quality of the material, how distracted the student is
or isn't, how fast they remember over the longer term and how fast they forget
over the short-term.

If you don't need to drill down to this level of detail, you can used the
``spaced`` algorithm in a less computer-memory-intensive way, by just making predictions
about a memory and getting the next schedule time for training.

The ``spaced`` package can track a memory indefinitely.

.. raw:: html

  <a class="reference internal" href="installation.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="quickstart.html"><span class="std std-ref">next</span></a>

.. toctree::
   :maxdepth: 2
   :caption: Contents:


