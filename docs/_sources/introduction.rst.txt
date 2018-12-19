  *Keep my memory green* 
  
  -- Charles Dickens

.. _introduction-introduction:

Introduction
============
``Spaced`` is a Python library that tells you when to study.  It provides
adaptive cognitive training schedules.

``Spaced`` suggests training dates which gradually become more dispersed from
each other over time.  If you train in such a way, you end up remembering more
than you would if you put the same amount of attention into your studies over a
short duration (say, through intense cramming).

This `spacing effect <https://en.wikipedia.org/wiki/Spacing_effect>`_ was
discovered by the German psychologist `Hermann
Ebbinghaus <https://en.wikipedia.org/wiki/Hermann_Ebbinghaus>`_ in the 1880s.

The algorithm used by ``spaced``, is a mash-up of a spaced-learning-algorithm
with a `simple control system <https://en.wikipedia.org/wiki/PID_controller>`_ and
a light weight machine learning routine.  It provides training date suggestions,
which become less frequent over time while being responsive to what a student
actually does.  

You can use the results of one training session to feed in better expectations
about how a student remembers and how they forget for their next session.  In
this way, your schedule will become more useful for the student.

.. raw:: html

  <a class="reference internal" href="installation.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="quickstart.html"><span class="std std-ref">next</span></a>

.. toctree::
   :maxdepth: 2
   :caption: Contents:


