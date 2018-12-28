.. _how_it_works:

  *Too much of what is called 'education' is little more than an expensive
  isolation from reality.* 
  
  -- Thomas Sowell

How it Works
=======

.. image:: _static/spaced.svg
    :target: _static/spaced.pdf
    :align: center

.. _how_it_works-files:

Files
-----

The ``spaced`` algorithm is made up of five different packages/files:

1. ``repetition.py`` - learning tracker features
2. ``graph.py`` - provides graphing features
3. ``animate.py`` - provides ffmpeg video features
4. ``pid.py`` - provides a proportional-integral-differential controller
5. ``cli.py`` - wrapper for the pytest package

The ``repetition`` package contains the majority of the scheduling code.  It's
class structure is primarily broken into the ``SpaceRepetitionReference``, the
``SpaceRepetitionFeedback`` and the ``SpaceRepetitionControl`` classes.

The common features and interface shared by they classes come from them inheriting from the
``SpaceRepetition`` class.

.. _how_it_works-spacerepetitionreference:

SpaceRepetitionReference
------------------------

The ``SpaceRepetitionReference`` is used to construct the system goals.  It sets
up the initial forgetting curves and the reference plasticity curve.  This is
done using an exponential decay to create a set of less and less aggressive
exponential decay curves.  The nature of these curves can be tuned using the
``fdecay0`` and ``fdecaytau`` input parameters.

The stickleback looking part of the reference graph, which can be seen below, is
made by restarting an exponential decay at the intersection of the forgetting
curve and the reference plasticity curve.

.. image:: _static/quickstart_reference.svg
    :target: _static/quickstart_reference.pdf
    :align: center

The reference plasticity curve, represented by the dark blue line above, is a
ratio of two different exponential functions:

.. code-block:: python

  #                    x**(1.0/plasticity_root)
  # r = ---------------------------------------------------------
  #     (x+plasticity_denominator_offset)**(1.0/plasticity_root)

The plasticity curve can be tuned using the ``plasticity_root`` and the
``plasticity_denominator_offset`` parameters.

.. _how_it_works-spacerepetitionfeedback:

SpaceRepetitionFeedback
-----------------------

The ``SpaceRepetitionFeedback`` class is used for accepting student feedback and
generating the new observed-plasticity curve.  This curve is built using the
``scipy.optimize`` ``curve_fit`` api.  It tries to find the ``plasticity_root``
and ``plasticity_denominator_offset`` parameters that draw a line that has the
same shape of the reference plasticity curve but fits the feedback data provided
by the student.  When fitting this curve it places an emphasis on the most
recently observed data.

.. _how_it_works-spacerepetitioncontrol:

SpaceRepetitionControl
----------------------

The ``SpaceRepetitionControl`` class generates an error signal by subtracting
the reference plasticity curve from the observed plasticity curve.  Then it
feeds this error signal into two PID controllers to change the reference
forgetting curve parameters, ``fdecay0`` and ``fdecaytau``, to look more like
those generated from the student's feedback.  To read about how to change these
control parameters look :ref:`here.<recipes-control>`

.. image:: _static/quickstart_control_after_two_events.png
    :target: _static/quickstart_control_after_two_events.pdf
    :align: center

The ``SpaceRepetitionControl`` class then finds the intersection between the
reference plasticity curve and the observed plasticity curve providing a
starting point to place a new reference plasticity curve and a new set of
forgetting curves riding on its back.

.. _how_it_works-learningtracker:

LearningTracker
---------------

The ``LearningTracker`` class aggregates the reference, feedback and control
features into one easy-to-use class.  Any tuning parameter that can be fed into
any of the other classes can be fed into it, and it will ensure that this
parameter is passed on properly.

The majority of the features offered by the ``LearningTracker`` are described in
the :ref:`recipes<recipes>` section.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. raw:: html

  <a class="reference internal" href="recipes.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="testing.html"><span class="std std-ref">next</span></a>

.. _umlet: http://www.umlet.com/
.. _umletino: http://www.umlet.com/umletino/umletino.html
.. _OMG: https://en.wikipedia.org/wiki/Object_Management_Group
.. _mandala: https://en.wikipedia.org/wiki/Sand_mandala
.. _drawit: https://github.com/vim-scripts/DrawIt
