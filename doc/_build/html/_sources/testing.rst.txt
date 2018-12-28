  *Vox populi, vox humbug.*
  
  -- William Tecumseh Sherman

.. _testing-testing:

Testing
=======

To run the test code, create a virtual environment in the ``spaced`` directory:

.. code-block:: bash

  python -m venv venv
  . ./venv/bin/activate

Then install the package within this same directory as an editable package:

.. code-block:: bash

  pip install --editable .

This will add an ``entry_point`` command line tool which is based upon the
``Click`` python library.  You can use this command from your terminal while in the
``spaced`` directory.  

To test the package, type:

.. code-block:: bash

  space test  # wrapper for pytest

A lot of the testing involves looking at graphs and videos to see if the package
is behaving as expected, you can find these graphs as ``pdf`` files in the
``results`` directory.  Likewise, the test videos have an ``mp4`` extension and
can be found in the same ``results`` directory.

.. raw:: html

  <a class="reference internal" href="how_it_works.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <span class="inactive-link">next</span>


.. toctree::
   :maxdepth: 2
   :caption: Contents:

