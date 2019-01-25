"""pid

The pid module provides a PID class which is an interface to a closed-loop,
proportional-integral-derivative controller.

**Example(s)**:
  Here is how you can build and use the PID class:

  .. code-block:: python

    from pid import PID

    controller = PID(
      kp=0.8,
      ki=0.1,
      kd=0.2
    )

    # imagine the following code is inside of a loop that is
    # contantly monitoring the thing you are trying to control
    reference = 11
    observed  = 9
    current_control_point = 10
    current_control_point -= controller.feedback(reference-observed)

To read more about pid controllers see `this link <https://en.wikipedia.org/wiki/PID_controller>`_.

"""
import numpy as np
class PID(object):

  PROPORTIONAL_CLAMP   = 8
  INTEGRATOR_CLAMP     = 8
  DIFFERENTIATOR_CLAMP = 8

  def __init__(self, kp, ki, kd, *args, **kwargs):
    self.kp         = kp
    self.ki         = ki
    self.kd         = kd
    self.error      = [0, 0]
    self.clamp_p    = PID.PROPORTIONAL_CLAMP
    self.clamp_i    = PID.INTEGRATOR_CLAMP
    self.clamp_d    = PID.DIFFERENTIATOR_CLAMP
    self.integrator = 0

  def reset(self):
    self.error      = [0, 0]
    self.integrator = 0

  def feedback(self, error):
    result         = 0
    proportional   = 0
    differentiator = 0
    sign           = 0

    self.error[0] = error

    proportional = error * self.kp
    sign         = np.sign(proportional)
    proportional = abs(proportional) if abs(proportional) < self.clamp_p else self.clamp_p
    proportional *= sign

    self.integrator  = self.integrator + error * self.ki
    sign             = np.sign(self.integrator)
    self.integrator  = abs(self.integrator) if abs(self.integrator) < self.clamp_i else self.clamp_i
    self.integrator *= sign

    differentiator  = (self.error[0] - self.error[1]) * self.kd
    sign            = np.sign(differentiator)
    differentiator  = abs(differentiator) if abs(differentiator) < self.clamp_d else self.clamp_d
    differentiator *= sign

    self.error[1] = error

    result  = proportional
    result += self.integrator
    result += differentiator

    return result


