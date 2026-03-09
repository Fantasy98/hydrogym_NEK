from mpi4py import MPI

# Import built-in print to avoid recursion
<<<<<<< HEAD
_print = __builtins__['print'] if isinstance(__builtins__, dict) else __builtins__.print
=======
_print = __builtins__['print'] if isinstance(__builtins__,
                                             dict) else __builtins__.print
>>>>>>> upstream/cl/hydrogym_v1.0-OptSolverDeps


def print(s):
  """Print only from first process"""
  if MPI.COMM_WORLD.Get_rank() == 0:
    _print(s)


def is_rank_zero():
  """Is this the first process?"""
  return MPI.COMM_WORLD.Get_rank() == 0
<<<<<<< HEAD

=======
>>>>>>> upstream/cl/hydrogym_v1.0-OptSolverDeps
