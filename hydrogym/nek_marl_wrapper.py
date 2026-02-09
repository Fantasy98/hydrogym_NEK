"""Compatibility shim for the Nek MARL gym wrapper."""

from hydrogym.nek.wrapper import NekMARLGymWrapper, load_nek_config, mpi_split

__all__ = ["NekMARLGymWrapper", "load_nek_config", "mpi_split"]
