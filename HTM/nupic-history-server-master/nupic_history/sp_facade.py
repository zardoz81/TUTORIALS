import uuid
import multiprocessing
import time

import numpy as np

from nupic_history import SpSnapshots as SNAPS
from nupic_history.utils import compressSdr
from nupic.math import topology


class SpFacade(object):

  def __init__(self, sp, ioClient, iteration=None):
    """
    A wrapper around the HTM Spatial Pooler that can save SP state to Redis for
    each compute cycle. Adds a "save=" kwarg to compute().

    :param sp: Either an instance of Spatial Pooler or a string model id
    :param ioClient: Instantiated IO client
    :param iteration: what iteration to resurrect the SP at
    """
    self._ioClient = ioClient
    if isinstance(sp, basestring):
      # Loading SP by id from IO.
      self._id = sp
      # Get the latest by default.
      if iteration is None:
        iteration = ioClient.getMaxIteration(self._id)
      self._iteration = iteration
    else:
      # New facade using given fresh SP.
      self._sp = sp
      self._id = str(uuid.uuid4()).split('-')[0]
      self._input = self._getZeroedInput()
      self._activeColumns = self._getZeroedColumns()
      self._iteration = sp.getIterationNum()
    self._state = None
    self._potentialPools = None
    self._inhibitionMasks = None


  def __str__(self):
    return "SP {} has seen {} iterations".format(
      self.getId(), self.getIteration()
    )


  def save(self):
    ioClient = self._ioClient
    id = self.getId()
    iteration = self.getIteration()
    ioClient.saveEncoding(self._input, id, iteration)
    ioClient.saveActiveColumns(self._activeColumns, id, iteration)
    ioClient.saveSpatialPooler(self._sp, id, iteration)


  def load(self):
    ioClient = self._ioClient
    id = self.getId()
    iteration = self.getIteration()
    print "Loading SP {} at iteration {}".format(id, iteration)
    self._sp = ioClient.loadSpatialPooler(
      id, iteration=iteration
    )
    if iteration == 0:
      self._input = self._getZeroedInput()
      print "loading zeroed AC"
      self._activeColumns = self._getZeroedColumns()
    else:
      self._input = ioClient.loadEncoding(id, iteration)
      print "loading AC from disk"
      self._activeColumns = ioClient.loadActiveColumns(id, iteration)


  def getId(self):
    """
    Unique id used for storage and retrieval in Redis.
    :return: id
    """
    return self._id


  def getIteration(self):
    """
    Represents the current iteration of data the SP has seen.
    :return: int
    """
    if hasattr(self, "_sp"):
      return self._sp.getIterationNum()
    else:
      return self._iteration


  def getInput(self):
    """
    :return: last seen input encoding
    """
    return self._input


  def getParams(self):
    """
    Utility to collect the SP params used at creation into a dict.
    :return: [dict] parameters
    """
    sp = self._sp
    if sp is None:
      params =self._ioClient.getSpParams(self.getId())
    else:
      params = {
        "numInputs": sp.getNumInputs(),
        "numColumns": sp.getNumColumns(),
        "columnDimensions": np.asarray(sp.getColumnDimensions(), dtype="uint32"),
        "numActiveColumnsPerInhArea": sp.getNumActiveColumnsPerInhArea(),
        "potentialPct": sp.getPotentialPct(),
        "globalInhibition": sp.getGlobalInhibition(),
        "localAreaDensity": sp.getLocalAreaDensity(),
        "stimulusThreshold": sp.getStimulusThreshold(),
        "synPermActiveInc": sp.getSynPermActiveInc(),
        "synPermInactiveDec": sp.getSynPermInactiveDec(),
        "synPermConnected": sp.getSynPermConnected(),
        "minPctOverlapDutyCycle": sp.getMinPctOverlapDutyCycles(),
        "dutyCyclePeriod": sp.getDutyCyclePeriod(),
        "boostStrength": sp.getBoostStrength(),
      }
    return params


  def compute(self, encoding, learn=False, save=False, multiprocess=False):
    """
    Pass-through to Spatial Pooler's compute() function, with the addition of
    the save option.
    :param encoding: encoding to pass to the sp
    :param learn: whether sp will learn on this compute cycle
    """
    sp = self._sp
    columns = self._getZeroedColumns()
    encoding = np.asarray(encoding, dtype="uint32")

    start = time.time()
    sp.compute(encoding, learn, columns)
    end = time.time()
    print("\tSP compute took %g seconds" % (end - start))

    self._input = encoding
    self._activeColumns = columns
    self._state = None
    if save:
      if multiprocess:
        p = multiprocessing.Process(target=self.save)
        p.start()
      else:
        self.save()


  def getState(self, *args, **kwargs):
    """
    Returns the requested state of the spatial pooler. This will either get the
    state from the sp instance itself and cache it in memory for further calls,
    or if the facade is inactive, it will fetch the state from redis.

    This function can be used to retrieve any saved state snapshots from Redis
    for any iteration the SP has seen. It will work for active and inactive
    facades.

    For example, to get the column overlaps at iteration 10:

      sp.getState(Snapshots.OVERLAPS, iteration=10)

    Will return an array of column overlaps (each being an array).

    The following snapshots can be retrieved:

      Snapshots.INPUT = "input"
      Snapshots.POT_POOLS = "potentialPools"
      Snapshots.CON_SYN = "connectedSynapses"
      Snapshots.PERMS = "permanences"
      Snapshots.ACT_COL = "activeColumns"
      Snapshots.OVERLAPS = "overlaps"
      Snapshots.ACT_DC = "activeDutyCycles"
      Snapshots.OVP_DC = "overlapDutyCycles"
      Snapshots.INH_MASKS = "inhibitionMasks"

    :param args:
    :param kwargs:
    :return:
    """
    columnIndex = None
    if "columnIndex" in kwargs:
      columnIndex = kwargs["columnIndex"]

    if self._state is None:
      self._state = {}
    out = dict()
    for snap in args:
      if not SNAPS.contains(snap):
        raise ValueError("{} is not available in SP History.".format(snap))
      out[snap] = self._getSnapshot(
        snap, iteration=self.getIteration(), columnIndex=columnIndex
      )
    return out


  def delete(self):
    """
    Deletes all traces of this SP instance from Redis.
    """
    self._ioClient.delete(self.getId())


  def getNumColumns(self):
    """
    Pass-through to getParams()
    :return: [int] number of columns in the SP
    """
    return self.getParams()["numColumns"]


  def _getZeroedColumns(self, dtype=None):
    numCols = self.getParams()["numColumns"]
    zeros = np.zeros(shape=(numCols,))
    if dtype is None: dtype = "uint32"
    return np.asarray(zeros, dtype=dtype)


  def _getZeroedInput(self, dtype=None):
    numInputs = self.getParams()["numInputs"]
    zeros = np.zeros(shape=(numInputs,))
    if dtype is None: dtype = "uint32"
    return np.asarray(zeros, dtype=dtype)


  def _getSnapshot(self, name, iteration=None, columnIndex=None):
    # Use the cache if we can.
    if name in self._state and iteration == self._iteration:
      return self._state[name]
    else:
      funcName = "_conjure{}".format(name[:1].upper() + name[1:])
      func = getattr(self, funcName)
      _start = time.time()
      result = func(iteration=iteration, columnIndex=columnIndex)
      _end = time.time()
      print "\t\t{}: {} seconds".format(funcName, (_end - _start))
      self._state[name] = result
      return result


  # None of the "_conjureXXX" functions below are directly called. They are all
  # called via string name by the _getSnapshot function, depending on what type
  # of snapshot data is being requested. They are called "conjureXXX" because
  # we won't really be sure where the information is coming from until the
  # function is called. It could come from the SpFacade instance, the
  # SpatialPooler instance, or Redis. It depends on whether the facade is active
  # or not, and whether an iteration is specified. In all cases where an
  # iteration in the past is specified, Redis will be the data source.


  def _conjureInput(self, **kwargs):
    return compressSdr(self._input)


  def _conjureActiveColumns(self, **kwargs):
    return compressSdr(self._activeColumns)


  def _conjureOverlaps(self, **kwargs):
    return self._sp.getOverlaps().tolist()


  def _conjurePotentialPools(self, **kwargs):
    # These only need to be fetched from the SP once.
    if self._potentialPools:
      return self._potentialPools
    sp = self._sp
    out = []
    for colIndex in range(0, sp.getNumColumns()):
      columnPool = self._getZeroedInput()
      columnPoolIndices = []
      sp.getPotential(colIndex, columnPool)
      for i, pool in enumerate(columnPool):
        if np.asscalar(pool) == 1.0:
          columnPoolIndices.append(i)
      out.append(columnPoolIndices)
    self._potentialPools = out
    return out


  def _conjureConnectedSynapses(self, **kwargs):
    columns = []
    sp = self._sp
    zeros = self._getZeroedInput()
    for colIndex in range(0, sp.getNumColumns()):
      connectedSynapses = zeros.copy()
      sp.getConnectedSynapses(colIndex, connectedSynapses)
      columns.append(np.nonzero(connectedSynapses)[0].tolist())
    return columns


  def _conjurePermanences(self, **kwargs):
    out = []
    numColumns = self.getNumColumns()
    sp = self._sp
    zeros = self._getZeroedInput(dtype="float32")
    for colIndex in range(0, numColumns):
      perms = zeros.copy()
      sp.getPermanence(colIndex, perms)
      out.append(np.around(perms, decimals=2).tolist())
    return out


  def _conjureActiveDutyCycles(self, **kwargs):
    sp = self._sp
    dutyCycles = self._getZeroedColumns(dtype="float32")
    sp.getActiveDutyCycles(dutyCycles)
    return dutyCycles.tolist()


  def _conjureBoostFactors(self, **kwargs):
    sp = self._sp
    boostFactors = self._getZeroedColumns(dtype="float32")
    sp.getBoostFactors(boostFactors)
    return boostFactors.tolist()


  def _conjureOverlapDutyCycles(self, **kwargs):
    sp = self._sp
    dutyCycles = self._getZeroedColumns()
    sp.getOverlapDutyCycles(dutyCycles)
    return dutyCycles.tolist()


  def _conjureInhibitionMasks(self, **kwargs):
    # These only need to be fetched from the SP once.
    if self._inhibitionMasks:
      return self._inhibitionMasks
    out = []
    numColumns = self.getNumColumns()
    for colIndex in xrange(numColumns):
      out.append(self._getInhibitionMask(colIndex))
    self._inhibitionMasks = out
    return out


  def _getInhibitionMask(self, colIndex):
    sp = self._sp
    return topology.neighborhood(
      colIndex, sp.getInhibitionRadius(), sp.getColumnDimensions()
    ).tolist()

