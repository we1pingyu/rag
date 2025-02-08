class ComputationPolicyInterface:
  """
  Computation policy interface
  """
  def generation_loop_normal(self, this, evaluate):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()

  def generation_loop_overlap_single_batch(self, this, evaluate):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()
  
  def generation_loop_overlap_multi_batch(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()

  def generation_loop_debug_single_batch(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()
  
  def generation_loop_debug_multi_batch(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()
  
  def generation_loop_debug_normal(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()

